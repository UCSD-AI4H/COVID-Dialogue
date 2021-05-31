from collections import namedtuple
from tqdm import tqdm, trange
import random
import os

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModelWithLMHead, AutoTokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from .gpt2_utils import GPT2LMHeadModelMultiDevicesWrapper


TextDataExample = namedtuple('TextDataExample', ['dialogue', 'tokens', 'mask'])


class DialoGPT:
    def __init__(self, model_size):
        self._tokenizer = AutoTokenizer.from_pretrained(
            f"microsoft/DialoGPT-{model_size}")

        if model_size in ['medium', 'large']:
            devices = [f'cuda:{i % torch.cuda.device_count()}' for i in
                       range(2 if model_size == 'medium' else 4)]
            self._model = GPT2LMHeadModelMultiDevicesWrapper(
                model_size, devices=devices)
        else:
            self._model = AutoModelWithLMHead.from_pretrained(
                f"microsoft/DialoGPT-{model_size}").to('cuda')

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}
        self._eval_steps = None
        self._log_dir = None
        self._log_file = None
        self._best_dev_loss = None

    def creat_log_dir(self, eval_steps, label):
        self._log_dir = f'{label}_training_logs'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, 'generations'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path, map_location='cuda'))
        print(f'Model {path} loaded.')

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def load_data(self, split, dialogues, max_length=1024):
        self._dataset[split] = []
        for dialogue in dialogues:
            tokens, mask = [], []
            for i, utterance in enumerate(dialogue):
                token_ids = self._tokenizer.encode(
                    utterance + self._tokenizer.eos_token)
                tokens.extend(token_ids)
                mask.extend([i % 2] * len(token_ids))
            tokens = tokens[:max_length]
            mask = mask[:max_length]

            self._dataset[split].append(TextDataExample(
                dialogue=dialogue, tokens=tokens, mask=mask))

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset
        self._model.train()

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='Training Epoch'):
            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()
            for example in batch:
                loss = self._get_loss(example) / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            self._global_step += 1
            if self._global_step % self._eval_steps == 0:
                self.gen_log()

    def evaluate(self):
        assert 'dev' in self._dataset
        self._model.eval()

        loss_list = []
        for example in self._dataset['dev']:
            with torch.no_grad():
                loss = self._get_loss(example)
            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def _get_loss(self, example):
        inputs = torch.tensor([example.tokens]).to(device='cuda')
        logits = self._model(inputs)[0]

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = inputs[:, 1:].contiguous()
        shift_masks = torch.tensor(
            example.mask[1:], dtype=torch.float).to('cuda')

        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(
            input=shift_logits.view(-1, shift_logits.size(-1)),
            target=shift_labels.view(-1))

        return torch.sum(loss * shift_masks) / torch.sum(shift_masks)

    def generate(self, chat_history):
        history_ids = []
        for sent in chat_history:
            history_ids.extend(
                self._tokenizer.encode(sent + self._tokenizer.eos_token))
        history_ids = torch.tensor([history_ids]).to('cuda')

        gen_ids = self._model.generate(
            history_ids,
            num_beams=10,
            length_penalty=2.0,
            max_length=min(1024, history_ids.shape[-1] + 200),
            min_length=history_ids.shape[-1] + 50,
            no_repeat_ngram_size=3,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        return self._tokenizer.decode(
            gen_ids[0, history_ids.shape[-1]:], skip_special_tokens=True)

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        self._log_file.flush()

        generation_file = open(
            f'{self._log_dir}/generations/step{self._global_step}.txt', 'w')
        for i in range(20):
            chat_history = self._dataset['dev'][i].dialogue[:-1]
            truth_text = self._dataset['dev'][i].dialogue[-1]

            with torch.no_grad():
                gen_text = self.generate(chat_history)

            print('CHAT_HISTORY:\n', file=generation_file)
            for u in chat_history:
                print('>>>', u, file=generation_file)
            print('-' * 50, f'\nGENERATION: {gen_text}\n', '-' * 50,
                  file=generation_file)
            print('-' * 50, f'\nTRUTH: {truth_text}\n', '=' * 50, '\n\n',
                  file=generation_file)
            generation_file.flush()

    @property
    def datasets(self):
        return self._dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()