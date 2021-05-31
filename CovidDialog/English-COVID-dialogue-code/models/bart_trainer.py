from collections import namedtuple
import random
from tqdm import tqdm, trange
import os

import torch
from torch.nn import functional as F

from fairseq.data.data_utils import collate_tokens
from fairseq.sequence_generator import SequenceGenerator

from transformers import AdamW, get_linear_schedule_with_warmup

from .bart_utils import BARTMultiGPUWrapper
from .bart_utils import label_smoothed_nll_loss


LIL_BATCH_SIZE = 1


TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'src_tokens', 'tgt_tokens'])


class BARTTrainer:
    def __init__(self, init='bart.large'):
        self._model = BARTMultiGPUWrapper(model_name=init)

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}

        self._log_dir = None
        self._eval_steps = None
        self._log_file = None
        self._best_dev_loss = None

    def create_training_log(self, eval_steps, label):
        self._log_dir = f'{label}_training_logs'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'generations'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

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

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path, map_location='cuda'))
        print(f'Model {path} loaded.')

    def load_data(self, split, src_texts, tgt_texts, src_max_len, tgt_max_len):
        assert len(src_texts) == len(tgt_texts)

        self._dataset[split] = []
        for src_text, tgt_text in tqdm(zip(src_texts, tgt_texts),
                                       total=len(src_texts),
                                       desc=f'loading {split} data'):
            src_tokens = self._model.encode(src_text)[-src_max_len:]
            tgt_tokens = self._model.encode(tgt_text)[:tgt_max_len]

            self._dataset[split].append(TextPairData(
                src_text=src_text,
                tgt_text=tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens))

    def train_epoch(self, batch_size, label_smooth_epsilon):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._model.split_to_gpus(n_gpus=min(2, torch.cuda.device_count()))
            self._model.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for j in range(0, len(batch), LIL_BATCH_SIZE):
                lil_batch = batch[j:j + LIL_BATCH_SIZE]

                src_lengths = torch.tensor(
                    [len(t.src_tokens) for t in lil_batch])
                src_tokens = collate_tokens(
                    [t.src_tokens for t in lil_batch],
                    pad_idx=self._model.dictionary.pad())
                tgt_tokens = collate_tokens(
                    [t.tgt_tokens for t in lil_batch],
                    pad_idx=self._model.dictionary.pad())

                loss = self._get_label_smoothed_nll_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    epsilon=label_smooth_epsilon)
                loss = loss * len(lil_batch) / batch_size

                if torch.isnan(loss):
                    print('warning: nan loss')
                    print(f'tgt_text: {lil_batch[0].tgt_text}')
                else:
                    loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            self._global_step += 1
            if self._global_step % self._eval_steps == 0:
                self.gen_log()

    def evaluate(self):
        assert 'dev' in self._dataset
        self._model.split_to_gpus(n_gpus=1)
        self._model.eval()

        loss_list = []
        for i in trange(0, len(self._dataset['dev']), LIL_BATCH_SIZE,
                        desc='Evaluating on Dev Set'):
            batch = self._dataset['dev'][i:i + LIL_BATCH_SIZE]

            src_lengths = torch.tensor(
                [len(t.src_tokens) for t in batch])
            src_tokens = collate_tokens(
                [t.src_tokens for t in batch],
                pad_idx=self._model.dictionary.pad())
            tgt_tokens = collate_tokens(
                [t.tgt_tokens for t in batch],
                pad_idx=self._model.dictionary.pad())

            with torch.no_grad():
                loss = self._get_label_smoothed_nll_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    epsilon=0.)

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, src_sents):
        self._model.split_to_gpus(n_gpus=1)
        self._model.eval()

        src_text = src_sents[0]
        generator = SequenceGenerator(
            tgt_dict=self._model.dictionary,
            max_len_b=200,
            min_len=50,
            beam_size=10,
            len_penalty=2.,
            no_repeat_ngram_size=3)

        src_tokens = self._model.encode(src_text)
        if src_tokens.shape[0] > 1024:
            src_tokens = src_tokens[-1024:]

        outputs = generator.generate(
            models=[self._model.model],
            sample={'net_input': {
                'src_tokens': src_tokens.unsqueeze(0).to('cuda'),
                'src_lengths': torch.tensor([len(src_tokens)]).to('cuda')
            }},
            bos_token=self._model.dictionary.bos())

        return [self._model.decode(outputs[0][0]['tokens'].cpu())]

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

        for example in self._dataset['dev']:
            gen_text = self.generate([example.src_text])[0]

            print('SOURCE:\n', example.src_text, '\n', '-' * 50, '\n',
                  'GENERATION:\n', gen_text, '\n', '-' * 50, '\n',
                  'TARGET:\n', example.tgt_text, '\n', '=' * 100, '\n\n\n',
                  file=generation_file)
            generation_file.flush()

    def _get_label_smoothed_nll_loss(
            self, src_lengths, src_tokens, tgt_tokens, epsilon):
        logits, extra = self._model(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=tgt_tokens)

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        smoothed_loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(shift_logits, dim=-1),
            target=shift_labels,
            epsilon=epsilon,
            ignore_index=self._model.dictionary.pad())

        return smoothed_loss

    @property
    def dataset(self):
        return self._dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()
