from collections import namedtuple
import random
from tqdm import tqdm, trange
import os

import torch
from torch.nn import functional as F
from torch.nn import Parameter

from fairseq.data.data_utils import collate_tokens
from fairseq.sequence_generator import SequenceGenerator

from transformers import AdamW, get_linear_schedule_with_warmup, BartTokenizer, BartForConditionalGeneration, \
    PreTrainedTokenizer
from transformers.modeling_bart import shift_tokens_right

from .bart_utils import BARTMultiGPUWrapper
from .bart_utils import label_smoothed_nll_loss

from typing import Tuple

LIL_BATCH_SIZE = 1
MLM_PROBABILITY = 0.15
SRC_MAX_LEN = 512
NET_WEIGHT = 1

SSL_MODEL_NAME = 'facebook/bart-large'

TextPairData = namedtuple('TextPairData', [
    'src_text', 'tgt_text', 'texts', 'src_tokens', 'tgt_tokens'])

tokenizer = BartTokenizer.from_pretrained(SSL_MODEL_NAME)


class BARTTrainer:
    def __init__(self, init='bart.large', shared_training='decoder'):
        self._model = BARTMultiGPUWrapper(model_name=init)

        # SSL model
        self._ssl_model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=SSL_MODEL_NAME)
        self._ssl_model=self._ssl_model.cuda()

        if shared_training == 'encoder':
            share_bart_encoder_layers(self._model, self._ssl_model)
            print('Dialog generation task and SSL task are using the same BART encoder.')
        else:
            share_bart_decoder_layers(self._model, self._ssl_model)
            print('Dialog generation task and SSL task are using the same BART decoder.')

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
                      weight_decay, adam_epsilon, shared_training):

        if shared_training == 'encoder':
            parameters = list(self._model.named_parameters()) + list(self._ssl_model.model.decoder.named_parameters())
        else:
            parameters = list(self._model.named_parameters()) + list(self._ssl_model.model.encoder.named_parameters())

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in parameters
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in parameters
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
                texts=src_text+' '+tgt_text,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens))

    def train_epoch(self, batch_size, label_smooth_epsilon, weight, text_type):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        print_train_loss_ssl = 0.0
        print_train_loss = 0.0
        print_train_loss_net = 0.0
        num = 0
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._model.split_to_gpus(n_gpus=min(2, torch.cuda.device_count()))
            self._model.train()
            self._ssl_model.train()

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

                loss_net = self._get_label_smoothed_nll_loss(
                    src_lengths=src_lengths,
                    src_tokens=src_tokens,
                    tgt_tokens=tgt_tokens,
                    epsilon=label_smooth_epsilon)

                # SSL training
                if text_type == 0:
                    texts = [t.texts for t in lil_batch]
                    ssl_batch = tokenizer.batch_encode_plus(texts, padding=True, max_length=SRC_MAX_LEN,
                                                            truncation=True, return_tensors='pt')

                    ssl_input_ids, ssl_label_ids = mask_tokens(ssl_batch.input_ids, tokenizer, MLM_PROBABILITY)
                    ssl_decoder_input_ids = shift_tokens_right(ssl_batch.input_ids, self._ssl_model.config.pad_token_id)
                elif text_type == 1:
                    ssl_input_ids, ssl_label_ids = mask_tokens(src_tokens, tokenizer, MLM_PROBABILITY)
                    ssl_decoder_input_ids = shift_tokens_right(src_tokens, self._ssl_model.config.pad_token_id)
                else:
                    ssl_input_ids, ssl_label_ids = mask_tokens(tgt_tokens, tokenizer, MLM_PROBABILITY)
                    ssl_decoder_input_ids = shift_tokens_right(tgt_tokens, self._ssl_model.config.pad_token_id)

                ssl_input_ids, ssl_label_ids, ssl_decoder_input_ids = ssl_input_ids.cuda(), ssl_label_ids.cuda(), ssl_decoder_input_ids.cuda()

                ssl_outputs = self._ssl_model(
                    input_ids=ssl_input_ids,
                    decoder_input_ids = ssl_decoder_input_ids,
                    labels=ssl_label_ids,
                )

                loss_ssl = ssl_outputs[0]
                print_train_loss_ssl += loss_ssl
                print_train_loss_net += loss_net
                num += 1

                # Total training loss
                loss = (loss_net*NET_WEIGHT + loss_ssl*weight) * len(lil_batch) / batch_size
                print_train_loss += loss * batch_size

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

        print("net training loss:", print_train_loss_net.item()/num)
        print("ssl training loss:", print_train_loss_ssl.item()/num)
        print("total training loss:", print_train_loss.item()/num)

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
        if src_tokens.shape[0] > SRC_MAX_LEN:
            src_tokens = src_tokens[-SRC_MAX_LEN:]

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
        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}')

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


def share_bart_encoder_layers(net, ssh):
    ssh.model.shared.weight = net._interface.model.encoder.embed_tokens.weight
    ssh.model.encoder.embed_positions.weight = net._interface.model.encoder.embed_positions.weight
    ssh.model.encoder.layernorm_embedding = net._interface.model.encoder.layernorm_embedding

    for i in range(0,12):
        ssh.model.encoder.layers[i].self_attn.k_proj = net._interface.model.encoder.layers[i].self_attn.k_proj
        ssh.model.encoder.layers[i].self_attn.v_proj = net._interface.model.encoder.layers[i].self_attn.v_proj
        ssh.model.encoder.layers[i].self_attn.q_proj = net._interface.model.encoder.layers[i].self_attn.q_proj
        ssh.model.encoder.layers[i].self_attn.out_proj = net._interface.model.encoder.layers[i].self_attn.out_proj

        ssh.model.encoder.layers[i].self_attn_layer_norm = net._interface.model.encoder.layers[i].self_attn_layer_norm
        ssh.model.encoder.layers[i].fc1 = net._interface.model.encoder.layers[i].fc1
        ssh.model.encoder.layers[i].fc2 = net._interface.model.encoder.layers[i].fc2
        ssh.model.encoder.layers[i].final_layer_normj = net._interface.model.encoder.layers[i].final_layer_norm


def share_bart_decoder_layers(net, ssh):
    ssh.model.shared.weight = net._interface.model.decoder.embed_tokens.weight
    ssh.model.decoder.embed_positions.weight = net._interface.model.decoder.embed_positions.weight
    ssh.model.decoder.layernorm_embedding = net._interface.model.decoder.layernorm_embedding

    for i in range(0, 12):
        ssh.model.decoder.layers[i].self_attn.k_proj = net._interface.model.decoder.layers[i].self_attn.k_proj
        ssh.model.decoder.layers[i].self_attn.v_proj = net._interface.model.decoder.layers[i].self_attn.v_proj
        ssh.model.decoder.layers[i].self_attn.q_proj = net._interface.model.decoder.layers[i].self_attn.q_proj
        ssh.model.decoder.layers[i].self_attn.out_proj = net._interface.model.decoder.layers[i].self_attn.out_proj

        ssh.model.decoder.layers[i].self_attn_layer_norm = net._interface.model.decoder.layers[i].self_attn_layer_norm

        ssh.model.decoder.layers[i].encoder_attn.k_proj = net._interface.model.decoder.layers[i].encoder_attn.k_proj
        ssh.model.decoder.layers[i].encoder_attn.v_proj = net._interface.model.decoder.layers[i].encoder_attn.v_proj
        ssh.model.decoder.layers[i].encoder_attn.q_proj = net._interface.model.decoder.layers[i].encoder_attn.q_proj
        ssh.model.decoder.layers[i].encoder_attn.out_proj = net._interface.model.decoder.layers[i].encoder_attn.out_proj
        ssh.model.decoder.layers[i].encoder_attn_layer_norm = net._interface.model.decoder.layers[i].encoder_attn_layer_norm
        ssh.model.decoder.layers[i].fc1 = net._interface.model.decoder.layers[i].fc1
        ssh.model.decoder.layers[i].fc2 = net._interface.model.decoder.layers[i].fc2
        ssh.model.decoder.layers[i].final_layer_normj = net._interface.model.decoder.layers[i].final_layer_norm


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels