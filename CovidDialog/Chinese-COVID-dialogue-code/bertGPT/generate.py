import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import json

import fire
import time
import numpy as np
import torch.multiprocessing as mp

from tqdm import tqdm

from collections import defaultdict
# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM

# uses bert chinese wordpiece tokenization
from pytorch_pretrained_bert import OpenAIAdam, BertTokenizer


def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


class MyDataset(Dataset):
    def __init__(self, *data):
        self.data = data

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data)

    def __len__(self):
        return len(self.data[0])


def collate_fn(batch):
    pad_id = 0
    input_ids = []
    output_ids = []
    input_mask = []
    output_mask =[]

    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    max_output_len = 0

    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx][0]):
            max_input_len = len(batch[btc_idx][0])
        if max_output_len < len(batch[btc_idx][1]):
            max_output_len = len(batch[btc_idx][1])
    # 使用pad_id对小于max_input_len的input_id进行补全

    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx][0])
        input_ids.append(batch[btc_idx][0])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))

        output_len = len(batch[btc_idx][1])
        output_ids.append(batch[btc_idx][1])
        output_ids[btc_idx].extend([pad_id] * (max_output_len - output_len))

        input_mask.append([1] * input_len + [pad_id] * (max_input_len - input_len))
        output_mask.append([1] * output_len + [pad_id] * (max_output_len - output_len))
    return tuple((torch.tensor(input_ids, dtype=torch.long), torch.tensor(output_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long), torch.tensor(output_mask, dtype=torch.long)))


def get_decoder(decoder_path, gpu_id):
    old_state_dict = torch.load(decoder_path, map_location=f'cuda:{gpu_id}')
    print(f'load from {decoder_path}')
    encoder = TransformerEncoder()
    decoder = TransformerDecoderLM()

    encoder_state_dict = encoder.state_dict()
    for i in encoder_state_dict.keys():
        encoder_state_dict[i] = old_state_dict['encoder.' + i]
    encoder.load_state_dict(encoder_state_dict)

    decoder_state_dict = decoder.state_dict()
    for i in decoder_state_dict.keys():
        decoder_state_dict[i] = old_state_dict['decoder.' + i]
    decoder.load_state_dict(decoder_state_dict)
    return encoder, decoder


def generate_sentences(test_data, tokenizer, decoder_path, rank, top_k, l):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{rank}")

    #------------------------LOAD MODEL-----------------
    encoder, decoder = get_decoder(decoder_path, gpu_id)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    test_dataset = MyDataset(*test_data)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)
    #------------------------END LOAD VALIDATE DATA--------------


    #------------------------START SAMPLE GENERETE-------------------
    if rank == 0:
        test_dataloader = tqdm(test_dataloader)
    for batch in test_dataloader:
        with torch.no_grad():
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask, _ = batch

            _, past = encoder(encoder_input, mask)

            sentence = []

            prev_pred = decoder_input[:, :1]
            sentence.append(prev_pred)

            length = 1
            # decoding loop
            for i in range(100):
                mask = F.pad(mask, (0, 1), "constant", 1.0)
                logits, past = decoder(prev_pred, mask, past=past, past_length=length)
                logits = logits.squeeze(1)
                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence.append(prev_pred)
                if prev_pred[0][0] == 102:
                    break
                length += 1

            sentence = torch.cat(sentence, dim=-1)
            predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())

            target = decoder_input.squeeze(dim=0)
            target_num = (target != 0).sum()
            reference = tokenizer.convert_ids_to_tokens(target[:target_num].tolist())

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())
            l.append(["".join(inputs[1:-1]), "".join(predict[1:-1]), "".join(reference[1:-1])])


    #------------------------END SAMPLE GENERETE-------------------


def sample_generate(
    top_k = 50,
    decoder_path='decoder.pth',
    process_num=1
    ):
    test_data = torch.load("test_data.pth")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    length = len(test_data[0])

    mgr = mp.Manager()
    l = mgr.list()
    processes = []
    for rank in range(process_num):
        if rank == process_num - 1:
            data = [d[int((rank / process_num) * length):] for d in test_data]
        else:
            data = [d[int((rank / process_num) * length) : int(((rank + 1) / process_num) * length)] for d in test_data]

        p = mp.Process(target=generate_sentences, args=(data, tokenizer, decoder_path, rank, top_k, l))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


    Dialog_list = []
    with open('generate_sentences.txt', 'w', encoding='utf-8') as f:
        for s in l:
            cases = dict()
            cases['input'] = s[0]
            cases['predict'] = s[1]
            cases['reference'] = s[2]
            Dialog_list.append(cases)
        json.dump(Dialog_list, f, ensure_ascii = False, indent = 4)


if __name__ == '__main__':
    fire.Fire(sample_generate)

