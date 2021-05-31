from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import json
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100


def seq2token_ids(source_seqs, target_seq):
    # 可以尝试对source_seq进行切分
    encoder_input = []
    for source_seq in source_seqs:
        # 去掉 xx：
        encoder_input += tokenizer.tokenize(source_seq[3:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[3:])  # 去掉 xx：

    # 设置不得超过 MAX_ENCODER_SIZE 大小
    if len(encoder_input) > MAX_ENCODER_SIZE - 1:
        if "[SEP]" in encoder_input[-MAX_ENCODER_SIZE:-1]:
            idx = encoder_input[:-1].index("[SEP]", -(MAX_ENCODER_SIZE - 1))
            encoder_input = encoder_input[idx + 1:]

    encoder_input = ["[CLS]"] + encoder_input[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    enc_len = len(encoder_input)
    dec_len = len(decoder_input)
    
    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    return encoder_input, decoder_input


def make_dataset(data, file_name='train_data.pth'):
    train_data = []

    for d in tqdm(data):
        d_len = len(d)
        for i in range(d_len // 2):
            encoder_input, decoder_input = seq2token_ids(d[:2 * i + 1], d[2 * i + 1])
            train_data.append((encoder_input, decoder_input))

    encoder_input, \
    decoder_input = zip(*train_data)

    encoder_input = list(encoder_input)
    decoder_input = list(decoder_input)

    encoder_len = len(encoder_input)
    decoder_len = len(decoder_input)

    train_data = [encoder_input, decoder_input]
    torch.save(train_data, file_name)


def process_dataset(dataset_file='../data/train_data.json', data_file='train_data.pth'):
    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)
    make_dataset(data, data_file)


print(f'Process the train dataset\n', 'train_data.pth')
process_dataset()

print(f'Process the validate dataset\n')
process_dataset('../data/validate_data.json', 'validate_data.pth')

print(f'Process the test dataset\n')
process_dataset('../data/test_data.json', 'test_data.pth')
