import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np

import fire
import time
import os
from tqdm import tqdm

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from chinese_gpt import TransformerEncoder, TransformerDecoderLM

# uses bert chinese wordpiece tokenization
from pytorch_pretrained_bert import OpenAIAdam

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


class BertGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()

        # for p in self.parameters():
        #     p.requires_grad=False

        self.decoder = TransformerDecoderLM()

    def forward(self, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input):
        _, past = self.encoder(encoder_input, mask_encoder_input)

        mask = torch.cat([mask_encoder_input, mask_decoder_input], dim=1)
        logits, _ = self.decoder(decoder_input, mask, past=past, past_length=0)

        return logits


def train_model(
    epochs=10,
    num_gradients_accumulation=4,
    batch_size=8,
    gpu_id=0,
    lr=1e-4,
    load_dir='decoder_model',
    decoder_model='original_pretrained_model_for_bertGPT.pth'
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    model = BertGPT()
    model.load_state_dict(torch.load(decoder_model))
    # model = nn.DataParallel(model, device_ids = [0])
    model = model.to(device)
    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    train_data = torch.load("train_data.pth")
    train_dataset = MyDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn)
    val_data = torch.load("validate_data.pth")
    val_dataset = MyDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size, num_workers=2, collate_fn=collate_fn)
    #------------------------END LOAD TRAIN DATA--------------
    

    #------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    print('train')
    print(len(optimizer_grouped_parameters[0]['params']))

    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=lr,
                        warmup=0.01,
                        max_grad_norm=1.0,
                        weight_decay=0.01,
                        t_total=num_train_optimization_steps)
    #------------------------END SET OPTIMIZER--------------


    #------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        for batch in tqdm(train_dataloader, desc='dirs'):
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
  
            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()
            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()
            times += 1
            
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                optimizer.step()
                optimizer.zero_grad()
        end = time.time()
        print('-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'time: {(end - start)}')
        print(f'loss: {losses / times}')
        start = end

        #------------------------validate------------------------
        model.eval()

        perplexity = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = [item.to(device) for item in batch]
                encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

                logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
                
                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
                perplexity += np.exp(loss.item())
                batch_count += 1

        print(f'validate perplexity: {perplexity / batch_count}')

        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)

        torch.save(model.state_dict(), os.path.join(direct_path, str(epoch) + "model.pth"))

    #------------------------END TRAINING-------------------


if __name__ == '__main__':
    fire.Fire(train_model)

