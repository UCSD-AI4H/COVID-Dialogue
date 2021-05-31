import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import AdamW, BertModel, BertConfig, get_linear_schedule_with_warmup

from tqdm import tqdm

import fire
import time
import os

# uses allennlp modules
from allennlp.nn import util

#os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

class transforers_model(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8
        )
        self.encoder = BertModel(encoder_config)

        decoder_config = BertConfig(
            num_hidden_layers=6,
            vocab_size=21128,
            hidden_size=512,
            num_attention_heads=8
        )
        decoder_config.is_decoder = True
        self.decoder = BertModel(decoder_config)

        self.linear = nn.Linear(512, 21128, bias=False)

    def forward(self, input_ids, mask_encoder_input, output_ids, mask_decoder_input):
        encoder_hidden_states, _ = self.encoder(input_ids, mask_encoder_input)
        # out: [batch_size, max_length, hidden_size]
        out, _ = self.decoder(output_ids, mask_decoder_input, encoder_hidden_states=encoder_hidden_states)
        
        out = self.linear(out)
#        print (out.size())
#        print (knowledge_bias.size())
        return out



def train_model(
    epochs=30,
    num_gradients_accumulation=4,
    batch_size=8,
    gpu_id=0,
    lr=1e-4,
    load_dir='decoder_model'
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')

    model = transforers_model()
    # model = nn.DataParallel(model, device_ids = [5,6,7])
    model = model.to(device)
#    model=nn.DataParallel(model.module.**,device_ids=[5,6,7])
#    device = torch.device("cuda:0")


    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD TRAIN DATA------------------
    
    train_data = torch.load("train_data.pth")
#    train_data = train_data[:100]
    
#    print (train_knowledge.size())
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    
    val_data = torch.load("validate_data.pth")
#    val_data = val_data[:100]
    
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)
    #------------------------END LOAD TRAIN DATA--------------
    

    #------------------------SET OPTIMIZER-------------------

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,\
        lr=lr,\
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=num_train_optimization_steps // 10
        num_training_steps=num_train_optimization_steps
    )
    #------------------------END SET OPTIMIZER--------------


    #------------------------START TRAINING-------------------
    update_count = 0

    f = open("valid_loss.txt", "w")
    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        #------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        for batch in tqdm(train_dataloader):
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
#            print (knowledge_input.size())
            logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)
            
#            logits = nn.parallel.data_parallel(model, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, knowledge_input, device_ids=[5,6,7])

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()

            times += 1
            update_count += 1
            max_grad_norm = 1.0
            
            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end = time.time()
        print('-'*20 + 'epoch' + str(epoch) + '-'*20)
        print('time:' + str(end - start))
        print('loss:' + str(losses / times))
        start = end

        #------------------------validate------------------------
        model.eval()

        perplexity = 0
        temp_loss = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = [item.to(device) for item in batch]

                encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch
#                logits = nn.parallel.data_parallel(model, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, knowledge_input)
                
                logits = model(encoder_input, mask_encoder_input, decoder_input, mask_decoder_input)

                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")

                temp_loss += loss.item()
                perplexity += np.exp(loss.item())

                batch_count += 1

        print('validate perplexity:' + str(perplexity / batch_count))
        print ("validate loss:" + str(temp_loss / batch_count))
        
        f.write('-'*20 + f"Epoch {epoch}" + '-'*20 + '\n')
        f.write(f"perplexity: {str(perplexity / batch_count)}" + "\n")
        f.write(f"loss: {str(temp_loss / batch_count)}" + "\n\n")
        
        direct_path = os.path.join(os.path.abspath('.'), load_dir)
        if not os.path.exists(direct_path):
            os.mkdir(direct_path)

        torch.save(model.state_dict(), os.path.join(os.path.abspath('.'), load_dir, str(epoch) + "model.pth"))
#        torch.save(model.state_dict(), os.path.join(os.path.abspath('.'), load_dir, str(epoch) + "model.pth"))

    #------------------------END TRAINING-------------------
    f.close()

if __name__ == '__main__':
    fire.Fire(train_model)

