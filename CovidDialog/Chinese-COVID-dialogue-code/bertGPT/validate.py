import json
import fire
from tqdm import tqdm

from metrics import cal_entropy, cal_length, calculate_metrics


def validate(file_name):
    with open(file_name, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)
    
    bleu_2scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0
    meteor_scores = 0
    sentences = []

    for d in tqdm(data):
        reference = list(d['reference'])
        predict = list(d['predict'])
        temp_bleu_2, \
        temp_bleu_4, \
        temp_nist_2, \
        temp_nist_4, \
        temp_meteor_scores = calculate_metrics(predict, reference)

        bleu_2scores += temp_bleu_2
        bleu_4scores += temp_bleu_4
        nist_2scores += temp_nist_2
        nist_4scores += temp_nist_4
        meteor_scores += temp_meteor_scores
        sentences.append(" ".join(predict))

    entro, dist = cal_entropy(sentences)
    mean_len, var_len = cal_length(sentences)
    num = len(sentences)
    print(f'avg: {mean_len}, var: {var_len}')
    print(f'entro: {entro}')
    print(f'dist: {dist}')
    print(f'bleu_2scores: {bleu_2scores / num}')
    print(f'bleu_4scores: {bleu_4scores / num}')
    print(f'nist_2scores: {nist_2scores / num}')
    print(f'nist_4scores: {nist_4scores / num}')
    print(f'meteor_scores: {meteor_scores / num}')
    

if __name__ == '__main__':
    fire.Fire(validate)
