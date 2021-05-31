import fire
import pickle
import math
import cleantext

from models.bart_trainer import BARTTrainer


BATCH_SIZE = 4
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.05
LABEL_SMOOTH_EPSILON = 0.1

SRC_MAX_LEN = 1024
TGT_MAX_LEN = 1024


def load_data(split):
    examples = pickle.load(open(f'data/english/{split}.pickle', 'rb'))

    src_texts, tgt_texts = [], []
    for dialogue in examples:
        utterances = dialogue['utterances']

        src = ''
        for i, utterance in enumerate(utterances):
            if i % 2 == 0:
                src += 'patient: ' + utterance + '\n'
            else:
                src_texts.append(src)
                tgt_texts.append(utterance)

                src += 'doctor: ' + utterance + '\n'

    return src_texts, tgt_texts


def main(log_path):
    trainer = BARTTrainer(init='bart.large')
    trainer.load_model(f'{log_path}/best_model.pt')

    src_texts, tgt_texts = load_data('test')
    trainer.load_data(
        split='dev',
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        src_max_len=SRC_MAX_LEN,
        tgt_max_len=TGT_MAX_LEN)

    test_nll = trainer.evaluate()
    test_ppl = math.exp(test_nll)
    print(f'Test NLL: {test_nll}; Test PPL: {test_ppl}')

    gen_file = open(f'{log_path}/test.hypo', 'w')
    gold_file = open(f'{log_path}/test.gold', 'w')
    formatted_file = open(f'{log_path}/test.log', 'w')
    for src, tgt in zip(src_texts, tgt_texts):
        gen_text = trainer.generate([src])[0]

        gen_text = cleantext.clean(gen_text, extra_spaces=True)
        tgt = cleantext.clean(tgt, extra_spaces=True)

        print(gen_text, file=gen_file)
        print(tgt, file=gold_file)

        print(f'CHAT_HISTORY:\n{src}', file=formatted_file)
        print(f'\nGROUND TRUTH:\n{tgt}', file=formatted_file)
        print(f'\nGENERATION:\n{gen_text}', file=formatted_file)
        print('=' * 100, '\n\n', file=formatted_file)


if __name__ == '__main__':
    fire.Fire(main)