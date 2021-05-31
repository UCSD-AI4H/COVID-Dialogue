import math
import fire
import pickle
import cleantext

from models.dialo_gpt import DialoGPT


BATCH_SIZE = 4
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.05

MAX_LENGTH = 1024


def load_data(split):
    examples = pickle.load(open(f'data/english/{split}.pickle', 'rb'))

    dialogues = []
    for example in examples:
        dialogues.append(example['utterances'])

    return dialogues


def main(log_path, model_size='medium'):
    trainer = DialoGPT(model_size)
    trainer.load_model(f'{log_path}/best_model.pt')

    dialogues = load_data('test')
    trainer.load_data(split='dev', dialogues=dialogues)

    test_nll = trainer.evaluate()
    test_ppl = math.exp(test_nll)
    print(f'Test NLL: {test_nll}; Test PPL: {test_ppl}')

    gen_file = open(f'{log_path}/test.hypo', 'w')
    gold_file = open(f'{log_path}/test.gold', 'w')
    formatted_file = open(f'{log_path}/test.log', 'w')
    for dialogue in dialogues:
        if len(dialogue) % 2 == 1:
            dialogue = dialogue[:-1]

        tgt = dialogue[-1]
        gen_text = trainer.generate(dialogue[:-1])

        gen_text = cleantext.clean(gen_text, extra_spaces=True)
        tgt = cleantext.clean(tgt, extra_spaces=True)

        print(gen_text, file=gen_file)
        print(tgt, file=gold_file)

        print(f'CHAT_HISTORY:\n', file=formatted_file)
        for i, utterance in enumerate(dialogue[:-1]):
            if i % 2 == 0:
                print(f'Patient: {utterance}', file=formatted_file)
            else:
                print(f'Doctor: {utterance}', file=formatted_file)
        print(f'\nGROUND TRUTH:\n{tgt}', file=formatted_file)
        print(f'\nGENERATION:\n{gen_text}', file=formatted_file)
        print('=' * 100, '\n\n', file=formatted_file)


if __name__ == '__main__':
    fire.Fire(main)