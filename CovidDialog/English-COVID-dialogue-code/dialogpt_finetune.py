import fire
import pickle

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


def main(model_size='medium', n_epochs=5):
    gpt2 = DialoGPT(model_size=model_size)

    for split in ['train', 'dev']:
        dialogues = load_data(split)
        gpt2.load_data(split=split, dialogues=dialogues, max_length=MAX_LENGTH)

    train_steps = n_epochs * (len(gpt2.datasets['train']) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    gpt2.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    gpt2.creat_log_dir(
        eval_steps=len(gpt2.datasets['train']) // BATCH_SIZE,
        label=f'dialogpt-{model_size}')
    for epoch in range(n_epochs):
        gpt2.train_epoch(batch_size=BATCH_SIZE)


if __name__ == '__main__':
    fire.Fire(main)