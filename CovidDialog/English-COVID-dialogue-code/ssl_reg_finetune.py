import fire
import pickle

from models.ssl_reg_trainer import BARTTrainer

BATCH_SIZE = 4
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.05
LABEL_SMOOTH_EPSILON = 0.1

SRC_MAX_LEN = 512
TGT_MAX_LEN = 512


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


# text_type=0: src_text + " " + trt_text
# text_type=1: src_text
# text_type=2: trt_text

def main(n_epochs=5, weight=0.1, text_type=0, shared_training='encoder'):
    trainer = BARTTrainer(init='bart.large', shared_training=shared_training)

    for split in ['train', 'dev']:
        src_texts, tgt_texts = load_data(split)
        trainer.load_data(
            split=split,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            src_max_len=SRC_MAX_LEN,
            tgt_max_len=TGT_MAX_LEN)

    train_steps = n_epochs * (len(trainer.dataset['train']) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    trainer.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON,
        shared_training=shared_training
    )

    log_name = f'bart&ssl_reg_{n_epochs}epochs'

    trainer.create_training_log(
        eval_steps=len(trainer.dataset['train']) // BATCH_SIZE,
        label=log_name)

    for epoch in range(n_epochs):
        trainer.train_epoch(
            batch_size=BATCH_SIZE,
            label_smooth_epsilon=LABEL_SMOOTH_EPSILON,
            weight=weight,
            text_type=text_type
        )


if __name__ == '__main__':
    fire.Fire(main)
