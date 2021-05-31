import os
import pickle
import re
import cleantext
import random
from transformers import GPT2Tokenizer
import string


DATA_DIR = 'data/english'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(
            DATA_DIR, 'COVID-Dialogue-Dataset-English.txt')):
        os.system(f'wget -P {DATA_DIR} https://raw.githubusercontent.com/UCSD'
                  f'-AI4H/COVID-Dialogue/master/COVID-Dialogue-Dataset'
                  f'-English.txt')

    dialogues_texts_dirty = open(os.path.join(
        DATA_DIR, 'COVID-Dialogue-Dataset-English.txt')).read().split('id=')

    dialogues = []
    for text in dialogues_texts_dirty:
        text = text[text.find('Description'):].strip()

        description = text[len('Description\n'):text.find('\nDialogue')]
        description = cleantext.clean(
            description, extra_spaces=True, lowercase=True)

        text = text[text.find('\nPatient:'):]

        utterances, last_person, valid = [], 'None', True
        for x in re.finditer('Doctor:|Patient:', text):
            if x.group() == last_person:
                valid = False
                break
            else:
                last_person = x.group()

            utterance = text[x.end():].split('Patient:')[0].split('Doctor:')[0]
            utterances.append(cleantext.clean(
                utterance, extra_spaces=True, lowercase=True))

        if valid and utterances:
            dialogues.append({
                'description': description,
                'utterances': utterances
            })

    print('#dialogs:', len(dialogues))

    random.seed(11111)
    random.shuffle(dialogues)

    train_size = int(0.8 * len(dialogues))
    dev_size = int(0.1 * len(dialogues))

    pickle.dump(dialogues[:train_size],
                open(f'{DATA_DIR}/train.pickle', 'wb'))
    pickle.dump(dialogues[train_size:train_size + dev_size],
                open(f'{DATA_DIR}/dev.pickle', 'wb'))
    pickle.dump(dialogues[train_size + dev_size:],
                open(f'{DATA_DIR}/test.pickle', 'wb'))

    print_fairseq_format()


def process_gpt2_bpe(sent):
    gpt2_tokens = tokenizer.tokenize(sent)

    processed_tokens = [gpt2_tokens[0]]
    for token in gpt2_tokens[1:]:
        if any([s in string.ascii_letters for s in token]):
            if not token.startswith('Ġ'):
                if processed_tokens[-1][-1] not in string.punctuation:
                    processed_tokens[-1] += '@@'
            else:
                token = token[1:]
        processed_tokens.append(token)
    return ' '.join(processed_tokens)


def print_fairseq_format():
    for split in ['train', 'dev', 'test']:
        dialogues = pickle.load(open(f'{DATA_DIR}/{split}.pickle', 'rb'))

        src_file = open(f'{DATA_DIR}/{split}.chat_history', 'w')
        tgt_file = open(f'{DATA_DIR}/{split}.response', 'w')
        for dialogue in dialogues:
            src = 'description: ' + dialogue['description'] + ' ; '
            for i, utterance in enumerate(dialogue['utterances']):
                if i % 2 == 0:
                    src += 'patient: ' + utterance + ' ; '
                else:
                    print(src, file=src_file)
                    print(utterance, file=tgt_file)
                    src += 'doctor: ' + utterance + ' ; '


if __name__ == '__main__':
    main()