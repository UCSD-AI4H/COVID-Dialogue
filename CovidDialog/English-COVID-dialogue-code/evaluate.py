import fire
from evaluation.metrics import nlp_metrics


def main(hypo_path, gold_path):
    hypo_file = open(hypo_path)
    gold_file = open(gold_path)

    nist, sbleu, bleu, meteor, entropy, diversity, avg_len = nlp_metrics(
        path_refs=[gold_file.name],
        path_hyp=hypo_file.name)

    print('nist:', nist)
    print('sbleu:', sbleu)
    print('bleu:', bleu)
    print('meteor:', meteor)
    print('entropy:', entropy)
    print('diversity:', diversity)
    print('avg_len:', avg_len)


if __name__ == '__main__':
    fire.Fire(main)