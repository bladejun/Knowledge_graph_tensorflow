"""This module for generating negative triple samples. (replace head or tail)"""

import random


def generate_negative_samples(input, output):
    head_entities = set()
    tail_entities = set()

    triples = []

    for line in open(input):
        h, t, r = line.strip().split('\t')

        head_entities.add(h)
        tail_entities.add(t)
        triples.append([h, t, r])

    head_entities = list(head_entities)
    tail_entities = list(tail_entities)
    with open(output, "w") as fo:
        for h, t, r in triples:
            head_neg = h
            tail_neg = t
            prob = random.random()
            if prob > 0.5:
                head_neg = random.choice(head_entities)
            else:
                tail_neg = random.choice(tail_entities)

            fo.write('\t'.join([head_neg, tail_neg, r])+'\n')


if __name__ == '__main__':
    generate_negative_samples("data/FB15k/train.txt", "data/FB15k/train_negative.txt")
    generate_negative_samples("data/WN18/train.txt", "data/WN18/train_negative.txt")