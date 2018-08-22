from pathlib import Path
from collections import Counter
from random import shuffle
from typing import Dict, List, Tuple

positive_path = Path('data/books/positive.review')
negative_path = Path('data/books/negative.review')

UNK = '<UNK>'


def token_freq(path: Path):
    with path.open(mode='r', encoding='utf-8') as data:
        lines = []
        for line in data:
            sentence = []
            for word in line.strip().split():
                separate = word.find(':')
                token = word[:separate]
                freq = int(word[separate + 1:])
                sentence.append((token, freq))
            lines.append(sentence)
        # print(sentence)
        return lines


def count_freq(path: Path, counter: Counter):
    lines = token_freq(path)
    for line in lines:
        for token, freq in line:
            # print(f'token => {token}\nfreq => {freq}')
            counter[token] += freq
    # for token, freq in :
    # 	print(f'{token}, {freq}')
    # counter[token] = freq+1
    # print(counter)
    return counter


def make_vocabulary(vocab_size: int, counter: Counter) -> Dict[str, int]:
    vocabulary = {}
    # print(f'original counter => {counter}')
    count_freq(positive_path, counter)
    # print(f'only pos word => {counter}')
    count_freq(negative_path, counter)
    # print(f'add neg word => {counter}')
    for index, (token, _) in enumerate(counter.most_common(vocab_size)):
        vocabulary[token] = index
    # print(vocabulary)
    vocabulary[UNK] = vocabulary.__len__()
    return vocabulary


def make_bag_of_word_vector(vocabulary: Dict[str, int], sentence: List[str]) -> List[int]:
    vector = [0] * vocabulary.__len__()
    for word in sentence:
        if word in vocabulary:
            index = vocabulary[word]
        else:
            index = vocabulary[UNK]
        vector[index] = 1
    return vector


def make_data(path: Path, target: int, vocabulary: Dict[str, int]) -> List[Tuple[List[int], int]]:
    data = []
    for sentence in token_freq(path):
        sentence, _ = zip(*sentence)
        sentence_vector = make_bag_of_word_vector(vocabulary, sentence)
        data.append((sentence_vector, target))
    return data


def return_with_target(vocab_size: int) -> Tuple[List[List[int]], List[int]]:
    vocabulary = make_vocabulary(vocab_size, Counter())

    pos_vectors, pos_targets = zip(*make_data(positive_path, 1, vocabulary))
    neg_vectors, neg_targets = zip(*make_data(negative_path, 0, vocabulary))

    dataset = list(zip(pos_vectors + neg_vectors, pos_targets + neg_targets))
    shuffle(dataset)
    data, target = zip(*dataset)
    # print(f'data => {type(data)}\ntarget => {type(target)}')
    return data, target


def iteration(data, target, batch_size: int):
    for sample_num in range(0, len(data) + 1, batch_size):
        yield data[sample_num * batch_size: (sample_num + 1) * batch_size], \
              target[sample_num * batch_size: (sample_num + 1) * batch_size]


if __name__ == '__main__':
    data, target = return_with_target(100)
    print(data)
    print(target)
    print(data.__len__())
    print(target.__len__())
