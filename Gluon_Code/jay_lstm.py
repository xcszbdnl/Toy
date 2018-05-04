#!/usr/bin/python
# -*- coding: UTF-8 -*-

from mxnet import nd
import mxnet as mx
import random
import sys

def data_iter_random(corpus_index, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_index) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_index[pos: pos + num_steps]
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices]
        )
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices])
        yield data, label

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[: batch_len * batch_size].reshape(
        batch_size, batch_len
    )
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + 1 + num_steps]
        yield data, label


def get_inputs(data, vocab_size):
    return [nd.one_hot(X, vocab_size) for X in data.T]


def rnn(inputs, state, *params):
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)

def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                vocab_size, is_lstm=False):
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        inputs = get_inputs(output, vocab_size)
        if is_lstm:
            pass
        else:
            Y, state_h = rnn(inputs, state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=0).asscalar)
        output.append(next_input)


if __name__ == '__main__':
    # print(sys.getdefaultencoding())
    with open('./Data/jaychou_lyrics.txt', encoding='utf-8') as f:
        corpus = f.read()
    corpus = corpus.replace('\n', ' ').replace('\r', ' ')
    print(corpus[:1])
    idx_to_char = list(set(corpus))
    char_to_idx = dict([char, i] for i, char in enumerate(idx_to_char))
    vocab_size = len(idx_to_char)
    print(vocab_size)
    corpus_idx = [char_to_idx[char] for char in corpus]
    print(corpus[:40])
    print(corpus_idx[:40])
    num_steps = 5
    # mx.nd.waitall()
    myseq = list(range(30))
    print('upload1')
    for data, label in data_iter_consecutive(myseq, 2, 3, ctx=mx.gpu()):
        print('data:', data, '\nlabel:', label)
    inputs = get_inputs(data, vocab_size)
    print(len(inputs))
    print(inputs[0].shape)

