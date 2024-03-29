"""
Basic Instructions:

  1. Cache the approximate retrieval. `python rq/fast_evaluate.py --preset wiki_valid --save-knns`
  2. Cache the exact vector distance. `python rq/fast_evaluate.py --preset wiki_valid --save-exact`
  3. Compute perplexity with exact distance. `python rq/fast_evaluate.py --preset wiki_valid --exact`

"""

import argparse
import collections
import json
import os
import math
import sys
import time

import numpy as np
import torch

from tqdm import tqdm

from data_structures import Dataset, Dstore
from vocab import Dictionary
import knnlm_func


def argument_parser():
    parser = argparse.ArgumentParser()

    # Filepaths.
    parser.add_argument('--vocab', default=None, type=str,
                        help='Path to vocab file.')
    parser.add_argument('--dstore', default=None, type=str,
                        help='Path to dstore.')
    parser.add_argument('--dstore-size', default=None, type=int)
    parser.add_argument('--eval-dstore', default=None, type=str,
                        help='Path to precomputed evaluation information. Similar to dstore, but for evaluation.')
    parser.add_argument('--eval-dstore-size', default=None, type=int)
    parser.add_argument('--eval-dstore-cache', default=None, type=str,
                        help='Path to additional evaluation information.')
    parser.add_argument('--eval-external-knns', default=None, type=str,
                        help='If set, then override the kNNs that would have been returned from faiss.')

    # Algorithm configuration.
    parser.add_argument('--k', default=1024)
    parser.add_argument('--exact', action='store_true',
                        help='If set, then use the exact distances (these should be cached with --save-exact).')

    # Commands.
    parser.add_argument('--save-knns', action='store_true')
    parser.add_argument('--save-exact', action='store_true')

    # Preset configuration.
    parser.add_argument('--preset', default=None, type=str,
                        help='Use a preset configuration for different datasets.')

    # Hardware specific.
    parser.add_argument('--load-pct', default=10, type=int,
                        help='Should be [0,100] corresponding to percent of keys to load in mem.')
    parser.add_argument('--cuda', action='store_true')

    return parser


def set_presets(args):
    if args.preset is None:
        args.preset = 'ptb_valid'

    if args.preset == 'wiki_valid':
        args.vocab = 'data-bin/wikitext-103/dict.txt'
        args.dstore = '/iesl/local/adrozdov/knnlm_data'
        args.dstore_size = 103225485
        args.eval_dstore = '/iesl/local/adrozdov/knnlm_data.valid'
        args.eval_dstore_cache = '/iesl/local/adrozdov/knnlm_data.valid.cache'
        args.eval_dstore_size = 217646

    if args.preset == 'ptb_valid':
        args.vocab = 'data-bin/ptb/dict.txt'
        args.dstore = './work_data/ptb.train'
        args.dstore_size = 1003610
        args.eval_dstore = './work_data/ptb.valid'
        args.eval_dstore_cache = './work_data/ptb.valid.cache'
        args.eval_dstore_size = 42355

    args.dstore_knn_index = f'{args.dstore}/knn.index'


#
# Serialization Methods
#

def save_knns(args, dataset, dstore):
    cache = collections.defaultdict(list)

    batch_size = 128

    print('Precomputing neighbors...')
    for start in tqdm(range(0, dataset.query.shape[0], batch_size)):
        end = min(start + batch_size, dataset.query.shape[0])

        query, target = dataset.query[start:end], dataset.target[start:end]
        dists, knns = dstore.get_knns(query)
        cache['dists'].append(dists)
        cache['knns'].append(knns)

    os.system(f'mkdir -p {args.eval_dstore_cache}')

    dists = np.concatenate(cache['dists'], 0)
    knns = np.concatenate(cache['knns'], 0)

    dstore_dists = np.memmap(f'{args.eval_dstore_cache}/dstore_cache_dists.npy', dtype=np.float32, mode='w+', shape=dists.shape)
    dstore_dists[:] = dists
    dstore_knns = np.memmap(f'{args.eval_dstore_cache}/dstore_cache_knns.npy', dtype=np.int32, mode='w+', shape=knns.shape)
    dstore_knns[:] = knns


def save_exact(args, dataset, dstore):

    keys = dstore.keys
    vals = dstore.vals
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    scores = dataset.dists

    new_dist = np.ones(scores.shape, dtype=scores.dtype)

    def run_block(start_k, end_k):
        batch_size = 1024
        in_mem_keys = np.empty(shape=(end_k - start_k, keys.shape[1]), dtype=keys.dtype)
        for start in tqdm(range(0, in_mem_keys.shape[0], batch_size), desc='load-pct'):
            end = min(start + batch_size, in_mem_keys.shape[0])
            in_mem_keys[start:end] = keys[start_k + start: start_k + end]

        batch_size = 128

        # TODO: GPU usage is low. Try using dataloader?
        for start in tqdm(range(0, query.shape[0], batch_size), desc='exact'):
            end = min(start + batch_size, query.shape[0])

            q_vecs = torch.from_numpy(query[start:end]).float().cuda()
            k_idx = knns[start:end]

            batch_keys = np.zeros(shape=(k_idx.shape[0], k_idx.shape[1], keys.shape[1]), dtype=keys.dtype)
            batch_mask = np.logical_and(k_idx >= start_k, k_idx < end_k)
            batch_keys[batch_mask] = in_mem_keys[k_idx[batch_mask] - start_k]

            # TODO: This is doing a lot of extra work, since many keys are blank.
            k_vecs = torch.from_numpy(batch_keys).float().cuda()
            d = -torch.sum((q_vecs[:, None, :] - k_vecs)**2, 2)

            d = d.cpu().numpy()
            batch_dist = new_dist[start:end].copy()
            batch_dist[batch_mask] = d[batch_mask]

            new_dist[start:end] = batch_dist

    block_size = int(args.load_pct / 100 * keys.shape[0])
    num_blocks = math.ceil(100 / args.load_pct)
    print(f'num_blocks = {num_blocks}')
    for i in range(num_blocks):
        start_k = i * block_size
        end_k = start_k + block_size
        if i == num_blocks - 1:
            end_k = keys.shape[0]
        assert start_k < end_k
        header = '\n\n' + '*' * 20 + f'{i * args.load_pct}/100 ({i}/{num_blocks})' + '*'*20 + '\n'
        header += f'slice = {start_k}:{end_k} of {keys.shape[0]}\n'
        header += '\n\n'
        print(header)
        run_block(start_k, end_k)

    dstore_exact_dists = np.memmap(f'{args.eval_dstore_cache}/dstore_cache_exact_dists.npy', dtype=np.float32, mode='w+', shape=scores.shape)
    dstore_exact_dists[:] = new_dist

    time.sleep(1)


#
# Main
#

def main(args):
    print('load dataset')
    dataset = Dataset(args)
    print('load dstore')
    dstore = Dstore(args)

    if args.save_knns:
        save_knns(args, dataset, dstore)
        print('done')
        sys.exit()

    print('load cache')
    dataset.load_cache()

    if args.save_exact:
        save_exact(args, dataset, dstore)
        print('done')
        sys.exit()

    if args.exact:
        print('load exact')
        dataset.load_exact_dists()
        dists = dataset.exact_dists
    else:
        dists = -1 * dataset.dists

    # Vocab.
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    print('found {} tokens in vocab {}'.format(len(vocab), args.vocab))

    # Context.
    context = {}
    context['args'] = args
    context['vocab'] = vocab
    context['dstore'] = dstore
    context['dataset'] = dataset
    context['dists'] = dists

    knnlm_func.run_eval_ppl(context)


if __name__ == '__main__':
    args = argument_parser().parse_args()

    set_presets(args)

    with torch.no_grad():
        main(args)

