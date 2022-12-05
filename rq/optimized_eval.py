import argparse
import collections
import json
import os
import math
import sys
import time

import faiss
import numpy as np
import torch

from tqdm import tqdm

from vocab import Dictionary


def argument_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--k', default=1024)

    parser.add_argument('--load-pct', default=10, type=int,
                        help='Should be [0,100] corresponding to percent of keys to load in mem.')
    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--exact', action='store_true',
                        help='If set, then use the exact distances (these should be cached with --save-exact).')
    parser.add_argument('--save-knns', action='store_true')
    parser.add_argument('--save-exact', action='store_true')
    parser.add_argument('--preset', default=None, type=str,
                        help='Use a preset configuration for different datasets.')

    return parser


def set_presets(args):
    if args.preset is None:
        args.preset = 'ptb_valid'

    if args.preset == 'ptb_valid':
        args.vocab = 'data-bin/ptb/dict.txt'
        args.dstore = './work_data/ptb.train'
        args.dstore_size = 1001735
        args.eval_dstore = './work_data/ptb.valid'
        args.eval_dstore_cache = './work_data/ptb.valid.cache'
        args.eval_dstore_size = 42099

    args.dstore_knn_index = f'{args.dstore}/knn.index'


def eval_ppl(p):
    return 2**(-p.mean()/np.log(2))


def get_knn_prob(dstore, target, dists, knns, cuda=False):
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    d = torch.from_numpy(dists).to(device).float()
    probs = torch.log_softmax(d, -1)

    index_mask = torch.eq(torch.from_numpy(dstore.vals[knns]).to(device).long().squeeze(-1), torch.from_numpy(target).to(device).long()).float()
    index_mask[index_mask == 0] = -10000 # for stability
    index_mask[index_mask == 1] = 0

    log_prob = torch.logsumexp(probs + index_mask, dim=-1).cpu()

    return log_prob


class Dataset(object):
    def __init__(self, args):
        self.args = args
        path = args.eval_dstore
        dstore_size = args.eval_dstore_size
        self.query = np.memmap(f'{path}/dstore_keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1024))
        self.target = np.memmap(f'{path}/dstore_vals.npy', dtype=np.int32, mode='r', shape=(dstore_size, 1))
        #self.prob = np.memmap(f'{path}/dstore_prob.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1))

        #for k in ['query', 'target', 'prob']:
        for k in ['query', 'target']:
            v = getattr(self, k)
            new_v = np.ones(v.shape, dtype=v.dtype)
            new_v[:] = v
            setattr(self, k, new_v)

    def load_cache(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size
        self.dists = np.memmap(f'{path}/dstore_cache_dists.npy', dtype=np.float32, mode='r', shape=(dstore_size, 1024))
        self.knns = np.memmap(f'{path}/dstore_cache_knns.npy', dtype=np.int32, mode='r', shape=(dstore_size, 1024))

    def load_exact_dists(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size
        filename = f'{path}/dstore_cache_exact_dists.npy'
        assert os.path.exists(filename)
        self.exact_dists = np.memmap(filename, dtype=np.float32, mode='r', shape=(dstore_size, 1024))


class Dstore(object):
    def __init__(self, args):
        path = args.dstore
        dstore_size = args.dstore_size

        self.sim_func = 'do_not_recomp_l2'
        self.k = 1024

        self.keys = np.memmap(f'{path}/dstore_keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1024))
        self.vals = np.memmap(f'{path}/dstore_vals.npy', dtype=np.int32, mode='r', shape=(dstore_size, 1))

        print('load index')
        indexfile = args.dstore_knn_index
        self.index = faiss.read_index(indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)

        self.half = True
        self.metric_type = 'l2'

    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, coeff):
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - coeff)
        coeffs[1] = np.log(coeff)
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

        return curr_prob

    def get_knns(self, query, k=None):
        if k is None:
            k = self.k
        if query.dtype == np.float16:
            query = query.astype(np.float32)
        dists, knns = self.index.search(query, k)
        return dists, knns


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


def run_eval_ppl(context):

    # Local variables.
    dstore = context['dstore']
    keys = dstore.keys
    vals = dstore.vals
    dataset = context['dataset']
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    dists = context['dists']

    print('get_knn_prob')
    knn_prob = get_knn_prob(dstore, target, dists, knns).view(-1, 1)
    lm_prob = torch.from_numpy(dataset.prob).float()
    ppl = eval_ppl(lm_prob)
    print(ppl)


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

    run_eval_ppl(context)


if __name__ == '__main__':
    args = argument_parser().parse_args()

    set_presets(args)

    with torch.no_grad():
        main(args)

