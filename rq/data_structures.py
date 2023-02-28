import os

import faiss
import numpy as np
import torch


class Dataset(object):
    def __init__(self, args):
        self.args = args
        path = args.eval_dstore
        dstore_size = args.eval_dstore_size
        # TODO: We should allow for more (or less) neighbors to be included.
        self.query = np.memmap(f'{path}/dstore_keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1024))
        self.target = np.memmap(f'{path}/dstore_vals.npy', dtype=np.int32, mode='r', shape=(dstore_size, 1))
        self.prob = np.memmap(f'{path}/dstore_prob.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1))

        for k in ['query', 'target', 'prob']:
            v = getattr(self, k)
            new_v = np.ones(v.shape, dtype=v.dtype)
            new_v[:] = v
            setattr(self, k, new_v)

    def load_cache(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size

        if not args.eval_external_knns:
            self.dists = np.memmap(f'{path}/dstore_cache_dists.npy', dtype=np.float32, mode='r', shape=(dstore_size, 1024))
            self.knns = np.memmap(f'{path}/dstore_cache_knns.npy', dtype=np.int32, mode='r', shape=(dstore_size, 1024))
        else:
            # TODO: We don't load approx. distances since we assume the neighbors were set without faiss.
            #self.dists = np.memmap(f'{path}/dstore_cache_dists.npy', dtype=np.float32, mode='r', shape=(dstore_size, 1024))
            self.dists = np.ones(dtype=np.float32, shape=(dstore_size, 1024))
            self.knns = np.memmap(args.eval_external_knns, dtype=np.int32, mode='r', shape=(dstore_size, 1024))

    def load_exact_dists(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size
        if not args.eval_external_knns:
            filename = f'{path}/dstore_cache_exact_dists.npy'
        else:
            filename = f'{args.eval_external_knns}.exact_dists.npy'
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

