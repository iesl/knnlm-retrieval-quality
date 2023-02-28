from tqdm import tqdm
import numpy as np
import torch


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

    # LM perplexity.
    print('get_knn_prob')
    knn_prob = get_knn_prob(dstore, target, dists, knns).view(-1, 1)
    lm_prob = torch.from_numpy(dataset.prob).float()
    ppl = eval_ppl(lm_prob)

    # kNN-LM perplexity.
    coeff_list = (np.arange(0, 100) / 100).tolist()
    new_ppl_list = []
    for coeff in tqdm(coeff_list, desc='coeff'):
        def fn():
            new_prob = dstore.combine_knn_and_vocab_probs(knn_prob, lm_prob, coeff)
            return eval_ppl(new_prob)
        new_ppl_list.append(fn())

    # Print a window around the best perplexity.
    topk = 5
    for ix in sorted(np.argsort(new_ppl_list)[:topk]):
        new_ppl = new_ppl_list[ix]
        coeff = coeff_list[ix]
        print(f'ppl = {ppl:.3f}, new_ppl = {new_ppl:.3f} ({coeff})')

    # Assign bins to each entry based on vector distance.
    number_of_bins = 64
    min_dist = (-1 * dists).min(-1)
    bins = find_bins(min_dist, number_of_bins)
    coeff_list = (np.arange(0, 100) / 100).tolist()
    bin_prob, bin_coeffs = dynamic_combine_knn_and_vocab_probs(knn_prob, lm_prob, bins, coeff_list)
    bin_ppl = eval_ppl(bin_prob)
    print(f'bin_ppl = {bin_ppl:.3f}, coeffs [{number_of_bins}] = {bin_coeffs}')


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    return curr_prob


def dynamic_combine_knn_and_vocab_probs(knn_prob, lm_prob, bins, coeff_list):
    bin_prob = torch.full(lm_prob.shape, 1, dtype=torch.float)
    bin_coeffs = []
    number_of_bins = bins.max().item() + 1
    for i in range(number_of_bins):
        mask = bins == i
        this_knn_prob = knn_prob[mask]
        this_lm_prob = lm_prob[mask]
        best_ppl, best_coeff = np.inf, None
        for coeff in coeff_list:
            this_prob = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, coeff)
            this_ppl = eval_ppl(this_prob)
            if this_ppl < best_ppl:
                best_ppl, best_coeff = this_ppl, coeff
        assert best_coeff is not None
        bin_coeffs.append(best_coeff)
        bin_prob[mask] = combine_knn_and_vocab_probs(this_knn_prob, this_lm_prob, best_coeff)
    assert (bin_prob < 1).all().item()
    return bin_prob, bin_coeffs


def find_bins(measure, number_of_bins):
    # Assign each entry to a bin based on statistics of the measure.
    bins = np.full(measure.shape, -1)
    pct_size = 100 / number_of_bins
    for i in range(number_of_bins):
        if i == number_of_bins - 1:
            pct_start = i * pct_size
            pct_end = 100
            pct_mask = np.logical_and(measure >= np.percentile(measure, pct_start), measure <= np.percentile(measure, pct_end))
        else:
            pct_start = i * pct_size
            pct_end = pct_start + pct_size
            pct_mask = np.logical_and(measure >= np.percentile(measure, pct_start), measure < np.percentile(measure, pct_end))
        bins[pct_mask] = i
    assert np.all(bins > -1).item()
    return bins

