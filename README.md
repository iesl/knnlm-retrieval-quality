# You can't pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM

This repository is an optimized version of [urvashik/knnlm](https://github.com/urvashik/knnlm) and includes script to reproduce experiments from our [EMNLP 2022 Findings](https://arxiv.org/abs/2210.15859) paper.

```
@inproceedings{drozdov2022knnlm,
    title = "You can't pick your neighbors, or can you? {W}hen and how to rely on retrieval in the {kNN-LM}",
    author = "Andrew Drozdov and Shufan Wang and Razieh Rahimi and Andrew McCallum and Hamed Zamani and Mohit Iyyer",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022"
}
```

This code is based on the original kNN-LM repo: https://github.com/urvashik/knnlm **NOTE: Please review the documentation from the original repo before proceeding.**

```
@inproceedings{khandelwal20generalization,
  title={{Generalization through Memorization: Nearest Neighbor Language Models}},
  author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## Install Dependencies

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install --editable .
pip install faiss-cpu
```

## Fast Evaluation

First run these steps from the original kNN-LM repo:

1. Prepare your data.
2. Train your model (our download a checkpoint).
3. Save the keys and values to a datastore, but use our code instead. We cache some additional properties (i.e. the next-token probabilities).
4. Build the faiss index.

Then cache the neighbors and vector distances. And finally evaluate the model.

```
# We use the wiki_valid preset for convenience, but please double check the filepaths and replace with your own.

python rq/fast_evaluate.py --preset wiki_valid --save_knns # Save the neighbors.
python rq/fast_evaluate.py --preset wiki_valid --save_exact # Save the exact vector distances.
python rq/fast_evaluate.py --preset wiki_valid --exact # Compute perplexity using exact vector distance.

# Note: The first two steps can be time consuming, but the last step should run very fast.
```
