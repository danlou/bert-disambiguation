# Analysis of BERT for Word Sense Disambiguation

This repository contains code and datasets for running the main experiments covered in ["Language Models and Word Sense Disambiguation: An Overview and Analysis"](https://arxiv.org/abs/2008.11608).

# CoarseWSD-20

The CoarseWSD-20 dataset is a coarse-grained sense disambiguation built from Wikipedia (nouns only) targetting 2 to 5 senses of 20 ambiguous words.
It was specifically designed to provide an ideal setting for evaluating WSD models (e.g. no senses in test sets missing from training), both quantitavely and qualitatively.

In this repository we share the following versions of the CoarseWSD-20 dataset used in our experiments:

- [Full CoarseWSD-20](https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20)
- [Balanced](https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20_balanced)
- [Nshots (1, 3, 10, 30 - w/3 sets for each)](https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20_nshot)
- [Fractional (1%, 5%, 10%, 50%, 100%)](https://github.com/danlou/bert-disambiguation/tree/master/data/CoarseWSD-20_ratios)
- [Out-of-domain](https://github.com/danlou/bert-disambiguation/blob/master/data/CoarseWSD-20.outofdomain.tsv)


# Installation

### Prepare Environment

This project was developed on Python 3.6.5 from Anaconda distribution v4.6.2. As such, the pip requirements assume you already have packages that are included with Anaconda (numpy, etc.).
After cloning the repository, we recommend creating and activating a new environment to avoid any conflicts with existing installations in your system:

```bash
$ git clone https://github.com/danlou/bert-disambiguation.git
$ cd bert-disambiguation
$ conda create -n bert-disambiguation python=3.6.5
$ conda activate bert-disambiguation
# $ conda deactivate  # to exit environment when done with project
```

### Additional Packages

To install additional packages used by this project run:

```bash
pip install -r requirements.txt
```

The WordNet package for NLTK isn't installed by pip, but we can install it easily with:

```bash
$ python -c "import nltk; nltk.download('wordnet')"
```
*Note*: The WordNet package is only needed to replicate the experiments on WordNet, but not for the rest of the experiments (e.g. in CoarseWSD or any other dataset).

# Feature Extraction

The feature extraction method used in the paper involves two steps: (1) computing sense embeddings from the training set and (2) leveraging those precomputed sense embeddings to disambiguate contextual embeddings by finding the most similar sense embedding.
These two steps have separate scripts, which can be used as explained below.

You may use the [create_1nn_vecs.py](https://github.com/danlou/bert-disambiguation/blob/master/create_1nn_vecs.py) script to create sense embeddings from a particular set from our CoarseWSD-20 datasets.

    $ python create_1nn_vecs.py -nlm_id bert-base-uncased -dataset_id CoarseWSD-20 -out_path vectors/CoarseWSD-20.bert-base-uncased.txt

If you want to train on a different training set, such as the balanced version of CoarseWSD-20, just replace '-dataset_id CoarseWSD-20' with '-dataset_id CoarseWSD-20_balanced'.

Precomputed sense embeddings for the full CoarseWSD-20 training set are also available at [vectors](https://github.com/danlou/bert-disambiguation/tree/master/vectors).

To evaluate the 1NN method, you may use the [eval_1nn.py](https://github.com/danlou/bert-disambiguation/blob/master/eval_1nn.py) script, providing paths for the test set and precomputed sense embeddings.

    $ python eval_1nn.py -nlm_id bert-base-uncased -dataset_id CoarseWSD-20 -sv_path vectors/CoarseWSD-20.bert-base-uncased.txt

# Fine-Tuning

[WIP] This is still being merged. In the meantime, you can check the code [here](https://github.com/kiamehr74/CG20WSD-bert-baseline/blob/master/run.py) if interested.

# FastText Baseline

To run our fastText experiments, first follow [these](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module) installation instructions.

In case you're interested in running the fastText baseline with pretrained embeddings run:

```bash
$ cd external/fastText  # from repo home
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
$ unzip crawl-300d-2M-subword.zip
```

The [ftx_baseline.py](https://github.com/danlou/bert-disambiguation/blob/master/ftx_baseline.py) script handles both creating the fastText classification models (FTX-Base and FTX-Crawl) and evaluating.

To configure the script, you can edit the dataset_id and model_id variables starting at [line 86](https://github.com/danlou/bert-disambiguation/blob/master/ftx_baseline.py#L86).

# Results

Predictions from our experiments are available at [results](https://github.com/danlou/bert-disambiguation/tree/master/results/CoarseWSD-20).

# Citation

We have a pre-print available [here](https://arxiv.org/abs/2008.11608):

```bibtex
@misc{loureiro2020language,
    title={Language Models and Word Sense Disambiguation: An Overview and Analysis},
    author={Daniel Loureiro and Kiamehr Rezaee and Mohammad Taher Pilehvar and Jose Camacho-Collados},
    year={2020},
    eprint={2008.11608},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
