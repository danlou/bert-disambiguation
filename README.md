# Analysis of BERT for Word Sense Disambiguation

This repository contains code and datasets for running the main experiments covered in "Language Models and Word Sense Disambiguation: An Overview and Analysis".

# CoarseWSD-20

The CoarseWSD-20 dataset is a coarse-grained sense disambiguation built from Wikipedia (nouns only) targetting 2 to 5 senses of 20 ambiguous words.
It's was specifically designed to provide an ideal setting for evaluating WSD models (e.g. no senses in test sets missing from training), both quantitavely and qualitatively.

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
$ git clone https://github.com/danlou/
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

# Feature Extraction


## Creating sense vectors for 1NN

You may use the create_1nn_vecs.py script to create 

    $ python create_1nn_vecs.py -nlm_id bert-base-uncased -out_path vectors/sense_vecs.bert-base-uncased.txt

## Example evaluating sense vectors

    $ python eval_1nn.py -nlm_id bert-base-uncased -sv_path vectors/sense_vecs.bert-base-uncased.txt


WIP

# Fine-Tuning

WIP (kiamehr)

# FastText Baseline

To run our fastText experiments, first follow [these](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module) installation instructions.

In case you're interested in running the fastText baseline with pretrained embeddings run:

```bash
$ cd external/fastText  # from repo home
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
$ unzip crawl-300d-2M-subword.zip
```

WIP

# Results

WIP

# Citation

We have a pre-print available [here]():

```bibtex
@article{XYZ,
  title={XYZ},
  author={XYZ},
  journal={ArXiv},
  year={2020},
  volume={abs/XYZ}
}
```