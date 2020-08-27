This folder contains the CoarseWSD-20 dataset (v1.0, Aug 2020) released with this paper:

Language Models and Word Sense Disambiguation: An Overview and Analysis
https://arxiv.org/abs/2008.11608
Daniel Loureiro* (dloureiro@fc.up.pt)
Kiamehr Rezaee* (k_rezaee@comp.iust.ac.ir)
Mohammad Taher Pilehvar (pilehvar@teias.institute)
Jose Camacho-Collados (camachocolladosj@cardiff.ac.uk)
(*first authors)

In the CoarseWSD-20 dataset each word has a specific folder with the following set of files:
- class_map.txt: A dictionary mapping sense labels to sense number.
- train.data.txt: Single tokenized and uncased sentence per line. First token is the position of the ambiguous word.
- train.gold.txt: Correct class/sense number for the ambiguous word in the same line at train.data.txt.
- test.data.txt: Single tokenized and uncased sentence per line. First token is the position of the ambiguous word.
- test.gold.txt: Correct class/sense number for the ambiguous word in the same line at test.data.txt.

There's also a Python module to read the dataset available at:
https://github.com/danlou/bert-disambiguation/blob/master/coarsewsd20_reader.py

For any questions, create an issue on GitHub (https://github.com/danlou/bert-disambiguation/), or contact the first authors. 
