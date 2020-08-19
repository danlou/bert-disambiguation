import logging
import argparse
from time import time

import json
import numpy as np

from nlm_encoder import TransformerEncoder

from coarsewsd20_reader import load_instances
from coarsewsd20_reader import ambiguous_words


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def create_vecs(args):
    sense_vecs = {}
    n_sents = 0

    for word in ambiguous_words:
        logging.info('Processing \'%s\' ...' % word)

        for inst_idx, inst in enumerate(load_instances(word, split='train', setname=args.dataset_id)):
            n_sents += 1

            if encoder.get_num_subtokens(inst['tokens']) >= args.max_seq_len:
                logging.error('%s:%d exceeds max_seq_len (%d).' % (word, inst_idx, args.max_seq_len))
                continue

            try:
                inst_vecs = encoder.token_embeddings([inst['tokens']])[0][0]
            except:
                logging.info('ERROR: %s:%d' % (word, inst_idx + 1))
                continue

            assert inst_vecs[inst['idx']][0] == word  # sanity check

            word_vec = inst_vecs[inst['idx']][1]
            word_cls = inst['class']

            try:
                sense_vecs[word_cls]['vecs_sum'] += word_vec
                sense_vecs[word_cls]['vecs_num'] += 1
            except KeyError:
                sense_vecs[word_cls] = {'vecs_sum': word_vec, 'vecs_num': 1}


    logging.info('Writing Sense Vectors to %s ...' % args.out_path)
    with open(args.out_path, 'w') as vecs_f:
        for sense, vecs_info in sense_vecs.items():
            vec = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])            
            vecs_f.write('%s %s\n' % (sense, vec_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Initial Sense Embeddings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-base-uncased')
    parser.add_argument('-dataset_id', help='Dataset name', required=False, default='CoarseWSD-20')
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-subword_op', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
    args = parser.parse_args()

    args.layers = [int(n) for n in args.layers.split(' ')]

    encoder_cfg = {
        'model_name_or_path': args.nlm_id,
        'min_seq_len': 0,
        'max_seq_len': args.max_seq_len,
        'layers': args.layers,
        'layer_op': 'sum',
        'subword_op': 'mean'
    }

    logging.info('Loading NLM ...')
    encoder = TransformerEncoder(encoder_cfg)

    create_vecs(args)
