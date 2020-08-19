import logging
import argparse
import json
from time import time
from datetime import datetime
from collections import defaultdict
from collections import Counter

import numpy as np

from nlm_encoder import TransformerEncoder

from vectorspace import VSM

from coarsewsd20_reader import coarse_senses
from coarsewsd20_reader import load_instances
from coarsewsd20_reader import ambiguous_words

from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def eval_nn(args):
    
    all_sense_preds = defaultdict(list)
    all_results = defaultdict(list)

    # matching test instances
    for amb_word in ambiguous_words:
        logging.info('Evaluating %s ...' % amb_word)

        for inst_idx, test_inst in enumerate(load_instances(amb_word, split='test', setname=args.dataset_id, mode=args.mode)):
            gold_sense = test_inst['class']

            if encoder.get_num_subtokens(test_inst['tokens']) >= args.max_seq_len:
                logging.error('%s:%d exceeds max_seq_len (%d).' % (amb_word, inst_idx, args.max_seq_len))
                
                preds = [('NULL', -1)]
                all_sense_preds[gold_sense].append(preds)
                all_results[amb_word].append((test_inst, preds))
                continue

            inst_vecs = encoder.token_embeddings([test_inst['tokens']])[0][0]

            assert inst_vecs[test_inst['idx']][0] == amb_word  # sanity check

            amb_word_vec = inst_vecs[test_inst['idx']][1]
            amb_word_vec = amb_word_vec / np.linalg.norm(amb_word_vec)

            preds = senses_vsm.most_similar_vec(amb_word_vec, topn=None)

            # filter preds for target word
            preds = [(sense, score) for sense, score in preds if sense.split('_')[0] == amb_word]

            all_sense_preds[gold_sense].append(preds)
            all_results[amb_word].append((test_inst, preds))

    # computing accuracies
    all_senses_accs = {}
    all_words_accs  = {}
    for amb_word in coarse_senses:
        n_word_correct, n_word_insts = 0, 0
        
        all_gold, all_pred = [], []
        for sense in coarse_senses[amb_word]:
            sense_preds = all_sense_preds[sense]
            if len(sense_preds) == 0:
                continue

            n_sense_correct = sum([1 for preds in sense_preds if preds[0][0] == sense])
            sense_acc = n_sense_correct / len(sense_preds)
            all_senses_accs[sense] = sense_acc

            n_word_correct += n_sense_correct
            n_word_insts += len(sense_preds)

            all_pred += [preds[0][0] for preds in sense_preds]
            all_gold += [sense] * len(sense_preds)
        
        word_recall_scores = recall_score(all_gold, all_pred, average=None)
        word_recall_MFS = word_recall_scores[0]
        word_recall_LFS = word_recall_scores[-1]

        word_precision_scores = precision_score(all_gold, all_pred, average=None)
        word_precision_MFS = word_precision_scores[0]
        word_precision_LFS = word_precision_scores[-1]

        print(amb_word, 'PRECISION', word_precision_MFS, word_precision_LFS)
        print(amb_word, 'RECALL', word_recall_MFS, word_recall_LFS)
        
        all_words_accs[amb_word] = n_word_correct / n_word_insts

    # writing perf summary and logging to stdout
    if args.mode != 'regular':
        summary_path = 'results/%s/1nn/%s/summary.%s.csv' % (args.dataset_id, args.nlm_id, args.mode)
    else:
        summary_path = 'results/%s/1nn/%s/summary.csv' % (args.dataset_id, args.nlm_id)

    with open(summary_path, 'w') as summary_f:
        summary_f.write('word,sense,n_insts,acc\n')
        for amb_word in coarse_senses:
            n_word_insts = 0
            for sense in coarse_senses[amb_word]:
                if sense not in all_senses_accs:
                    continue
                
                sense_acc = all_senses_accs[sense]
                n_sense_insts = len(all_sense_preds[sense])
                n_word_insts += n_sense_insts
                summary_f.write('%s,%s,%d,%f\n' % (amb_word, sense, n_sense_insts, sense_acc))

            word_acc = all_words_accs[amb_word]
            summary_f.write('%s,%s,%d,%f\n' % (amb_word, 'ALL', n_word_insts, word_acc))

    # store full results for further analysis
    for amb_word in all_results:

        if args.mode != 'regular':
            word_results_path = 'results/%s/1nn/%s/%s.%s.jsonl' % (args.dataset_id, args.nlm_id, amb_word, args.mode)
        else:
            word_results_path = 'results/%s/1nn/%s/%s.jsonl' % (args.dataset_id, args.nlm_id, amb_word)
        
        with open(word_results_path, 'w') as word_results_f:
            for inst_idx, (test_inst, inst_matches) in enumerate(all_results[amb_word]):
                jsonl_results = {'idx': inst_idx, 'matches': inst_matches, 'gold': test_inst['class'], 'tokens': test_inst['tokens']}
                word_results_f.write('%s\n' % json.dumps(jsonl_results, sort_keys=True))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nearest Neighbors Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-base-uncased')
    parser.add_argument('-dataset_id', help='Dataset name', required=False, default='CoarseWSD-20')
    parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
    parser.add_argument('-mode', type=str, default='regular', help='MFS or LFS', required=False,
                        choices=['regular', 'mfs', 'lfs'])
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-subword_op', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False,
                        choices=['mean', 'first', 'sum'])
    args = parser.parse_args()

    args.layers = [int(n) for n in args.layers.split(' ')]

    if args.nlm_id not in args.sv_path.split('/')[-1].split('.'):  # catch mismatched nlms/sense_vecs
        logging.fatal("Provided sense vectors don't seem to match nlm_id (%s)." % args.nlm_id)
        raise SystemExit('Fatal Error.')

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

    logging.info('Loading VSM ...')
    senses_vsm = VSM(args.sv_path, normalize=True)

    eval_nn(args)
