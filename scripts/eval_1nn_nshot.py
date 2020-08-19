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


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nearest Neighbors Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-large-uncased')
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-subword_op', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False,
                        choices=['mean', 'first', 'sum'])
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

    nshot_datasets = []
    nshot_datasets.append(['s1n1', 'nshots/set1', '_1', 'vectors/nshots/CoarseWSD-20.s1n1.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s1n3', 'nshots/set1', '_3', 'vectors/nshots/CoarseWSD-20.s1n3.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s1n10', 'nshots/set1', '_10', 'vectors/nshots/CoarseWSD-20.s1n10.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s1n30', 'nshots/set1', '_30', 'vectors/nshots/CoarseWSD-20.s1n30.%s.txt' % args.nlm_id])
    
    nshot_datasets.append(['s2n1', 'nshots/set2', '_1', 'vectors/nshots/CoarseWSD-20.s2n1.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s2n3', 'nshots/set2', '_3', 'vectors/nshots/CoarseWSD-20.s2n3.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s2n10', 'nshots/set2', '_10', 'vectors/nshots/CoarseWSD-20.s2n10.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s2n30', 'nshots/set2', '_30', 'vectors/nshots/CoarseWSD-20.s2n30.%s.txt' % args.nlm_id])
    
    nshot_datasets.append(['s3n1', 'nshots/set3', '_1', 'vectors/nshots/CoarseWSD-20.s3n1.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s3n3', 'nshots/set3', '_3', 'vectors/nshots/CoarseWSD-20.s3n3.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s3n10', 'nshots/set3', '_10', 'vectors/nshots/CoarseWSD-20.s3n10.%s.txt' % args.nlm_id])
    nshot_datasets.append(['s3n30', 'nshots/set3', '_30', 'vectors/nshots/CoarseWSD-20.s3n30.%s.txt' % args.nlm_id])


    for set_id, set_path, n_suffix, vecs_path in nshot_datasets:

        logging.info('Loading VSM ...')
        senses_vsm = VSM(vecs_path, normalize=True)

        if args.nlm_id not in vecs_path.split('/')[-1].split('.'):  # catch mismatched nlms/sense_vecs
            logging.fatal("Provided sense vectors don't seem to match nlm_id (%s)." % args.nlm_id)
            raise SystemExit('Fatal Error.')

        all_sense_preds = defaultdict(list)
        all_results = defaultdict(list)

        # matching test instances
        for amb_word in ambiguous_words:

            logging.info('Evaluating %s ...' % amb_word)

            try:
                # nshot_insts = load_instances(amb_word + n_suffix, split='test', setname=set_path)
                nshot_insts = load_instances(amb_word + n_suffix, split='test', setname=set_path, mode='mfs')
                # nshot_insts = load_instances(amb_word + n_suffix, split='test', setname=set_path, mode='lfs')
            except:
                logging.warn('Failed for %s - %s' % (set_id, amb_word))
                continue

            for inst_idx, test_inst in enumerate(nshot_insts):
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
            for sense in coarse_senses[amb_word]:
                sense_preds = all_sense_preds[sense]
                if len(sense_preds) == 0:
                    continue

                n_sense_correct = sum([1 for preds in sense_preds if preds[0][0] == sense])
                sense_acc = n_sense_correct / len(sense_preds)
                all_senses_accs[sense] = sense_acc

                n_word_correct += n_sense_correct
                n_word_insts += len(sense_preds)
            
            if n_word_insts > 0:
                all_words_accs[amb_word] = n_word_correct / n_word_insts
            else:
                all_words_accs[amb_word] = 0

        # writing perf summary and logging to stdout
        summary_path = 'results/1nn_nshot/%s/%s/summary.mfs.csv' % (args.nlm_id, set_id)
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
            with open('results/1nn_nshot/%s/%s/%s.mfs.jsonl' % (args.nlm_id, set_id, amb_word), 'w') as word_results_f:
                for inst_idx, (test_inst, inst_matches) in enumerate(all_results[amb_word]):
                    jsonl_results = {'idx': inst_idx, 'matches': inst_matches, 'gold': test_inst['class'], 'tokens': test_inst['tokens']}
                    word_results_f.write('%s\n' % json.dumps(jsonl_results, sort_keys=True))
