import logging
import json
from collections import defaultdict
import numpy as np

from coarsewsd20_reader import ambiguous_words
from coarsewsd20_reader import load_instances
from coarsewsd20_reader import coarse_senses
from coarsewsd20_reader import sense2word

import fasttext


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def rem_prefix(label):
    return label.replace('__label__', '')


def convert_dataset(dataset_id):
    for split in ['train', 'test']:
        for amb_word in ambiguous_words:
            with open('data/fasttext_data/%s.fasttext.%s.%s' % (dataset_id, amb_word, split), 'w') as word_split_f:
                for inst in load_instances(amb_word, split=split, setname=dataset_id):
                    inst_str = '__label__%s %s' % (inst['class'], ' '.join(inst['tokens']))
                    word_split_f.write('%s\n' % inst_str)


def create_models(use_pretrained=False):

    for amb_word in ambiguous_words:
        logging.info('Generating model for \'%s\' ...' % amb_word)
        train_path = 'data/fasttext_data/%s.fasttext.%s.train' % (dataset_id, amb_word)
        if use_pretrained:
            model = fasttext.train_supervised(input=train_path,
                                              pretrainedVectors='external/fastText-0.9.1/crawl-300d-2M.vec',
                                              epoch=25, lr=0.5, dim=300, loss='ova')
            model_fn = '%s.fasttext.%s.crawl-300d-2M.model.bin' % (dataset_id, amb_word)

        else:
            model = fasttext.train_supervised(input=train_path,
                                              epoch=25, lr=0.5, dim=100, loss='ova')
            model_fn = '%s.fasttext.%s.base-100d.model.bin' % (dataset_id, amb_word)

        model.save_model('data/fasttext_models/' + model_fn)


def test_model(amb_word, model_id, dataset_id):

    model_path = 'data/fasttext_models/%s.fasttext.%s.%s.model.bin' % (dataset_id, amb_word, model_id)
    test_path = 'data/fasttext_data/%s.fasttext.%s.test' % (dataset_id, amb_word)
    logging.info('Processing %s with %s ...' % (amb_word, model_path))

    model = fasttext.load_model(model_path)

    results = []
    with open(test_path) as test_f:
        for inst_idx, instance_str in enumerate(test_f):

            elems = instance_str.strip().split()
            gold_label, inst_tokens = elems[0], elems[1:]
            gold_sense = rem_prefix(gold_label)

            matches = model.predict(' '.join(inst_tokens), k=-1)
            matches = [(rem_prefix(label), score) for label, score in zip(matches[0], matches[1])]

            # filter matches for word
            matches = [(sense, score) for sense, score in matches if sense.split('_')[0] == amb_word]
            results.append((inst_idx, inst_tokens, gold_sense, matches))

    # store full results for further analysis
    with open('results/%s/fasttext/%s/%s.jsonl' % (dataset_id, model_id, amb_word), 'w') as word_results_f:
        for inst_idx, inst_tokens, gold_sense, matches in results:
            jsonl_results = {'idx': inst_idx, 'matches': matches, 'gold': gold_sense, 'tokens': inst_tokens}
            word_results_f.write('%s\n' % json.dumps(jsonl_results, sort_keys=True))

    return results


if __name__ == '__main__':

    # specify CoarseWSD-20 subset
    dataset_id = 'CoarseWSD-20'
    # dataset_id = 'CoarseWSD-20_balanced'
    convert_dataset(dataset_id)

    # FTX-Base
    model_id = 'base-100d'
    create_models(use_pretrained=False)

    # FTX-Crawl
    # model_id = 'crawl-300d-2M'
    # create_models(use_pretrained=True)  # takes a while ...

    # test models
    all_results = {}
    for amb_word in ambiguous_words:
        all_results[amb_word] = test_model(amb_word, model_id, dataset_id)

    # computing accuracies
    all_senses_accs = {}
    all_words_accs  = {}
    all_sense_preds = defaultdict(list)
    for amb_word in coarse_senses:
        n_word_correct, n_word_insts = 0, 0
        for amb_word_sense in coarse_senses[amb_word]:

            for (_, _, gold_sense, preds) in all_results[amb_word]:
                if gold_sense == amb_word_sense:
                    all_sense_preds[amb_word_sense].append(preds)

            if len(all_sense_preds[amb_word_sense]) == 0:
                continue
            
            amb_word_sense_preds = all_sense_preds[amb_word_sense]

            n_sense_correct = sum([1 for preds in amb_word_sense_preds if preds[0][0] == amb_word_sense])
            sense_acc = n_sense_correct / len(amb_word_sense_preds)
            all_senses_accs[amb_word_sense] = sense_acc

            n_word_correct += n_sense_correct
            n_word_insts += len(amb_word_sense_preds)
        
        all_words_accs[amb_word] = n_word_correct / n_word_insts
        logging.info('%s - %f' % (amb_word, all_words_accs[amb_word]))


    # writing perf summary and logging to stdout
    summary_path = 'results/%s/fasttext/%s/summary.csv' % (dataset_id, model_id)
    with open(summary_path, 'w') as summary_f:
        summary_f.write('word,sense,n_insts,acc\n')
        for amb_word in coarse_senses:
            n_word_insts = 0
            for amb_word_sense in coarse_senses[amb_word]:
                if amb_word_sense not in all_senses_accs:
                    continue
                
                sense_acc = all_senses_accs[amb_word_sense]
                n_sense_insts = len(all_sense_preds[amb_word_sense])
                summary_f.write('%s,%s,%d,%f\n' % (amb_word, amb_word_sense, n_sense_insts, sense_acc))
                # print(amb_word_sense, n_insts, sense_acc)
                n_word_insts += n_sense_insts

            word_acc = all_words_accs[amb_word]
            summary_f.write('%s,%s,%d,%f\n' % (amb_word, 'ALL', n_word_insts, word_acc))

