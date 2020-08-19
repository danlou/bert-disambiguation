import random
import lxml.etree
from collections import defaultdict

from coarsewsd20_reader import ambiguous_words
from coarsewsd20_reader import get_sk_mappings


def read_xml_sents(xml_path):
    with open(xml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('<sentence '):
                sent_elems = [line]
            elif line.startswith('<wf ') or line.startswith('<instance '):
                sent_elems.append(line)
            elif line.startswith('</sentence>'):
                sent_elems.append(line)
                yield lxml.etree.fromstring(''.join(sent_elems))


def get_id_mappings(keys_path):
    id2sks = {}
    with open(keys_path) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2sks[id_] = keys
    return id2sks


semcor_xml_path  = 'external/wsd_eval/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'
semcor_keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt'

semeval_xml_path  = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml'
semeval_keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt'

semcor_converted_path = 'data/CoarseWSD-20.outofdomain.txt'
semcor_counter_path = 'data/CoarseWSD-20.outofdomain.counts.txt'


sk_mappings = get_sk_mappings()

sense_instances = defaultdict(list)

# for xml_data, xml_keys in [(semcor_xml_path, semcor_keys_path)]:
# for xml_data, xml_keys in [(semeval_xml_path, semeval_keys_path)]:
for xml_data, xml_keys in [(semcor_xml_path, semcor_keys_path), (semeval_xml_path, semeval_keys_path)]:

    id2sks = get_id_mappings(xml_keys)

    for sent_idx, sent_et in enumerate(read_xml_sents(xml_data)):
        entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses', 'pos', 'id']}
        for ch in sent_et.getchildren():
            for k, v in ch.items():
                entry[k].append(v)
            entry['token_mw'].append(ch.text)

            if 'id' in ch.attrib.keys():
                entry['senses'].append(id2sks[ch.attrib['id']])
            else:
                entry['senses'].append(None)

        entry['token'] = sum([t.split() for t in entry['token_mw']], [])
        entry['sentence'] = ' '.join([t for t in entry['token_mw']])

        # handling multi-word expressions, mapping allows matching tokens with mw features
        idx_map_abs = []
        idx_map_rel = [(i, list(range(len(t.split()))))
                        for i, t in enumerate(entry['token_mw'])]
        token_counter = 0
        for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
            idx_tokens = [i+token_counter for i in idx_tokens]
            token_counter += len(idx_tokens)
            idx_map_abs.append([idx_group, idx_tokens])

        for mw_idx, tok_idxs in idx_map_abs:
            if entry['senses'][mw_idx] is None:
                continue

            for sk in entry['senses'][mw_idx]:
                # if (sk in sk_mappings) and (sk in relevant_sks):
                if sk in sk_mappings:
                    sense_instances[sk_mappings[sk]].append((tok_idxs[0], entry['token']))


    with open(semcor_converted_path, 'w') as converted_f:
        for sense, instances in sense_instances.items():
            for idx, tokens in instances:
                word = sense.split('_')[0]
                converted_f.write('%s\t%s\t%d\t%s\n' % (word, sense, idx, ' '.join(tokens)))

    with open(semcor_counter_path, 'w') as counter_f:
        for sense, instances in sense_instances.items():
            counter_f.write('%s\t%d\n' % (sense, len(instances)))
