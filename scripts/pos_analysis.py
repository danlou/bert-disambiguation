import sys
from nltk.corpus import wordnet as wn

results_path = sys.argv[1]

sk2pos = {}
for synset in wn.all_synsets():
    for lemma in synset.lemmas():
        sk2pos[lemma.key()] = synset.pos()

pos_confusion_matrix = {}
pos_confusion_matrix['n'] = {'n': 0, 'v': 0, 'a': 0, 'r': 0}
pos_confusion_matrix['v'] = {'n': 0, 'v': 0, 'a': 0, 'r': 0}
pos_confusion_matrix['a'] = {'n': 0, 'v': 0, 'a': 0, 'r': 0}
pos_confusion_matrix['s'] = {'n': 0, 'v': 0, 'a': 0, 'r': 0}
pos_confusion_matrix['r'] = {'n': 0, 'v': 0, 'a': 0, 'r': 0}


with open(results_path) as results_f:
    for line in results_f:
        elems = line.strip().split('\t')
        inst_id, gold_sk, matches = elems[0], elems[1], elems[2:]
        matches = [m.split('|') for m in matches]
        matches = [(sk, float(sim)) for sk, sim in matches]

        gold_pos = sk2pos[gold_sk.split(',')[0]]
        pred_pos = sk2pos[matches[0][0]]

        # merges 'A' with 'S'
        if gold_pos == 's':
            gold_pos = 'a'
        if pred_pos == 's':
            pred_pos = 'a'

        pos_confusion_matrix[gold_pos][pred_pos] += 1


print('N', pos_confusion_matrix['n'])
print('V', pos_confusion_matrix['v'])
print('A', pos_confusion_matrix['a'])
print('R', pos_confusion_matrix['r'])
