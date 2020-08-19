
# mapping instance ids to sensekeys and csis
inst_id_mapping = {}

inst_id_sk_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt'
with open(inst_id_sk_path) as inst_id_sk_f:
    for line in inst_id_sk_f:
        elems = line.split()
        inst_id, sks = elems[0], elems[1:]
        inst_id_mapping[inst_id] = {'sks': sks, 'csi': []}


inst_id_csi_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.csi.key.txt'
with open(inst_id_csi_path) as inst_id_csi_f:
    for line in inst_id_csi_f:
        elems = line.split()
        inst_id, csi = elems[0], elems[1]
        inst_id_mapping[inst_id]['csi'] = csi


# collecting pos instance ids
inst_id_sk_nouns_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.NOUN.key.txt'
inst_id_sk_verbs_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.VERB.key.txt'
inst_id_sk_adjs_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.ADJ.key.txt'
inst_id_sk_advs_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.ADV.key.txt'


inst_id_nouns = []
inst_id_verbs = []
inst_id_adjs = []
inst_id_advs = []


with open(inst_id_sk_nouns_path) as inst_id_sk_nouns_f:
    for line in inst_id_sk_nouns_f:
        elems = line.split()
        inst_id, sks = elems[0], elems[1:]
        inst_id_nouns.append(inst_id)


with open(inst_id_sk_verbs_path) as inst_id_sk_verbs_f:
    for line in inst_id_sk_verbs_f:
        elems = line.split()
        inst_id, sks = elems[0], elems[1:]
        inst_id_verbs.append(inst_id)


with open(inst_id_sk_adjs_path) as inst_id_sk_adjs_f:
    for line in inst_id_sk_adjs_f:
        elems = line.split()
        inst_id, sks = elems[0], elems[1:]
        inst_id_adjs.append(inst_id)


with open(inst_id_sk_advs_path) as inst_id_sk_advs_f:
    for line in inst_id_sk_advs_f:
        elems = line.split()
        inst_id, sks = elems[0], elems[1:]
        inst_id_advs.append(inst_id)


# writing converted sk to csi pos mappings
inst_id_csi_nouns_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.csi.NOUN.key.txt'
inst_id_csi_verbs_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.csi.VERB.key.txt'
inst_id_csi_adjs_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.csi.ADJ.key.txt'
inst_id_csi_advs_path = 'external/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.csi.ADV.key.txt'


with open(inst_id_csi_nouns_path, 'w') as inst_id_csi_nouns_f:
    for noun_inst in inst_id_nouns:
        inst_id_csi_nouns_f.write('%s %s\n' % (noun_inst, inst_id_mapping[noun_inst]['csi']))


with open(inst_id_csi_verbs_path, 'w') as inst_id_csi_verbs_f:
    for verb_inst in inst_id_verbs:
        inst_id_csi_verbs_f.write('%s %s\n' % (verb_inst, inst_id_mapping[verb_inst]['csi']))


with open(inst_id_csi_adjs_path, 'w') as inst_id_csi_adjs_f:
    for adj_inst in inst_id_adjs:
        inst_id_csi_adjs_f.write('%s %s\n' % (adj_inst, inst_id_mapping[adj_inst]['csi']))


with open(inst_id_csi_advs_path, 'w') as inst_id_csi_advs_f:
    for adv_inst in inst_id_advs:
        inst_id_csi_advs_f.write('%s %s\n' % (adv_inst, inst_id_mapping[adv_inst]['csi']))

