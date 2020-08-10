from functools import lru_cache
from collections import Counter
from nltk.corpus import wordnet as wn


class NoSynset(Exception):
   pass


class WN_Utils():

    def __init__(self):

        self.map_sk2syn = {}
        self.map_syn2csi = {}

        self.load_sk2syn()
        self.load_csi()

    def load_sk2syn(self):
        for synset in wn.all_synsets():
            for lemma in synset.lemmas():
                self.map_sk2syn[lemma.key()] = synset

    def load_csi(self):
        with open('data/csi_data/wn_synset2csi.txt') as csi_map_f:
            for line in csi_map_f:
                elems = line.strip().split('\t')
                wn_offset, csi_labels = elems[0], elems[1:]
                wn_offset = wn_offset.lstrip('wn:')
                syn = wn.of2ss(wn_offset)
                self.map_syn2csi[syn.name()] = csi_labels[0]

    @lru_cache()
    def syn2sks(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)
        return list(set([lemma.key() for lemma in synset.lemmas()]))

    @lru_cache()
    def syn2pos(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)
        return synset.pos()

    @lru_cache()
    def syn2lemmas(self, synset, include_pos=False):
        if isinstance(synset, str):
            synset = wn.synset(synset)

        lemmas = synset.lemma_names()
        if include_pos:
            lemmas = ['%s|%s' % (lem, synset.pos()) for lem in lemmas]
        return lemmas

    @lru_cache()
    def syn2lexname(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)
        
        return synset.lexname()

    @lru_cache()
    def syn2offset(self, synset):
        return synset.offset()

    @lru_cache()
    def syn2csi(self, synset):
        try:
            return self.map_syn2csi[synset.name()]
        except KeyError:  # synset not covered
            return None

    @lru_cache()
    def sk2syn(self, sk):
        return self.map_sk2syn[sk]

    @lru_cache()
    def sk2lemma(self, sk, use_ws=False):
        # lemma_name = wn.lemma_from_key(sk).name()  # alt?s
        lemma_name = sk.split('%')[0]
        if use_ws:
            lemma_name = lemma_name.replace('_', ' ')
        return lemma_name

    @lru_cache()
    def sk2pos(self, sk):
        # merging ADJ with ADJ_SAT
        sk_types_map = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 'a'}
        sk_type = int(sk.split('%')[1].split(':')[0])
        return sk_types_map[sk_type]
        # syn = self.sk2syn(sk)
        # return self.syn2pos(syn)

    @lru_cache()
    def sk2lexname(self, sk):
        syn = self.sk2syn(sk)
        return self.syn2lexname(syn)

    @lru_cache()
    def sk2csi(self, sk):
        syn = self.sk2syn(sk)
        return self.syn2csi(syn)

    @lru_cache()
    def lemma2syns(self, lemma, pos=None):

        if '|' in lemma:  # custom format, overrides arg
            lemma, pos = lemma.split('|')

        lemma = lemma.replace(' ', '_')

        # merging ADJ with ADJ_SAT
        if pos in ['a', 's']:
            syns = wn.synsets(lemma, pos='a') + wn.synsets(lemma, pos='s')
        else:
            syns = wn.synsets(lemma, pos=pos)

        if len(syns) > 0:
            return syns
        else:
            raise NoSynset('No synset for lemma=\'%s\', pos=\'%s\'.' % (lemma, pos))

    @lru_cache()
    def lemma2sks(self, lemma, pos=None):
        sks = set()

        if '|' in lemma:  # custom format, overrides arg
            lemma, pos = lemma.split('|')
        lemma = lemma.replace(' ', '_')

        for syn in self.lemma2syns(lemma, pos=pos):
            for sk in self.syn2sks(syn):
                if self.sk2lemma(sk, use_ws=False) == lemma:
                    sks.add(sk)

        return list(sks)

    @lru_cache()
    def lemma2lexnames(self, lemma, pos=None):
        lexnames = set()
        for syn in self.lemma2syns(lemma, pos=pos):
            lexnames.add(self.syn2lexname(syn))
        return list(lexnames)

    @lru_cache()
    def lemma2csis(self, lemma, pos=None):
        csis = set()
        for syn in self.lemma2syns(lemma, pos=pos):
            csis.add(self.syn2csi(syn))
        return list(csis)
    
    @lru_cache()
    def synid2syn(self, synid):
        return wn.of2ss(synid)

    @lru_cache()
    def synname2syn(self, synname):
        return wn.synset(synname)

    def get_all_syns(self):
        return list(wn.all_synsets())

    def get_all_lemmas(self, replace_ws=True):
        all_wn_lemmas = list(wn.all_lemma_names())
        if replace_ws:
            all_wn_lemmas = [lemma.replace('_', ' ') for lemma in all_wn_lemmas]
        return all_wn_lemmas

    def get_all_sks(self):
        # return list(self.map_sk2syn.keys())
        return self.map_sk2syn.keys()

    def get_all_lexnames(self):
        lexnames = set()
        for syn in self.get_all_syns():
            lexnames.add(self.syn2lexname(syn))
        return list(lexnames)

    def get_all_csis(self):
        # TO-DO
        pass

    def convert_postag(self, postag):
        # merges ADJ with ADJ_SAT
        postags_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r', 'ADJ_SAT': 'a'}
        if postag in postags_map.values():
            return postag
        elif postag in postags_map:
            return postags_map[postag]
        else:
            # raise exception
            return None


if __name__ == '__main__':

    wn_utils = WN_Utils()
    print(wn_utils.lemma2sks('hydrophobia'))
