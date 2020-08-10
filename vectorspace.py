import numpy as np


class VSM(object):

    def __init__(self, vecs_path, normalize=True):
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        self.load_txt(vecs_path)

        if normalize:
            self.normalize()

    def load_txt(self, vecs_path):
        self.vectors = []
        with open(vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))

                # if line_idx % 100000 == 0:
                #     print(line_idx)

        self.vectors = np.vstack(self.vectors)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def normalize(self, norm='l2'):
        self.vectors = (self.vectors.T / np.linalg.norm(self.vectors, axis=1)).T

    def get_vec(self, label):
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar_vec(self, vec, topn=10):
        # TO-DO: tidy up...
        # sims = np.dot(self.vectors, vec).astype(np.float32)
        sims = np.dot(self.vectors, vec)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()
