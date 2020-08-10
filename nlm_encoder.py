import mkl
mkl.set_dynamic(0)
mkl.set_num_threads(6)

import torch as th
import numpy as np

from transformers import BertModel, BertTokenizer


class TransformerEncoder():

    def __init__(self, nlm_config):
        self.nlm_config = nlm_config
        self.nlm_model = None
        self.nlm_tokenizer = None

        self.load_nlm(nlm_config['model_name_or_path'])


    def load_nlm(self, model_name_or_path):

        if model_name_or_path.startswith('bert-'):
            self.nlm_model = BertModel.from_pretrained(model_name_or_path, output_hidden_states=True)
            self.nlm_tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

            self.cls_encoding = self.nlm_tokenizer.encode(self.nlm_tokenizer.cls_token, add_special_tokens=False)[0]
            self.sep_encoding = self.nlm_tokenizer.encode(self.nlm_tokenizer.sep_token, add_special_tokens=False)[0]
            self.pad_encoding = self.nlm_tokenizer.encode(self.nlm_tokenizer.pad_token, add_special_tokens=False)[0]

        else:
            # TO-DO
            raise(BaseException('Invalid model_name - %s' % model_name_or_path))

        self.nlm_model.eval()
        self.nlm_model.to('cuda')


    def encode_token(self, token):
        # returns list of subtokens
        return self.nlm_tokenizer.encode(token, add_special_tokens=False)


    def get_encodings(self, tokens):
        return [self.encode_token(t) for t in tokens]


    def flatten_encodings(self, encodings):
        return sum(encodings, [])


    def add_special_encodings(self, encodings):

        model_name_or_path = self.nlm_config['model_name_or_path']

        if model_name_or_path.startswith('bert-'):
            return [self.cls_encoding] + encodings + [self.sep_encoding]

        else:
            # TO-DO
            raise(BaseException('Invalid model_name - %s' % model_name_or_path))


    def add_padding_encodings(self, encodings, max_len):
        encodings += [self.pad_encoding] * (max_len - len(encodings))
        return encodings


    def get_attention_mask(self, encodings):
        att_mask = []
        for enc in encodings:
            if enc == self.pad_encoding:
                att_mask.append(0)
            else:
                att_mask.append(1)
        return att_mask


    def merge_subword_embeddings(self, tokens, encodings, embeddings, return_tokens=True):
        # align and merge subword embeddings
        tok_embeddings = []
        encoding_idx = 0
        for tok, tok_encodings in zip(tokens, encodings):

            if self.nlm_config['subword_op'] == 'mean':
                tok_embedding = th.zeros(embeddings.shape[-1]).to('cuda')
                for _ in tok_encodings:
                    tok_embedding += embeddings[encoding_idx]
                    encoding_idx += 1
                tok_embedding = tok_embedding / len(tok_encodings)  # avg of subword embs

            elif self.nlm_config['subword_op'] == 'first':
                tok_embedding = embeddings[encoding_idx]
                for _ in tok_encodings:
                    encoding_idx += 1  # just move idx

            else:
                raise(BaseException('Invalid subword_op - %s' % self.nlm_config['subword_op']))

            tok_embedding = tok_embedding.detach().cpu().numpy()

            if return_tokens:
                tok_embeddings.append((tok, tok_embedding))
            else:
                tok_embeddings.append(tok_embedding)

        return tok_embeddings


    def get_num_features(self, tokens, n_special_toks=2):
        return len(self.get_encodings(tokens)) + n_special_toks


    def get_num_subtokens(self, tokens):
        return len(self.get_encodings(tokens))


    def get_token_embeddings_batch(self, batch_sent_tokens, return_tokens=True):

        batch_sent_encodings = [self.get_encodings(sent_tokens) for sent_tokens in batch_sent_tokens]        
        batch_max_len = max([len(self.flatten_encodings(e)) for e in batch_sent_encodings]) + 2

        # prepare nlm input
        input_ids, input_mask = [], []
        for sent_tokens, sent_encodings in zip(batch_sent_tokens, batch_sent_encodings):

            sent_encodings = self.flatten_encodings(sent_encodings)
            sent_encodings = self.add_special_encodings(sent_encodings)
            sent_encodings = self.add_padding_encodings(sent_encodings, batch_max_len)
            input_ids.append(sent_encodings)

            sent_attention = self.get_attention_mask(sent_encodings)
            input_mask.append(sent_attention)

            assert len(sent_encodings) == len(sent_attention)


        input_ids = th.tensor(input_ids).to('cuda')
        input_mask = th.tensor(input_mask).to('cuda')
        with th.no_grad():

            if self.nlm_config['model_name_or_path'].startswith('xlnet-'):
                pooled, batch_hidden_states = self.nlm_model(input_ids, attention_mask=input_mask)
                last_layer = batch_hidden_states[-1]

            else:
                last_layer, pooled, batch_hidden_states = self.nlm_model(input_ids, attention_mask=input_mask)

        # select layers of interest
        sel_hidden_states = [batch_hidden_states[i] for i in self.nlm_config['layers']]

        # merge subword embeddings
        merged_batch_hidden_states = []
        for layer_hidden_states in sel_hidden_states:
            merged_layer_hidden_states = []
            for sent_idx, sent_embeddings in enumerate(layer_hidden_states):
                sent_embeddings = sent_embeddings[1:-1]  # ignoring special tokens

                sent_tokens = batch_sent_tokens[sent_idx]
                sent_encodings = batch_sent_encodings[sent_idx]

                sent_embeddings = self.merge_subword_embeddings(sent_tokens, sent_encodings, sent_embeddings, return_tokens=return_tokens)
                merged_layer_hidden_states.append(sent_embeddings)
            merged_batch_hidden_states.append(merged_layer_hidden_states)

        # combine layers
        combined_batch_embeddings = []
        for sent_idx, sent_tokens in enumerate(batch_sent_tokens):

            combined_sent_embeddings = []
            for tok_idx in range(len(sent_tokens)):
                tok_layer_vecs = []
                for layer_idx in range(len(merged_batch_hidden_states)):
                    tok_layer_vecs.append(merged_batch_hidden_states[layer_idx][sent_idx][tok_idx][1])

                if len(tok_layer_vecs) == 1:
                    tok_combined_vec = tok_layer_vecs[0]
                
                else:
                    tok_layer_vecs = np.array(tok_layer_vecs)

                    if self.nlm_config['layer_op'] == 'sum':
                        tok_combined_vec = tok_layer_vecs.sum(axis=0)

                tok = merged_batch_hidden_states[layer_idx][sent_idx][tok_idx][0]
                combined_sent_embeddings.append((tok, tok_combined_vec))
            
            combined_batch_embeddings.append(combined_sent_embeddings)

        return [combined_batch_embeddings]


    def token_embeddings(self, batch_sent_tokens, return_tokens=True):
        return self.get_token_embeddings_batch(batch_sent_tokens, return_tokens=return_tokens)


    def is_valid(self, tokens):
        encodings = self.flatten_encodings(self.get_encodings(tokens))
        if (len(encodings) + 2) > self.nlm_config['max_seq_len']:
            return False
        else:
            return True


if __name__ == '__main__':

    encoder_cfg = {
        'model_name_or_path': 'bert-large-uncased',
        'weights_path': '',
        'min_seq_len': 0,
        'max_seq_len': 32,
        'layers': [-1, -2, -3, -4],
        'layer_op': 'sum',
        'subword_op': 'mean'
    }

    enc = TransformerEncoder(encoder_cfg)

    tokenized_sents = [['Hello', 'world', '!'], ['Bye', 'now', ',', 'see', 'you', 'later', '?']]
    
    embs = enc.get_token_embeddings_batch(tokenized_sents)
