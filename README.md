# bert-disambiguation

# Example creating sense vectors

    $ python create_1nn_vecs.py -nlm_id bert-base-uncased -out_path vectors/sense_vecs.bert-base-uncased.txt

# Example evaluating sense vectors

    $ python eval_1nn.py -nlm_id bert-base-uncased -sv_path vectors/sense_vecs.bert-base-uncased.txt
    