from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pdb
import argparse
import numpy as np
import torch
from scipy import linalg
import nltk
nltk.download('punkt')
import pdb
from modules.models import InferSent 


def calc_fid(model, true_data, gen_data, eps=1e-6, cond=False):
    reference = list()
    if cond == False:
        with open(true_data) as real_data:
            for text in real_data:
                reference.append(text)
    if cond == True:
        with open(true_data) as real_data:
            for text in real_data:
                reference.append(text.split(" ",1)[1])

    reference_2 = list()
    if cond == False:
        with open(gen_data) as fake_data:
            for text in fake_data:
                reference_2.append(text)    
    if cond == True:
        with open(gen_data) as fake_data:
            for text in fake_data:
                reference_2.append(text.split(" ",1)[1])  

    model.build_vocab(reference + reference_2, tokenize=True)
    embeddings = model.encode(reference, bsize=128, tokenize=True) 
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    
    embeddings_2 = model.encode(reference_2, bsize=128, tokenize=True) 
    mu_2 = np.mean(embeddings_2, axis=0)
    sigma_2 = np.cov(embeddings_2, rowvar=False)

    assert mu.shape == mu_2.shape
    assert sigma.shape == sigma_2.shape

    diff = mu - mu_2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma.dot(sigma_2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma.shape[0]) * eps
        covmean = linalg.sqrtm((sigma + offset).dot(sigma_2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma)
            + np.trace(sigma_2) - 2 * tr_covmean)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_data', type=str)
    parser.add_argument('--gen_data', type=str)
    parser.add_argument('--infersent_path', type=str)
    parser.add_argument('--glove_path', type=str)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cond', type=bool, default=False)
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(args.infersent_path))
    infersent.set_w2v_path(args.glove_path)

    infersent = infersent.to(args.device)  

    fid = calc_fid(infersent, args.true_data, args.gen_data,args.cond)
    print('FID: {:0.3f}'.format(fid))