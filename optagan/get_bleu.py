from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pdb
import argparse
import numpy as np
import nltk
nltk.download('punkt')
import pdb
from nltk.translate.bleu_score import SmoothingFunction
from multiprocessing import Pool

def calc_blue(true_data, gen_data, ngram, sent, cond):
    if sent == 0:
        reference = list()
        if cond == False:
            with open(true_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text.lower())
                    reference.append(text)
        if cond == True:
            with open(true_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text.split(" ",1)[1].lower())
                    reference.append(text)  
    
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        j = 0
        if cond == False:
            with open(gen_data) as test_data:
                for hypothesis in test_data:
                    j += 1
                    hypothesis = nltk.word_tokenize(hypothesis.lower())
                    bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1))
                    if (j % 1000) == 0:
                        print("Evaluated 1000 sentences.")
        if cond == True:
            with open(gen_data) as test_data:
                for hypothesis in test_data:
                    j += 1
                    hypothesis = nltk.word_tokenize(hypothesis.split(" ",1)[1].lower())
                    bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1))
                    if (j % 1000) == 0:
                        print("Evaluated 1000 sentences.")
    else:
        reference = list()
        if cond == False:
            with open(true_data) as real_data:
                for i, text in enumerate(real_data):
                    if i > sent:
                        break
                    text = nltk.word_tokenize(text.lower())
                    reference.append(text)
        if cond == True:
            with open(true_data) as real_data:
                for i, text in enumerate(real_data):
                    if i > sent:
                        break
                    text = nltk.word_tokenize(text.split(" ",1)[1].lower())
                    reference.append(text)
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        j = 0
        if cond == False:
            with open(gen_data) as test_data:
                for i, hypothesis in enumerate(test_data):
                    if i > sent:
                        break              
                    j += 1
                    hypothesis = nltk.word_tokenize(hypothesis.lower())
                    bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1))
                    if (j % 1000) == 0:
                        print("Evaluated 1000 sentences.")            
        if cond == True:
            with open(gen_data) as test_data:
                for i, hypothesis in enumerate(test_data):
                    if i > sent:
                        break              
                    j += 1
                    hypothesis = nltk.word_tokenize(hypothesis.split(" ",1)[1].lower())
                    bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1))
                    if (j % 1000) == 0:
                        print("Evaluated 1000 sentences.")            
    return sum(bleu)/len(bleu)
    
def calc_blue_parallel(true_data, gen_data, ngram, sent, cond):
    if sent == 0:
        reference = list()
        if cond == False:
            with open(true_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text.lower())
                    reference.append(text)
        if cond == True:
            with open(true_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text.split(" ",1)[1].lower())
                    reference.append(text)

        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        if cond == False:
            with open(gen_data) as test_data:
                for hypothesis in test_data:
                    hypothesis = nltk.word_tokenize(hypothesis.lower())
                    bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(reference, hypothesis, weight, SmoothingFunction().method1)))
        if cond == True:
            with open(gen_data) as test_data:
                for hypothesis in test_data:
                    hypothesis = nltk.word_tokenize(hypothesis.split(" ",1)[1].lower())
                    bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(reference, hypothesis, weight, SmoothingFunction().method1)))
        score = 0
        cnt = 0
        for i in bleu:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
    else:
        reference = list()
        if cond == False:
            with open(true_data) as real_data:
                for i, text in enumerate(real_data):
                    if i > sent:
                        break
                    text = nltk.word_tokenize(text.lower())
                    reference.append(text)
        if cond:
            with open(true_data) as real_data:
                for i, text in enumerate(real_data):
                    if i > sent:
                        break
                    text = nltk.word_tokenize(text.split(" ",1)[1].lower())
                    reference.append(text)
    
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        if cond == False:
            with open(gen_data) as test_data:
                for i, hypothesis in enumerate(test_data):
                    if i > sent:
                        break              
                    hypothesis = nltk.word_tokenize(hypothesis.lower())
                    bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(reference, hypothesis, weight, SmoothingFunction().method1)))
        if cond:
            with open(gen_data) as test_data:
                for i, hypothesis in enumerate(test_data):
                    if i > sent:
                        break              
                    hypothesis = nltk.word_tokenize(hypothesis.split(" ",1)[1].lower())
                    bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(reference, hypothesis, weight, SmoothingFunction().method1)))
        score = 0.0
        cnt = 0
        for i in bleu:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()           
    return score / cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str)
    parser.add_argument('--hypo', type=str)
    parser.add_argument('--ngram', type=int, default=2)
    parser.add_argument('--max_sent', type=int, default=0, help="Number of sentences as ref and hypo, 0 means all")
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--cond', type=bool, default=False)
    
    args = parser.parse_args()
    func = calc_blue_parallel if args.parallel else calc_blue
    blue_res = func(args.ref, args.hypo, args.ngram, args.max_sent, args.cond)
    print("Finished calculating Bleu")
    print('Bleu-{}: {:0.3f}'.format(args.ngram,blue_res))