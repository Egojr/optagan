import torch
import os
import numpy as np
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import SmoothingFunction
from multiprocessing import Pool
import torch.nn.functional as F
import torch.distributions as distributions
from transformers.modeling_utils import top_k_top_p_filtering

def safe_log(z):
    return torch.log(z + 1e-7)

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def generate_grid(zmin, zmax, dz, device, ndim=2):
    """generate a 1- or 2-dimensional grid
    Returns: Tensor, int
        Tensor: The grid tensor with shape (k^2, 2),
            where k=(zmax - zmin)/dz
        int: k
    """

    if ndim == 2:
        x = torch.arange(zmin, zmax, dz)
        k = x.size(0)

        x1 = x.unsqueeze(1).repeat(1, k).view(-1)
        x2 = x.repeat(k)

        return torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1).to(device), k

    elif ndim == 1:
        return torch.arange(zmin, zmax, dz).unsqueeze(1).to(device)

def calc_blue_parallel_func(reference, test_data, ngram, sent, cond=False):
    if sent == 0:
        refs = list()
        if cond == False:
            for text in reference:
                text = nltk.word_tokenize(text.lower())
                refs.append(text)
        if cond == True:
            for text in reference:
                text = nltk.word_tokenize(text.split(" ",1)[1].lower())
                refs.append(text)
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        if cond==False:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis.lower())
                bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(refs, hypothesis, weight, SmoothingFunction().method1)))
        if cond==True:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis.split(" ",1)[1].lower())
                bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(refs, hypothesis, weight, SmoothingFunction().method1)))

        score = 0
        cnt = 0
        for i in bleu:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
    else:
        refs = list()
        if cond == False:
            for i, text in enumerate(reference):
                if i > sent:
                    break
                text = nltk.word_tokenize(text.lower())
                refs.append(text)
        if cond == True:
            for i, text in enumerate(reference):
                if i > sent:
                    break
                text = nltk.word_tokenize(text.split(" ",1)[1].lower())
                refs.append(text)
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        if cond == False:
            for i, hypothesis in enumerate(test_data):
                if i > sent:
                    break              
                hypothesis = nltk.word_tokenize(hypothesis.lower())
                bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(refs, hypothesis, weight, SmoothingFunction().method1)))
        if cond == True:
            for i, hypothesis in enumerate(test_data):
                if i > sent:
                    break              
                hypothesis = nltk.word_tokenize(hypothesis.split(" ",1)[1].lower())
                bleu.append(pool.apply_async(nltk.translate.bleu_score.sentence_bleu, args=(refs, hypothesis, weight, SmoothingFunction().method1)))
        score = 0.0
        cnt = 0
        for i in bleu:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()           
    return  score / cnt

def pad_seq(list_, value, length):
    out = []
    mask = []
    for elem in list_:
        out.append([*elem[:length], *[value] * (length - len(elem[:length]))])
        mask.append([*[1] * len(elem[:length]), *[0] * (length - len(elem[:length]))])
    return out, mask

def rollout(act_model, latent_z, tokenizer_decoder, length, batch_size, temperature, detac=True):
    act_vals = []
    act_text = []
    # act_logits = []
    entropy = []
    act_logprobs = []
    context = tokenizer_decoder.encode('<BOS>')
    context = torch.tensor(context, dtype=torch.long, device="cuda")
    context = context.squeeze(0).repeat(batch_size, 1)
    act_generated = context

    # with torch.no_grad():
    for _ in range(length):
        act_inputs = {'input_ids': act_generated, 'past': latent_z}
        outputs = act_model(**act_inputs, detach=detac)
        next_token_logits = outputs[0][:, -1, :] / temperature
        act_vals.append(outputs[2][:,-1])
        # act_logits.append(next_token_logits)
        probs = F.softmax(next_token_logits, dim=-1)
        dist = distributions.Categorical(probs)
        entropy.append(dist.entropy())
        next_token = dist.sample()
        act_logprobs.append(dist.log_prob(next_token))
        act_generated = torch.cat([act_generated, next_token.unsqueeze(-1)], dim=-1)
        if all(tokenizer_decoder.encode('<EOS>')[0] in sent for sent in act_generated):
            break

    for i in range(len(act_generated)):
        # clean up EOS after tokens
        sent = act_generated[i,:].tolist()
        if tokenizer_decoder.encode('<EOS>')[0] in sent:
            sent = sent[: sent.index(50259) + 1] # Stop when EOS appears
        text_x1 = tokenizer_decoder.decode(sent, clean_up_tokenization_spaces=False)
        text_x1 = text_x1.split()[1:-1]
        text_x1 = ' '.join(text_x1)
        act_text.append(text_x1)
    return act_text, act_logprobs, act_vals, entropy

def rollout_test(act_model, latent_z, tokenizer_decoder, length, batch_size, top_k_val, top_p_val):
    act_text = []
    context = tokenizer_decoder.encode('<BOS>')
    context = torch.tensor(context, dtype=torch.long, device="cuda")
    context = context.squeeze(0).repeat(batch_size, 1) # batch_size
    act_generated = context

    with torch.no_grad():
        for _ in range(length):
            act_inputs = {'input_ids': act_generated, 'past': latent_z}
            outputs = act_model(**act_inputs)  
            next_token_logits = outputs[0][:, -1, :]
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k_val, top_p_val)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            act_generated = torch.cat([act_generated, next_token.unsqueeze(-1)], dim=-1)
            if all(tokenizer_decoder.encode('<EOS>')[0] in sent for sent in act_generated):
                break

    for i in range(len(act_generated)):
        # clean up EOS after tokens
        sent = act_generated[i,:].tolist()
        if tokenizer_decoder.encode('<EOS>')[0] in sent:
            sent = sent[: sent.index(50259) + 1] # Stop when EOS appears
        text_x1 = tokenizer_decoder.decode(sent, clean_up_tokenization_spaces=False)
        text_x1 = text_x1.split()[1:-1]
        text_x1 = ' '.join(text_x1)
        act_text.append(text_x1)
    return act_text
