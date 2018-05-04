%%writefile data.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import Counter, OrderedDict

from copy import deepcopy

flatten = lambda l: [item for sublist in l for item in sublist]

import json
import nltk
from nltk import word_tokenize
import os


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj

def token2id(context_tokens):
    idx = 0
    token_map = dict()
    size = 0
    for i, token in enumerate(context_tokens):
        length = len(str(token)) + 1 
        token_map[size] = [token, i]
       
        size = size + length
    return token_map
    
def LoadSquadDataset(data_path,max_len=600):
    dataset = json.load(open(data_path,'r'))
    data_p=[]
    qn, an = 0, 0
    skipped = 0
    
    for articles_id in range(len(dataset['data'])):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            context_tokens = word_tokenize(context)
            if len(context_tokens) > max_len: 
                continue

            answer_map = token2id( context_tokens)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = word_tokenize(question)
                answers = qas[qid]['answers']
                qn += 1
                num_answers = list(range(1))

                for ans_id in num_answers:
                    text = qas[qid]['answers'][ans_id]['text']
                    text_tokens = word_tokenize(text)
                    answer_start = qas[qid]['answers'][ans_id]['answer_start']
                    answer_end = answer_start + len(text)
                    last_word_answer = len(text_tokens[-1]) # add one to get the first char

                    try:
                        a_start_idx = answer_map[answer_start][1]
                        a_end_idx = answer_map[answer_end - last_word_answer][1]
                        data_p.append([context_tokens,question_tokens,text_tokens,a_start_idx,a_end_idx])
                        
                    except Exception as e:
                        skipped += 1
                    an += 1
    
    print("Skipped {}, {} question/answer".format(skipped, len(data_p)))
    
    return data_p
    
        
def prepareBatch(data, batch_size):
    start = 0
    end = batch_size
    while end < len(data):
        batch = data[start:end]
        temp = end
        end = end + batch_size
        start = temp
        yield batch
    
    if end >= len(data):
        batch = data[start:]
        yield batch
    
def padSentences(batch,w_to_ix): # for Squad dataset
    doc, q, s, e = list(zip(*batch))
    doc_max = max([p.size(1) for p in doc])
    ques_max = max([qq.size(1) for qq in q])
    
    doc_p, q_p = [],[]
    for i in range(len(batch)):
        nb_pads = doc_max - doc[i].size(1)
        if (nb_pads > 0):
            sentence = nb_pads * [w_to_ix['<pad>']] 
            sentence = torch.cuda.LongTensor(sentence)
            sentence = Variable(sentence).view(1,-1)
            doc_p.append(torch.cat([doc[i], sentence],1))
        else:
            doc_p.append(doc[i])
            
        nb_pads = ques_max - q[i].size(1)
        if (nb_pads > 0):
            sentence = nb_pads * [w_to_ix['<pad>']] 
            sentence = torch.cuda.LongTensor(sentence)
            sentence = Variable(sentence).view(1,-1)
            q_p.append(torch.cat([q[i], sentence],1))
        else:
            q_p.append(q[i])

    docs  = torch.cat(doc_p)
    questions = torch.cat(q_p)
    
    doc_lens = torch.cuda.LongTensor([list(map(lambda s: s == 0, t.data)).count(False) for t in docs])
    question_lens = torch.cuda.LongTensor([list(map(lambda s: s == 0, t.data)).count(False) for t in questions])
    
    starts = torch.cat(s)
    ends = torch.cat(e)
    return docs, questions, doc_lens, question_lens, starts, ends
    
def preprop(dataset, token2idx=None):
    docs,qu,_,start,end = list(zip(*dataset))
    
    if token2idx is None:
        token2idx={'<pad>':0,'<unk>':1} 

        for tk in flatten(docs)+flatten(qu):
            if tk not in token2idx.keys():
                token2idx[tk]=len (token2idx)

    print("Successfully Build %d vocabs" % len(token2idx))
    
    data_p=[]
    for i in range(len(docs)):
        temp=[]
        sentence = [ 
            token2idx[t] if t in token2idx.keys() else token2idx["<unk>"] 
            for t in docs[i] ] 
        temp.append (Variable (torch.cuda.LongTensor (sentence).unsqueeze(0)))
        sentence = [ 
            token2idx[t] if t in token2idx.keys() else token2idx["<unk>"] 
            for t in qu[i] ]
        temp.append (Variable (torch.cuda.LongTensor (sentence).unsqueeze(0)))
        temp.append (Variable (torch.cuda.LongTensor ([start[i]])).unsqueeze(0))
        temp.append (Variable (torch.cuda.LongTensor ([end[i]])).unsqueeze(0))
        data_p.append (temp)
    print("Preprop Complete!")

    
    return token2idx, data_p

  