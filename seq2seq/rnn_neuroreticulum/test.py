import pickle
from functools import reduce

import jieba
import numpy as np
import os
import math
def get_possible(word1,word2):
    data_dir = "./data"
    vocab_file = os.path.join(data_dir, "vocab.zh.pkl")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f, encoding='bytes')
    word_emb = np.load('nnlm_word_embeddings.zh.npy')

    vocab={j:i for i,j in enumerate(vocab)}
    try:
        word1_id = vocab[word1]
        word2_id = vocab[word2]
    except:
        return 1  #防止没有词汇直接pass 过去看前后词的影响

    word1_emb = word_emb[word1_id]
    word2_emb = word_emb[word2_id]
    # print(word1_emb) #1*50 的向量
    return cosin_distance(word1_emb,word2_emb)

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return math.log(abs(dot_product / ((normA * normB) ** 0.5)),0.5)
if __name__=='__main__':
    while 1:
        res_posb=[]
        text=input('请输入一句话：')
        ls=jieba.lcut(text)
        if len(ls)<=1:
            print(text,'-----------','这是一句话')
            continue
        for i in range(len(ls)-1):
            word_1,word_2=ls[i:i+2]
            res_posb.append(get_possible(word_1,word_2))
        result= reduce(lambda x, y: x * y, res_posb)
        print('句子概率值：',result)


