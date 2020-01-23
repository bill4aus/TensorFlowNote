#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


import codecs
import collections
from six.moves import cPickle
import numpy as np
import os

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "text_test.txt")
        vocab_file = os.path.join(data_dir, "vocab.zh.pkl")
        self.preprocess(input_file, vocab_file) #处理数据 把文本处理句子成 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 78, 1, 1, 530, 86, 108, 107, 1, 810, 3, 427, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq] #进行频率的筛选的过程
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} #每个不重复词的位置id索引字典
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]  #把文本数据全部放到一个list  [['上学'，'生活'],,,,,,,]

        self.vocab, self.words = self.build_vocab(lines)  # 一个词的列表 一个是关于字典的词 以及独一无二的词的索引位置
        self.vocab_size = len(self.words) #统计词量
        #print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f) #把字典的词频系列写进去文档保存
        raw_data = [[0] * self.seq_length+[self.vocab.get(w, 1) for w in line] +[2] * self.seq_length for line in lines]
        self.raw_data = raw_data  #把每句话转换成 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 78, 1, 1, 530, 86, 108, 107, 1, 810, 3, 427, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        #对应的数值为词表里面的id下标，前面的0和后面的2
    def create_batches(self):
        xdata, ydata = list(),list()
        for row in self.raw_data: #row 为每一个句子向量化的东西
            for ind in range(self.seq_length, len(row)):  #20 60
                xdata.append(row[ind-self.seq_length:ind])
                ydata.append([row[ind]]) #下一个单词  目标单词
                # print(xdata)
                # print(ydata)
                # input()
                #[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
                #[[1], [94]]
        self.num_batches = int(len(xdata) / self.batch_size)
        # print(self.num_batches)
        # input()
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])
        # print(xdata[:self.num_batches * self.batch_size])
        # print(ydata)
        # input()
        self.x_batches = np.split(xdata, self.num_batches, 0) #分割批次进行喂数据
        self.y_batches = np.split(ydata, self.num_batches, 0)#分割批次进行喂数据

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
