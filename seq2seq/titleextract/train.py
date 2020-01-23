# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import pickle
import jieba
import json
import random

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

#

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


kk =500
corpus = []

with open('./news2016zh_valid.json','r',encoding='utf-8') as f:
    textlineslinee = f.read()
    textlineslines = textlineslinee.split('\n')
    random.shuffle(textlineslines)
    # print(len(textlineslines))
    for line in textlineslines[:kk]:
        try:
            newsjoson=json.loads(line)
            title = newsjoson['title']
            content = newsjoson['content']
            corpus.append((' '.join(jieba.cut(content)),' '.join(jieba.cut(title))))
        except Exception as e:
            # raise e
            print(str(e))
# exit()
print(corpus)
# exit()
# corpus = corpus[:]
maxFeature=10000

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘的我你他好很',split=" ",num_words=maxFeature)  #创建一个Tokenizer对象


corpusX=[]
corpusY=[]
for content,title in corpus:
    corpusX.append(content)
    corpusY.append('\t ' + title + ' \n')


tokenizer.fit_on_texts(corpusX+corpusY)
word2id=tokenizer.word_index #得到每个词的编号
id2word=tokenizer.index_word #得到每个编号对应的词
# print(vocab)
# print(id2word)
# print(len(word2id))


x_train_word_ids=tokenizer.texts_to_sequences(corpusX)
y_train_word_ids_=tokenizer.texts_to_sequences(corpusY)
y_train_word_ids = [idlist[1:] for idlist in y_train_word_ids_]
# print(x_train_word_ids)
# x_train_word_ids = np.array(x_train_word_ids)

# max_encoder_seq_length = max([len(wdlist) for wdlist in x_train_word_ids])
# max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
# print(max_encoder_seq_length)

trainX = pad_sequences(x_train_word_ids,maxlen=500, dtype='int')
trainY_ = pad_sequences(y_train_word_ids_,maxlen=30, dtype='int')
trainY = pad_sequences(y_train_word_ids,maxlen=30, dtype='int')

# trainX = trainX.reshape(*trainX.shape, 1)
# trainY_ = trainY_.reshape(*trainY_.shape, 1)
# trainY = trainY.reshape(*trainY.shape, 1)

print(trainX)
print(trainY.shape)

# 输入时间序列的长度，即 用多少个连续样本预测一个输出

# exit()

# with open("en.file", "wb") as f:
#     pickle.dump({'id2en':id2en,'en2id':en2id}, f)
# with open("zh.file", "wb") as f:
#     pickle.dump({'id2ch':id2ch,'ch2id':ch2id}, f)

# with open("config.file", "wb") as f:
#     pickle.dump({'max_encoder_seq_length':max_encoder_seq_length,'max_decoder_seq_length':max_decoder_seq_length}, f)

######################## model ##############################

EN_VOCAB_SIZE = len(word2id)
CH_VOCAB_SIZE = len(word2id)
HIDDEN_SIZE = 256

LEARNING_RATE = 0.002
BATCH_SIZE = 100
EPOCHS = 10


def seq2seq_model(input_length,output_sequence_length,vocab_size):
    model = tf.keras.models.Sequential()
    model.add(Embedding(input_dim=vocab_size,output_dim = 128,input_length=500))
    model.add(Bidirectional(GRU(128, return_sequences = False)))
    model.add(Dense(128, activation="relu"))
    model.add(RepeatVector(30))
    model.add(Bidirectional(GRU(128, return_sequences = True)))
    model.add(TimeDistributed(Dense(vocab_size, activation = 'softmax')))
    model.compile(loss = sparse_categorical_crossentropy, 
                  optimizer = Adam(1e-3))
    model.summary()
    return model
model = seq2seq_model(500,30,len(word2id))





opt = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(trainX, trainY,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)
# model.fit_generator(generate_arrays_from_file(),steps_per_epoch=1,epochs=EPOCHS)

# model.save('ss2ss_test.h5')



# print(ch2id['\n'])
# print(ch2id['\t'])


# exit()



# # ################################################### predict ############################################################
# #预测模型中的encoder和训练中的一样，都是输入序列，输出几个状态。而decoder和训练中稍有不同，因为训练过程中的decoder端的输入是可以确定的，因此状态只需要初始化一次，而预测过程中，需要多次初始化状态，因此将状态也作为模型输入。

# # encoder模型和训练相同
# encoder_model = keras.Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])

# # 预测模型中的decoder的初始化状态需要传入新的状态
# decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,))
# decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,))
# decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,))
# decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,))

# # 使用传入的值来初始化当前模型的输入状态
# decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
# decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
# decoder_outputs = decoder_dense(decoder_h2)

# decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
#                       [decoder_outputs, state_h1, state_c1, state_h2, state_c2])


# task =[
#     # '张柏芝 张雨绮 吴卓林 谢霆锋',x_train_word_ids
#     # '农村 小伙 城里 偷笑 房子',
#     # '荷花 西湖 金粟词话 采莲女 林逋 荷叶',
#     # '荷花 西湖',
#     # '杨迪 脱口秀',
# ]





# input_sen = "女子误搭假出租找零和发票均为假昨晚6时许，北京女子小暖(化名)在路边打了一辆出租车，下车时司机找了她50元，回家发现竟是假钱。小暖赶紧拨打了的票上的电话，却发现的票不仅是假的，就连这辆出租车也是假的。据小暖介绍，她从世贸天阶路边拦了一辆出租车，前往春秀路。到达目的地后，“的哥”给她找了零钱，其中有一张50元纸币。她像往常一样，索取了的票，就下了车。下车后，她才发现，刚才“的哥”找她的这张50块钱有点不对劲，仔细观察后，确认是一张假币。小暖急了，她找出那张的票，拨打了上面的出租车公司电话想投诉，可是接电话的人告诉她这根本不是出租车公司。小暖这才意识到，这个“的哥”给她的是张假的票，而且从车到票再到钱，没有一样是真的。小暖在个人微博上晒出了假的票和假币，并披露了这件事。目前，她已在线向平安北京反映了此事，希望有关部门能加强对类似假出租的打击力度。如何辨别假出租车呢?1.“假出租车”只能“克隆”正规出租车的外形，却无法“克隆”正规出租车的刷卡、打票设备，所以“黑车”司机总会以借口，死活要乘客付现金。2.“假出租车”使用的都是假发票，乘客如果发现发票名称与顶灯、车身上的公司标识不符，该车十有八九就是假冒的。但上述识别都是在乘客已上车的情况下，比较被动。3.正规出租车副驾驶座位置都会贴有出租车司机的个人信息和运营许可证，并有该出租车公司的公章等信息。“假出租车”一般都不会有，就算有也都是凑合一下，很明显可以看出是假的。本文来源前瞻网，未经前瞻网书面授权，禁止转载，违者将被追究法律责任！"
# char2id = [ word2id.get(i,'') for i in jieba.cut(input_sen)]
# while '' in char2id:
#     char2id.remove('')
# print(char2id)
# # exit()
# input_data = pad_sequences([char2id],500)

# result = model.predict(input_data)[0][-len(input_sen):]
# result_label = [np.argmax(i) for i in result]
# # dict_res = {i:j for j,i in word2id.items()}
# # print([id2word.get(i) for i in  result_label])
# print(result_label)



# h1, c1, h2, c2 = encoder_model.predict(input_data)
# # target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
# # target_seq[0, 0, word2id['\t']] = 1
# outputs = []
# while True:
#     output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
#     sampled_token_index = np.argmax(output_tokens[0, -1, :])
#     outputs.append(sampled_token_index)
#     target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
#     target_seq[0, 0, sampled_token_index] = 1
#     if sampled_token_index == word2id['\n'] or len(outputs) > 20: break

# # print(en_data[k])
# # print(enchar)
# print(''.join([id2ch[i] for i in outputs]))
