# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import pickle
import jieba
import json
import random
#

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


kk =50
corpus = []

with open('./news2016zh_valid.json','r',encoding='utf-8') as f:
    textlineslinee = f.read()
    textlineslines = textlineslinee.split('\n')
    random.shuffle(textlineslines)
    # print(len(textlineslines))
    for line in textlineslines[:kk]:
        try:

            newsjoson=json.loads(line)
            # print(newsjoson)
            # exit()

            linearray=line.split('_!_')
            title = newsjoson['title']
            content = newsjoson['content']
            # print('title:'+title)
            # print('content:'+content)

            corpus.append((' '.join(jieba.cut(content)),' '.join(jieba.cut(title))))

        except Exception as e:
            # raise e
            print(str(e))


# exit()
# print(corpus)
# exit()

corpus = corpus[:]


# exit()
#
# with open('../../datasets/cmn.txt', 'r', encoding='utf-8') as f:
#     data = f.read()
#     data = data.split('\n')
#     # print(data)
#     # data = data[:100]
# print(data[-500:])
# print(len(data))
# data = data[0:5000]
# # exit()
#
# en_data = [line.split('\t')[0] for line in data]
# ch_data = ['\t' + line.split('\t')[1] + '\n' for line in data]

en_data = [content for content,title in corpus]
ch_data = ['\t ' + title + ' \n' for content,title in corpus]


# print('英文数据:\n', en_data[:10])
print('\n中文数据:\n', ch_data[:10])
# exit()

# 分别生成中英文字典
# en_vocab = set(''.join(en_data))
en_vocab = set(' '.join(en_data).split(' '))
id2en = list(en_vocab)
en2id = {c:i for i,c in enumerate(id2en)}

# ch_vocab = set(''.join(ch_data))
ch_vocab = set(' '.join(ch_data).split(' '))

id2ch = list(ch_vocab)
ch2id = {c:i for i,c in enumerate(id2ch)}

# print('\n英文字典:\n', en2id)
# print('\n中文字典共计\n:', ch2id)


with open("en.file", "wb") as f:
    pickle.dump({'id2en':id2en,'en2id':en2id}, f)
with open("zh.file", "wb") as f:
    pickle.dump({'id2ch':id2ch,'ch2id':ch2id}, f)

# 建立字典,将文本数据映射为数字数据形式。
# en_num_data = [[en2id[en] for en in line ] for line in en_data]
# ch_num_data = [[ch2id[ch] for ch in line] for line in ch_data]
# de_num_data = [[ch2id[ch] for ch in line][1:] for line in ch_data]

en_num_data = [[en2id[en] for en in line.split(' ') ] for line in en_data]
ch_num_data = [[ch2id[ch] for ch in line.split(' ')] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line.split(' ')][1:] for line in ch_data]


print('char:', en_data[1])
print('index:', en_num_data[1])



#one hot 数据格式改为onehot的格式

# 获取输入输出端的最大长度
max_encoder_seq_length = max([len(txt) for txt in en_num_data])
max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
print('max encoder length:', max_encoder_seq_length)
print('max decoder length:', max_decoder_seq_length)

# exit()

with open("config.file", "wb") as f:
    pickle.dump({'max_encoder_seq_length':max_encoder_seq_length,'max_decoder_seq_length':max_decoder_seq_length}, f)

corpus =None

def entext2token(enchar):
    enchar = enchar.split(' ')
    # print(enchar)
    tokenarray = [[en2id[en] for en in enchar]]
    encoder_input_data = np.zeros((1, max_encoder_seq_length, len(en2id)), dtype='float32')
    # for i in range(len(tokenarray)):
    for t, j in enumerate(tokenarray):
        encoder_input_data[0, t, j] = 1.
    return encoder_input_data

# def token2onehot(encoder_input_data):
#     pass

def cntext2token(cnchar):
    cnchar = cnchar.split(' ')
    # print(enchar)
    tokenarray = [[ch2id[zh] for zh in cnchar]]
    decoder_input_data = np.zeros((1, max_decoder_seq_length, len(ch2id)), dtype='float32')
    for i in range(len(tokenarray)):
        for t, j in enumerate(tokenarray[i]):
            decoder_input_data[i, t, j] = 1.
    return decoder_input_data

def generate_arrays_from_file():
    # while 1:
    encoderarray=[]
    for line in en_data:
        # print()
        # entext2token(line)
        testdata=entext2token(line)
        # yield (x, y)
        # yield testdata
        encoderarray.append(testdata)

    decoderarray = []
    for line in ch_data:
        # print()
        # entext2token(line)
        testdata=cntext2token(line)
        # yield (x, y)
        # yield testdata
        decoderarray.append(testdata)

    targetarray = []
    for line in ch_data:
        # print()
        # entext2token(line)
        testdata=cntext2token(line)
        # yield (x, y)
        # yield testdata
        targetarray.append(testdata)
    for a,b,c in zip(encoderarray,decoderarray,targetarray):
        # print(a,b,c)
        yield ([a,b],c)

# for x in generate_arrays_from_file():
    
#     print(x[0].shape)
#     print(x[1].shape)
#     print(x[2].shape)

# encoder_input_data = np.zeros((len(en_num_data), max_encoder_seq_length, len(en2id)), dtype='float32')
# decoder_input_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
# decoder_target_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
# print(encoder_input_data.shape)
# for i in range(len(ch_num_data)):
#     for t, j in enumerate(en_num_data[i]):
#         encoder_input_data[i, t, j] = 1.
#     for t, j in enumerate(ch_num_data[i]):
#         decoder_input_data[i, t, j] = 1.
#     for t, j in enumerate(de_num_data[i]):
#         decoder_target_data[i, t, j] = 1.

# print('index data:\n', en_num_data[1])
# print('one hot data:\n', encoder_input_data[1])
# print(encoder_input_data.shape)
# print(decoder_input_data.shape)





# exit()

#
# def encoder_input_data():
#     encoder_input_data =[]
#     for text in en_data:
#         enchar = text.split(' ')
#         tokenarray = [[en2id[en] for en in enchar]]
#         vec = np.zeros((max_encoder_seq_length, len(en2id)), dtype='float32')
#         for t, j in enumerate(tokenarray):
#             vec[ t, j] = 1.
#         # yield encoder_input_data.append(vec)
#         yield vec
#
# def decoder_input_data():
#     encoder_input_data =[]
#     for text in ch_data:
#         enchar = text.split(' ')
#         tokenarray = [[zh2id[en] for en in enchar]]
#         vec = np.zeros((max_decoder_seq_length, len(zh2id)), dtype='float32')
#         for t, j in enumerate(tokenarray):
#             vec[ t, j] = 1.
#         # yield encoder_input_data.append(vec)
#         yield vec
# def decoder_target_data():
#     encoder_input_data =[]
#     for text in ch_data:
#         enchar = text.split(' ')
#         tokenarray = [[zh2id[en] for en in enchar[1:]]]
#         vec = np.zeros((max_decoder_seq_length, len(zh2id)), dtype='float32')
#         for t, j in enumerate(tokenarray):
#             vec[ t, j] = 1.
#         # yield encoder_input_data.append(vec)
#         yield vec

# exit()





######################## model ##############################

EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)
HIDDEN_SIZE = 256

LEARNING_RATE = 0.002
BATCH_SIZE = 5
EPOCHS = 30


encoder_inputs = keras.Input(shape=(None, EN_VOCAB_SIZE),name='encoder_input')
#emb_inp = Embedding(output_dim=HIDDEN_SIZE, input_dim=EN_VOCAB_SIZE)(encoder_inputs)
encoder_h1, encoder_state_h1, encoder_state_c1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(encoder_inputs)
encoder_h2, encoder_state_h2, encoder_state_c2 = keras.layers.LSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)

# encode_model = keras.Model(encoder_inputs, encoder_h2)
# encode_model.save('encode_model.h5')


decoder_inputs = keras.Input(shape=(None, CH_VOCAB_SIZE),name='decoder_input')
#emb_target = Embedding(output_dim=HIDDEN_SIZE, input_dim=CH_VOCAB_SIZE, mask_zero=True)(decoder_inputs)
lstm1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
lstm2 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
decoder_dense = keras.layers.Dense(CH_VOCAB_SIZE, activation='softmax')

decoder_h1, _, _ = lstm1(decoder_inputs, initial_state=[encoder_state_h1, encoder_state_c1])
decoder_h2, _, _ = lstm2(decoder_h1, initial_state=[encoder_state_h2, encoder_state_c2])
decoder_outputs = decoder_dense(decoder_h2)
#
# decode_model = keras.Model(decoder_inputs, decoder_outputs)
# decode_model.save('decode_model.h5')

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
opt = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCHS,
#           validation_split=0.05)
model.fit_generator(generate_arrays_from_file(),steps_per_epoch=1,epochs=EPOCHS)

model.save('ss2ss_test.h5')



print(ch2id['\n'])
print(ch2id['\t'])


exit()

################################################### predict ############################################################
#预测模型中的encoder和训练中的一样，都是输入序列，输出几个状态。而decoder和训练中稍有不同，因为训练过程中的decoder端的输入是可以确定的，因此状态只需要初始化一次，而预测过程中，需要多次初始化状态，因此将状态也作为模型输入。

# encoder模型和训练相同
encoder_model = keras.Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])

# 预测模型中的decoder的初始化状态需要传入新的状态
decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,))
decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,))
decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,))
decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,))

# 使用传入的值来初始化当前模型的输入状态
decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
decoder_outputs = decoder_dense(decoder_h2)

decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
                      [decoder_outputs, state_h1, state_c1, state_h2, state_c2])


task =[
    # '张柏芝 张雨绮 吴卓林 谢霆锋',
    # '农村 小伙 城里 偷笑 房子',
    # '荷花 西湖 金粟词话 采莲女 林逋 荷叶',
    # '荷花 西湖',
    # '杨迪 脱口秀',
]


for enchar in task:
    # test_data = encoder_input_data[k:k + 1]
    test_data = entext2token(enchar)
    print(test_data.shape)
    h1, c1, h2, c2 = encoder_model.predict(test_data)
    target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
    target_seq[0, 0, ch2id['\t']] = 1
    outputs = []
    while True:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        outputs.append(sampled_token_index)
        target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
        target_seq[0, 0, sampled_token_index] = 1
        if sampled_token_index == ch2id['\n'] or len(outputs) > 20: break

    # print(en_data[k])
    print(enchar)
    print(''.join([id2ch[i] for i in outputs]))
