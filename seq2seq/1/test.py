# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import pickle
#
# sourcearray = []
# with open('../../datasets/letters_source.txt','r',encoding='utf-8') as f:
#     for line in f.readlines():
#         linearray = list(line.replace('\n',''))
#         sourcearray.append(linearray)
#
# targetarray = []
# with open('../../datasets/letters_target.txt','r',encoding='utf-8') as f:
#     for line in f.readlines():
#         linearray = list(line.replace('\n',''))
#         targetarray.append(linearray)
#
# corpus = []
# for (s,t) in zip(sourcearray,targetarray):
#     corpus.append((s,t))
#
#
# print(corpus[0:5])
# exit()



# raw_data=[

# ('你是傻子','你是瓜娃儿'),
# ('你怎么了',"你咋个了"),
# ("你怎么那么傻啊","你咋个楞个傻哦"),
# ('怎么弄嘛',"咋个弄嘛"),
# ("傻子都知道","瓜娃儿都晓得"),
# ("你知道吗","你晓得不"),
# ("怎么了","咋个了"),
# ("你知道啊","你晓得哦"),
# ("傻子啊","瓜娃儿哦"),
# ("你知道吗","你晓得不"),
# ("操你妈","日你先人板板"),
# ("粘住了","巴到了"),
# ("很重","邦重"),
# ("倒霉","背时"),
# ("不会","不得"),
# ("不要扭","不要动"),
# ("刚才","才将"),
# ("抡你两耳光","产你两耳屎"),
# ("撒谎","扯谎"),
# ("打鼾","扯蒲憨"),
# ("吃饭","吃茫茫"),
# ("吃霸王餐","吃跑堂"),
# ("讨厌的人","烂屁娃儿"),
# ("吹牛聊天","吹牛扯把子"),
# ("捉住","逮到"),
# ("白搭","等于零"),
# ("截住","短倒"),
# ("正确","对头"),
# ("下次","二回"),
# ("以后","二天"),
# ("很烦","烦求得很"),
# ("坐车","赶车"),
# ("上面","高头"),
# ("强行","鼓到"),
# ("胡说","鬼扯"),
# ("好吓人啊","好黑人哦"),
# ("骗人","豁人"),
# ("假如","假比"),
# ("跨过来","卡过来"),
# ("没有","莫得"),
# ("抓子嘛","干什么"),
# ("到达","拢了"),
# ("爽","安逸"),
# ("讨饭的","舔盘子的"),
# ("转弯","倒拐"),
# ("好玩","好耍"),
# ("挖苦","弯酸"),
# ("不表态","稳起"),
# ("糟糕","喔霍"),
# ("我试一下","我搞一哈"),
# ("仔细","下细"),
# ("乡下","乡坝头"),
# ("危险","悬火"),
# ("一回","一盘"),
# ("暗地里","阴到"),
# ("真的是","硬是"),
# ("钻过来","拱过来"),
# ("看到","雀到"),
# ("没事","莫得事"),
# ("是不是真的","是不是哦"),

# ]


# data = []
# for putonghua,sichuanhua in raw_data:
#     # (' '.join(list(putonghua)),' '.join(list(sichuanhua)))
#     data.append(' '.join(list(putonghua))+'\t'+' '.join(list(sichuanhua)))


with open('../../datasets/cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    data = data.split('\n')










print(data[-500:])
print(len(data))
data = data[0:5000]
# exit()


# stag = 'start'
# etag = 'end'

stag = '\t'
etag = '\n'





# en_data = [' '.join(list(line.split('\t')[0])) for line in data]
# ch_data = [stag +' ' + ' '.join(list(line.split('\t')[1])) +' '+ etag for line in data]

# en_data = [line.split('\t')[0] for line in data]
# ch_data = [stag +' ' + line.split('\t')[1] +' '+ etag for line in data]


en_data = [line.split('\t')[0] for line in data]
ch_data = [stag +line.split('\t')[1] +etag for line in data]



print(en_data)
print(ch_data)

# exit()






print('英文数据:\n', en_data[:10])
print('\n中文数据:\n', ch_data[:10])

# 分别生成中英文字典
en_vocab = set(''.join(en_data))
# en_vocab =set(' '.join(en_data).split(' '))

id2en = list(en_vocab)
en2id = {c:i for i,c in enumerate(id2en)}

ch_vocab = set(''.join(ch_data))
# ch_vocab =set(' '.join(ch_data).split(' '))
id2ch = list(ch_vocab)
ch2id = {c:i for i,c in enumerate(id2ch)}

print('\n英文字典:\n', en2id)
print('\n中文字典共计\n:', ch2id)

# exit()

with open("en.file", "wb") as f:
    pickle.dump({'id2en':id2en,'en2id':en2id}, f)
with open("zh.file", "wb") as f:
    pickle.dump({'id2ch':id2ch,'ch2id':ch2id}, f)

# 建立字典,将文本数据映射为数字数据形式。
en_num_data = [[en2id[en] for en in line ] for line in en_data]
ch_num_data = [[ch2id[ch] for ch in line] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line][1:] for line in ch_data]


print('char:', en_data[1])
print('index:', en_num_data[1])



#one hot 数据格式改为onehot的格式

# 获取输入输出端的最大长度
max_encoder_seq_length = max([len(txt) for txt in en_num_data])
max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
print('max encoder length:', max_encoder_seq_length)
print('max decoder length:', max_decoder_seq_length)

with open("config.file", "wb") as f:
    pickle.dump({'max_encoder_seq_length':max_encoder_seq_length,'max_decoder_seq_length':max_decoder_seq_length}, f)

encoder_input_data = np.zeros((len(en_num_data), max_encoder_seq_length, len(en2id)), dtype='float32')
decoder_input_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
decoder_target_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')

for i in range(len(ch_num_data)):
    for t, j in enumerate(en_num_data[i]):
        encoder_input_data[i, t, j] = 1.
    for t, j in enumerate(ch_num_data[i]):
        decoder_input_data[i, t, j] = 1.
    for t, j in enumerate(de_num_data[i]):
        decoder_target_data[i, t, j] = 1.

print('index data:\n', en_num_data[1])
print('one hot data:\n', encoder_input_data[1])
print(encoder_input_data.shape)
print(decoder_input_data.shape)


print(ch2id)
print(ch2id[etag])
print(ch2id[stag])

# exit()
######################## model ##############################

EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)
HIDDEN_SIZE = 256

LEARNING_RATE = 0.002
BATCH_SIZE = 50
EPOCHS = 300


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
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)
model.save('ss2ss_test.h5')



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



def entext2token(enchar):
    tokenarray = [[en2id[en] for en in enchar]]
    encoder_input_data = np.zeros((1, 16, len(en2id)), dtype='float32')
    for i in range(len(tokenarray)):
        for t, j in enumerate(tokenarray[i]):
            encoder_input_data[i, t, j] = 1.
    return encoder_input_data

task =[
    'i love you',
    'i hate you',
    'i miss you',
    'i run ',
    'i kill you',
    'i see you',
    # '你是傻子'
    # '你怎么知道',
    # '你怎么傻',

]


for enchar in task:
    # test_data = encoder_input_data[k:k + 1]
    test_data = entext2token(enchar)
    print(test_data.shape)
    h1, c1, h2, c2 = encoder_model.predict(test_data)
    target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
    target_seq[0, 0, ch2id[stag]] = 1
    outputs = []
    while True:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        outputs.append(sampled_token_index)
        target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
        target_seq[0, 0, sampled_token_index] = 1
        if sampled_token_index == ch2id[etag] or len(outputs) > 20: break

    # print(en_data[k])
    print(enchar)
    print(''.join([id2ch[i] for i in outputs]))
