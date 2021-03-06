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

# from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import * 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

#

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# tf.keras.backend.set_floatx('float32')  


n_simples =5000
# n_simples =76000
corpus = []
# squence_length = 16
BATCH_SIZE = 300
# maxFeature=180
maxFeature=10000
HIDDEN_SIZE = 256
lr = 0.002
EMBEDDING_DIM = 60

EPOCHS = 300

stag = 'start'
etag = 'end'
ptag = ' '
btag = ' '


# # 四川话 任务
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

# for putonghua,sichuanhua in raw_data:
#     corpus.append((' '.join(list(putonghua)),' '.join(list(sichuanhua))))




## 标题生成 任务
# with open('./news2016zh_valid.json','r',encoding='utf-8') as f:
#     textlineslinee = f.read()
#     textlineslines = textlineslinee.split('\n')
#     random.shuffle(textlineslines)
#     # print(len(textlineslines))
#     for line in textlineslines[:n_simples]:
#         try:
#             newsjoson=json.loads(line)
#             title = newsjoson['title']
#             content = newsjoson['content']
#             corpus.append((' '.join(list(content)),' '.join(list(title))))
#         except Exception as e:
#             # raise e
#             print(str(e))



# 英语翻译 任务
with open('../../datasets/cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    data = data.split('\n')
for line in data:
    english = line.split('\t')[0].lower()
    chinese = ' '.join(jieba.cut(line.split('\t')[1],cut_all=True))
    corpus.append((english,chinese))

# # 青云 任务
# with open('../../datasets/qingyun.csv', 'r', encoding='utf-8') as f:
#     data = f.read()
#     data = data.split('\n')
# for line in data:
#     try:
#         english = line.split('|')[0]
#         chinese = line.split('|')[1]
#         corpus.append((english,chinese))
#     except Exception as e:
#         # raise e
#         pass





corpus = corpus[:n_simples]
print(corpus[900:2000])


# exit()

tokenizer_source = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘',num_words=maxFeature)  #创建一个Tokenizer对象
tokenizer_target = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘',num_words=maxFeature)  #创建一个Tokenizer对象


corpusX=[]
corpusYin=[]
corpusYout=[]
for content,title in corpus:
    corpusX.append(content)
    corpusYin.append(stag +btag+ title)# +btag+ etag 
    corpusYout.append(title+btag+ etag)



max_encoder_seq_length = max([len(wtext.split(' ')) for wtext in corpusX])
max_decoder_seq_length = max([len(txt.split(' ')) for txt in corpusYin])

print(max_encoder_seq_length)
print(max_decoder_seq_length)

# print(corpusX)
# print(corpusYin)
# print(corpusYout)

# exit()

tokenizer_source.fit_on_texts(corpusX)
tokenizer_target.fit_on_texts(corpusYin+corpusYout)


# word2id_source={ k:tokenizer_source.word_index[k]-1 for k in tokenizer_source.word_index} #得到每个词的编号
# id2word_source={ k-1:tokenizer_source.index_word[k] for k in tokenizer_source.index_word} #得到每个编号对应的词

# word2id_target={ k:tokenizer_target.word_index[k]-1 for k in tokenizer_target.word_index} #得到每个词的编号
# id2word_target={ k-1:tokenizer_target.index_word[k] for k in tokenizer_target.index_word} #得到每个编号对应的词
# print(vocab)

word2id_source=tokenizer_source.word_index #得到每个词的编号
id2word_source=tokenizer_source.index_word #得到每个编号对应的词

word2id_target=tokenizer_target.word_index #得到每个词的编号
id2word_target=tokenizer_target.index_word #得到每个编号对应的词


def doc2v(tokenizer_source,encoder_text,MAX_LEN,VOCAB_SIZE_SOURCE):
    print(encoder_text)
    encoder_sequences = tokenizer_source.texts_to_sequences([encoder_text])
    print(encoder_sequences)
    encoder_input = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32',)# padding='post', truncating='post'
    # encoder_input = np.zeros((1, MAX_LEN, VOCAB_SIZE_SOURCE), dtype="float32")
    # for seqs in encoder_sequences:
    #     for j, seq in enumerate(seqs):
    #         # print(j,seq)
    #         encoder_input[0][j][seq-1] = 1.

    return encoder_input


print(word2id_source)
# print(id2word_source)

print(word2id_target)
# print(id2word_target)

VOCAB_SIZE_SOURCE = len(word2id_source)+1
VOCAB_SIZE_TARGET = len(word2id_target)+1
print(VOCAB_SIZE_SOURCE)
print(VOCAB_SIZE_TARGET)

print(word2id_target.get(stag))
print(word2id_target.get(etag))

# testd = doc2v(tokenizer_source,'hi ',max_encoder_seq_length,VOCAB_SIZE_SOURCE)
# print(testd)
# print(testd.shape)


# exit()

# saving
with open('tokenizer_source.pickle', 'wb') as handle:
    pickle.dump(tokenizer_source, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tokenizer_target.pickle', 'wb') as handle:
    pickle.dump(tokenizer_target, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("config.file", "wb") as f:
    pickle.dump({'max_encoder_seq_length':max_encoder_seq_length,'max_decoder_seq_length':max_decoder_seq_length}, f)


def text2seq(tokenizer_source,tokenizer_target,encoder_text, decoder_in_text, decoder_out_text):
    # print('encode text : {}'.format(encoder_text[0]))
    # print('decode in text : {}'.format(decoder_in_text[0]))
    # print('decode out text : {}'.format(decoder_out_text[0]))

    encoder_sequences = tokenizer_source.texts_to_sequences(encoder_text)
    decoder_in_sequences = tokenizer_target.texts_to_sequences(decoder_in_text)
    decoder_out_sequences = tokenizer_target.texts_to_sequences(decoder_out_text)

    # print('encode text sequence : {}'.format(encoder_sequences[0]))
    # print('decode text sequence : {}'.format(decoder_sequences[0]))

    return encoder_sequences, decoder_in_sequences,decoder_out_sequences

def padding(encoder_sequences, decoder_in_sequences,decoder_out_sequences, max_encoder_seq_length,max_decoder_seq_length):
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_encoder_seq_length, dtype='int32', padding='post', truncating='post')
    decoder_output_data = pad_sequences(decoder_out_sequences, maxlen=max_decoder_seq_length, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(decoder_in_sequences, maxlen=max_decoder_seq_length, dtype='int32', padding='post', truncating='post')

    # print('encode input data : {}'.format(encoder_input_data[0]))
    # print('decode input data : {}'.format(decoder_input_data[0]))

    # return encoder_sequences, decoder_in_sequences,decoder_out_sequences
    return encoder_input_data, decoder_input_data,decoder_output_data


def model_data_flow_creator(encoder_input_data,decoder_input_data,decoder_output_data,num_samples, max_encoder_seq_length,max_decoder_seq_length, VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET):
    pass


    # encoder_input = encoder_input_data
    # decoder_input = decoder_input_data

    encoder_input = np.zeros((num_samples, max_encoder_seq_length, VOCAB_SIZE_SOURCE), dtype="float32")
    decoder_input = np.zeros((num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET), dtype="float32")
    decoder_output = np.zeros((num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET), dtype="float32")
    # decoder_output = np.array([wlist[1:] for wlist in decoder_input_data])

    # for i, seqs in enumerate(decoder_input_data):
    #     for j, seq in enumerate(seqs):
    #         if j > 0:
    #             decoder_output[i][j][seq] = 1.

    # -1  因为 所有的字的索引 统一减了1
    for i in range(num_samples):
        # print(i)
        for t, j in enumerate(encoder_input_data[i]):
            encoder_input[i, t, j-1] = 1.
        for t, j in enumerate(decoder_input_data[i]):
            decoder_input[i, t, j-1] = 1.
        for t, j in enumerate(decoder_output_data[i]):
            # print(t,j)
            # print(decoder_input_data[i][0])
            decoder_output[i, t, j-1] = 1.
    return encoder_input,decoder_input,decoder_output


# exit()
def datagenerator():
    # batches = (len(corpusX)+BATCH_SIZE-1)//BATCH_SIZE
    batches = (len(corpusX))//BATCH_SIZE
    # print(batches)
    while True:
        for i in range(batches):
            pass

            encoder_sequences, decoder_in_sequences,decoder_out_sequences = text2seq(tokenizer_source,tokenizer_target,corpusX[BATCH_SIZE*i:BATCH_SIZE*(i+1)], corpusYin[BATCH_SIZE*i:BATCH_SIZE*(i+1)],corpusYout[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
            encoder_input_data, decoder_output_data,decoder_input_data = padding(encoder_sequences, decoder_in_sequences,decoder_out_sequences, max_encoder_seq_length,max_decoder_seq_length)
            num_samples = len(encoder_sequences)

            # encoder_input,decoder_input,decoder_output = model_data_flow_creator(encoder_sequences,decoder_in_sequences,decoder_out_sequences, num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)

            yield ({"encodinput":encoder_input_data,"decodinput":decoder_input_data},{"output":decoder_output_data})
            # yield ({"encodinput":encoder_input,"decodinput":decoder_input},{"output":decoder_output})




######################## model ##############################

def bidirectional_lstm():
    """
    Encoder-Decoder-seq2seq
    """

    EMBEDDING_DIM = 60
    HIDDEN_UNITS =HIDDEN_SIZE


    # encoder
    encoder_inputs = Input(shape=(max_encoder_seq_length,),name="encodinput")
    # encoder_inputs = Input(shape=(None,VOCAB_SIZE_SOURCE),name="encodinput")
    # input_length =squence_length
    # emb_inp = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_SOURCE)(encoder_inputs)
    emb_inp = Embedding(VOCAB_SIZE_SOURCE,EMBEDDING_DIM)(encoder_inputs)

    # dropout_U = 0.2, dropout_W = 0.2 ,
    encoder_LSTM = LSTM(HIDDEN_UNITS,  return_sequences=True, return_state=True)
    rev_encoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True, go_backwards=True)
    #
    encoder_outputs, state_h, state_c = encoder_LSTM(emb_inp)
    rev_encoder_outputs, rev_state_h, rev_state_c = rev_encoder_LSTM(encoder_outputs)
    #
    final_state_h = Add()([state_h, rev_state_h])
    final_state_c = Add()([state_c, rev_state_c])

    encoder_states = [final_state_h, final_state_c]

    # decoder
    decoder_inputs = Input(shape=(max_decoder_seq_length,),name="decodinput")
    # decoder_inputs = Input(shape=(None,VOCAB_SIZE_TARGET),name="decodinput")
    # input_length =squence_length,
    # emb_target = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_TARGET, mask_zero=True)(decoder_inputs)
    emb_target = Embedding(VOCAB_SIZE_TARGET,EMBEDDING_DIM, mask_zero=True)(decoder_inputs)

    decoder_LSTM = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _, = decoder_LSTM(emb_target, initial_state=encoder_states)
    decoder_dense = Dense(units=VOCAB_SIZE_TARGET, activation="linear",name="output")
    decoder_outputs = decoder_dense(decoder_outputs)

    # modeling
    model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
    # model.compile(optimizer=rmsprop, loss="mse", metrics=["accuracy"])

    # return model
    print(model.summary())

    # x_train, x_test, y_train, y_test = train_test_split(data["article"], data["summaries"], test_size=0.2)
    # model.fit([x_train, y_train], y_train, batch_size=BATCH_SIZE,
    #           epochs=EPOCHS, verbose=1, validation_data=([x_test, y_test], y_test))

    """
    infer / predict
    """
    encoder_model_inf = Model(encoder_inputs, encoder_states)

    decoder_state_input_H = Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_C = Input(shape=(HIDDEN_SIZE,))

    emb_target = Embedding(VOCAB_SIZE_TARGET,EMBEDDING_DIM, mask_zero=True)(decoder_inputs)

    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(emb_target, initial_state=decoder_state_inputs)

    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model_inf = Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)

    # scores = model.evaluate([x_test, y_test], y_test, verbose=0)

    # print('LSTM test scores:', scores)
    # print('\007')






    # # encoder模型和训练相同
    # encoder_model = keras.Model(encoder_inputs, [state_h, state_c, rev_state_h, rev_state_c])

    # # 预测模型中的decoder的初始化状态需要传入新的状态
    # decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,))
    # decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,))
    # decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,))
    # decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,))

    # # 使用传入的值来初始化当前模型的输入状态
    # decoder_h1, state_h1, state_c1 = decoder_LSTM(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
    # # decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
    # decoder_outputs = decoder_dense(decoder_h1)

    # # decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
    # #                       [decoder_outputs, state_h1, state_c1, state_h2, state_c2])

    # decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
    #                       [decoder_outputs, state_h1, state_c1, state_h2, state_c2])






    return model, encoder_model_inf, decoder_model_inf
    # return model,encoder_model,decoder_model

def simpleModel_embedding():

    encoder_inputs = Input(shape=(None,),name="encodinput",dtype=tf.int32)
    # encoder_inputs = keras.Input(shape=(max_encoder_seq_length,),name='encodinput')
    emb_inp = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_SOURCE)(encoder_inputs)
    encoder_h1, encoder_state_h1, encoder_state_c1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(emb_inp)
    encoder_h2, encoder_state_h2, encoder_state_c2 = keras.layers.LSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)

    # encode_model = keras.Model(encoder_inputs, encoder_h2)
    # encode_model.save('encode_model.h5')


    decoder_inputs = Input(shape=(None,),name="decodinput",dtype=tf.int32)
    # decoder_inputs = keras.Input(shape=(max_decoder_seq_length,),name='decodinput')
    emb_target = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_TARGET, mask_zero=True)(decoder_inputs)
    lstm1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    lstm2 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(VOCAB_SIZE_TARGET, activation='softmax',name="output")

    decoder_h1, _, _ = lstm1(emb_target, initial_state=[encoder_state_h1, encoder_state_c1])
    decoder_h2, _, _ = lstm2(decoder_h1, initial_state=[encoder_state_h2, encoder_state_c2])
    decoder_outputs = decoder_dense(decoder_h2)
    #
    # decode_model = keras.Model(decoder_inputs, decoder_outputs)
    # decode_model.save('decode_model.h5')

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)


    encoder_model = keras.Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])

    # 预测模型中的decoder的初始化状态需要传入新的状态
    decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,))

    # 使用传入的值来初始化当前模型的输入状态
    emb_target = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_TARGET, mask_zero=True)(decoder_inputs)
    decoder_h1, state_h1, state_c1 = lstm1(emb_target, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
    decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
    decoder_outputs = decoder_dense(decoder_h2)

    decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
                          [decoder_outputs, state_h1, state_c1, state_h2, state_c2])




    return model,encoder_model,decoder_model



# def seq2seq_model(input_length,output_sequence_length,vocab_size):
#     model = tf.keras.models.Sequential()
#     model.add(Embedding(input_dim=vocab_size,output_dim = 128,input_length=input_length))
#     model.add(Bidirectional(GRU(128, return_sequences = False)))
#     model.add(Dense(128, activation="relu"))
#     model.add(RepeatVector(30))
#     model.add(Bidirectional(GRU(128, return_sequences = True)))
#     model.add(TimeDistributed(Dense(vocab_size, activation = 'softmax')))
#     # model.compile(loss = sparse_categorical_crossentropy, 
#     #               optimizer = Adam(1e-3))
#     # model.summary()
#     return model
# model = seq2seq_model(squence_length,100,VOCAB_SIZE)



# model,enco_model,deco_model = bidirectional_lstm()
# model,encoder_model,decoder_model = bidirectional_lstm()
# model,encoder_model,decoder_model = simpleModel()
model,encoder_model,decoder_model = simpleModel_embedding()




# opt = keras.optimizers.RMSprop(lr=lr, clipnorm=1.0)
opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# print(model.get_layer('embedding').tensor)
# print(model.get_layer('embedding_1'))
# print(model.get_layer('lstm'))
# print(model.get_layer('lstm_1'))
# print(model.get_layer('lstm_2'))
# print(model.get_layer('lstm_3'))
print(model.to_json())
# exit()
##按照原数据序列长度 生成训练数据
#all datasets

encoder_sequences, decoder_in_sequences,decoder_out_sequences = text2seq(tokenizer_source,tokenizer_target,corpusX, corpusYin,corpusYout)
encoder_input, decoder_input, decoder_output = padding(encoder_sequences, decoder_in_sequences,decoder_out_sequences, max_encoder_seq_length,max_decoder_seq_length)

num_samples = len(encoder_sequences)
# encoder_input,decoder_input,decoder_output = model_data_flow_creator(encoder_sequences,decoder_in_sequences,decoder_out_sequences, num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)




model.fit([encoder_input,decoder_input], decoder_output,
          # batch_size=1,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          # validation_split=0.05
          )


# # 训练方式： 分批代入计算
# # generator

# # batches = (len(corpusX)+BATCH_SIZE-1)//BATCH_SIZE
# batches = (len(corpusX))//BATCH_SIZE
# print('len corpusX :{}'.format(len(corpusX)))
# print('batches:{}'.format(batches))
# model.fit_generator(datagenerator(),steps_per_epoch=batches,epochs=EPOCHS)



model.save('s2s-2.h5')



#predict


# model=None
# corpus =None
# s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
# s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'



task =[
    # '你怎么知道',
    # '你怎么傻',
    'i love you',
    'i hate you',
    'i miss her',
    'i run away',
    'he will kill you',
    'he want to see you',

]


def gen_starter(stag):
    # print(stag)
    # print(word2id_target[stag])
    decoder_sequences = tokenizer_target.texts_to_sequences([stag])
    # print(decoder_sequences)
    decoder_input = pad_sequences(decoder_sequences, maxlen=max_decoder_seq_length, dtype='int32',)# padding='post', truncating='post'
    # print(decoder_input)
    return decoder_input

for enchar in task:
    test_data = doc2v(tokenizer_source,enchar,max_encoder_seq_length,VOCAB_SIZE_SOURCE)
    print(test_data)
    # print(test_data.shape)
    h1, c1, h2, c2 = encoder_model.predict(test_data)

    # target_seq = np.zeros((1, VOCAB_SIZE_TARGET))
    # target_seq[0, word2id_target[stag]] = 1

    target_seq=gen_starter(stag)

    outputs = []
    while True:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        outputs.append(sampled_token_index)

        # target_seq = np.zeros(( 1,VOCAB_SIZE_TARGET))
        # target_seq[0, sampled_token_index] = 1.

        target_seq = gen_starter(id2word_target.get(sampled_token_index))

        if sampled_token_index == word2id_target[etag] or len(outputs) > 20: break

    # print(en_data[k])
    print(enchar)
    print(' '.join([id2word_target.get(i,'None') for i in outputs]))
