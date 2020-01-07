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


kk =5000
corpus = []
squence_length = 100
BATCH_SIZE = 10
maxFeature=10000

lr = 0.01

EPOCHS = 10

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
            corpus.append((' '.join(list(content)),' '.join(list(title))))
        except Exception as e:
            # raise e
            print(str(e))
# exit()
# print(corpus)
# exit()
# corpus = corpus[:]


tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘',split=" ",num_words=maxFeature)  #创建一个Tokenizer对象


corpusX=[]
corpusYin=[]
corpusYout=[]
for content,title in corpus:
    corpusX.append(content)
    corpusYin.append('\t ' + title + ' \n')
    corpusYout.append(title)


tokenizer.fit_on_texts(corpusX+corpusYin+corpusYout)
word2id=tokenizer.word_index #得到每个词的编号
id2word=tokenizer.index_word #得到每个编号对应的词
# print(vocab)
print(word2id.get('\t'))
print(word2id.get('\n'))
# print(len(word2id))
VOCAB_SIZE = len(word2id)
# exit()

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def text2seq(tokenizer,encoder_text, decoder_text):
    # tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
    return encoder_sequences, decoder_sequences

def padding(encoder_sequences, decoder_sequences, MAX_LEN):
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    return encoder_input_data, decoder_input_data

def decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
    decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")
    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq-1] = 1.
    # print(decoder_output_data.shape)
    return decoder_output_data

# encoder_sequences, decoder_sequences = text2seq(tokenizer,corpusX, corpusYin,squence_length)

# # print(encoder_sequences)
# # print(decoder_sequences)

# encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, squence_length)

# print(encoder_input_data.shape)
# print(decoder_input_data.shape)
# num_samples = len(encoder_sequences)
# decoder_output_data = decoder_output_creater(decoder_input_data, num_samples, squence_length, len(word2id))

# print(decoder_output_data.shape)
# exit()


# x_train_word_ids=tokenizer.texts_to_sequences(corpusX)
# y_train_word_ids_=tokenizer.texts_to_sequences(corpusY)
# y_train_word_ids = [idlist[1:] for idlist in y_train_word_ids_]



# print(x_train_word_ids)
# x_train_word_ids = np.array(x_train_word_ids)

# max_encoder_seq_length = max([len(wdlist) for wdlist in x_train_word_ids])
# max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
# print(max_encoder_seq_length)

# trainX = pad_sequences(x_train_word_ids,maxlen=500, dtype='int')
# trainY_ = pad_sequences(y_train_word_ids_,maxlen=30, dtype='int')
# trainY = pad_sequences(y_train_word_ids,maxlen=30, dtype='int')

# trainX = trainX.reshape(*trainX.shape, 1)
# trainY_ = trainY_.reshape(*trainY_.shape, 1)
# trainY = trainY.reshape(*trainY.shape, 1)

# print(trainX)
# print(trainY.shape)

# exit()
# np.zeros((1500,100,20000),dtype="float32")
# np.zeros((1500,100,20000),dtype="float32")
def datagenerator():
    # encoder_sequences, decoder_sequences = text2seq(tokenizer,corpusX, corpusYin,squence_length)

    # encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, squence_length)

    # num_samples = len(encoder_sequences)
    # decoder_output_data = decoder_output_creater(decoder_input_data, num_samples, squence_length, len(word2id))

    # print(encoder_input_data.shape)
    # print(decoder_input_data.shape)
    # print(decoder_output_data.shape)
    # for a,b,c in zip(encoder_input_data,decoder_input_data,decoder_output_data):
    #     print(a.shape,b.shape,c.shape)
    #     yield ({"encodinput":a,"decodinput":b},{"output":c})

    batches = (len(corpusX)+BATCH_SIZE-1)//BATCH_SIZE
    # print(batches)
    while True:
        for i in range(batches):
            pass
            encoder_sequences, decoder_sequences = text2seq(tokenizer,corpusX[BATCH_SIZE*i:BATCH_SIZE*(i+1)], corpusYin[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
            encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, squence_length)
            num_samples = len(encoder_sequences)
            decoder_output_data = decoder_output_creater(decoder_input_data, num_samples, squence_length, len(word2id))
            # print(encoder_input_data.shape)
            # print(decoder_input_data.shape)
            # print(decoder_output_data.shape)

            yield ({"encodinput":encoder_input_data,"decodinput":decoder_input_data},{"output":decoder_output_data})




######################## model ##############################

def bidirectional_lstm():
    """
    Encoder-Decoder-seq2seq
    """

    # MAX_ART_LEN =
    # MAX_SUM_LEN =
    EMBEDDING_DIM = 60
    HIDDEN_UNITS =256

    # LEARNING_RATE = 0.002
    # BATCH_SIZE = 32
    # EPOCHS = 5

    # input_shape=(input_length, input_dim)


    # en_shape = np.shape(encoder_sequences[0])
    # den_shape = np.shape(decoder_sequences[0])

    # print(en_shape)
    # print(den_shape)

    # encoder
    # encoder_inputs = Input(shape=(squence_length,),name="encodinput")
    encoder_inputs = Input(shape=(None,),name="encodinput")
    # input_length =squence_length
    emb_inp = Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word2id))(encoder_inputs)

    # dropout_U = 0.2, dropout_W = 0.2 ,
    encoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True)
    rev_encoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True, go_backwards=True)
    #
    encoder_outputs, state_h, state_c = encoder_LSTM(emb_inp)
    rev_encoder_outputs, rev_state_h, rev_state_c = rev_encoder_LSTM(emb_inp)
    #
    final_state_h = Add()([state_h, rev_state_h])
    final_state_c = Add()([state_c, rev_state_c])

    encoder_states = [final_state_h, final_state_c]

    # decoder
    # decoder_inputs = Input(shape=(squence_length,),name="decodinput")
    decoder_inputs = Input(shape=(None,),name="decodinput")
    # input_length =squence_length,
    emb_target = Embedding(output_dim=EMBEDDING_DIM, input_dim=len(word2id), mask_zero=True)(decoder_inputs)

    decoder_LSTM = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _, = decoder_LSTM(emb_target, initial_state=encoder_states)
    decoder_dense = Dense(units=len(word2id), activation="linear",name="output")
    decoder_outputs = decoder_dense(decoder_outputs)

    # modeling
    model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
    # model.compile(optimizer=rmsprop, loss="mse", metrics=["accuracy"])

    # return model
    # print(model.summary())

    # x_train, x_test, y_train, y_test = train_test_split(data["article"], data["summaries"], test_size=0.2)
    # model.fit([x_train, y_train], y_train, batch_size=BATCH_SIZE,
    #           epochs=EPOCHS, verbose=1, validation_data=([x_test, y_test], y_test))

    """
    infer / predict
    """
    # encoder_model_inf = Model(encoder_inputs, encoder_states)

    # decoder_state_input_H = Input(shape=(HIDDEN_UNITS,))
    # decoder_state_input_C = Input(shape=(HIDDEN_UNITS,))

    # decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    # decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs, initial_state=decoder_state_inputs)

    # decoder_states = [decoder_state_h, decoder_state_c]
    # decoder_outputs = decoder_dense(decoder_outputs)

    # decoder_model_inf = Model([decoder_inputs]+decoder_state_inputs,
    #                      [decoder_outputs]+decoder_states)

    # scores = model.evaluate([x_test, y_test], y_test, verbose=0)

    # print('LSTM test scores:', scores)
    # print('\007')

    # return model, encoder_model_inf, decoder_model_inf
    return model



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
# model = seq2seq_model(squence_length,100,len(word2id))


# model,enco_model,deco_model = bidirectional_lstm()
model = bidirectional_lstm()



# opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
rmsprop = keras.optimizers.RMSprop(lr=lr, clipnorm=1.0)
model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
model.summary()

# model.fit([encoder_input_data,decoder_input_data], decoder_output_data,
#           # batch_size=1,
#           epochs=EPOCHS,
#           # validation_split=0.05
#           )
batches = (len(corpusX)+BATCH_SIZE-1)//BATCH_SIZE
print('batches:{}'.format(batches))
model.fit_generator(datagenerator(),steps_per_epoch=batches,epochs=EPOCHS)
model.save('s2s.h5')



#predict


# model=None
# corpus =None
# s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
# s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'





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




# for enchar in task:
#     # test_data = encoder_input_data[k:k + 1]
#     test_data = entext2token(enchar)
#     print(test_data.shape)
#     h1, c1, h2, c2 = encoder_model.predict(test_data)
#     target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
#     target_seq[0, 0, ch2id['\t']] = 1
#     outputs = []
#     while True:
#         output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         outputs.append(sampled_token_index)
#         target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
#         target_seq[0, 0, sampled_token_index] = 1
#         if sampled_token_index == ch2id['\n'] or len(outputs) > 20: break

#     # print(en_data[k])
#     print(enchar)
#     print(''.join([id2ch[i] for i in outputs]))
