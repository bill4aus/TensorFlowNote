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


n_simples =500
# n_simples =76000
corpus = []
# squence_length = 16
BATCH_SIZE = 20
# maxFeature=180
maxFeature=10000
HIDDEN_SIZE = 256
lr = 0.002
EMBEDDING_DIM = 20

EPOCHS = 30

stag = 'start'
etag = 'end'
ptag = ' '
btag = ' '



# 英语翻译 任务
with open('../../datasets/tieba.dialogues', 'r', encoding='utf-8') as f:
    data = f.read()
    data = data.split('\n')
for line in data:
    try:
        english = line.split('\t')[0]
        chinese = line.split('\t')[1]
        corpus.append((english,chinese))
    except Exception as e:
        # raise e
        pass
    






corpus = corpus[:n_simples]
print(corpus[1032:1064])
print(len(corpus))







tokenizer_source = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘',num_words=maxFeature)  #创建一个Tokenizer对象
tokenizer_target = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘',num_words=maxFeature)  #创建一个Tokenizer对象


corpusX=[]
corpusYin=[]
corpusYout=[]
for content,title in corpus:
    corpusX.append(list(content))
    corpusYin.append(stag +btag+ ' '.join(list(title)))# +btag+ etag 
    corpusYout.append(' '.join(list(title))+btag+ etag)


max_encoder_seq_length = max([len(wlist) for wlist in corpusX])
max_decoder_seq_length = max([len(txt) for txt in corpusYin])

print(max_encoder_seq_length)
print(max_decoder_seq_length)
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
    encoder_sequences = tokenizer_source.texts_to_sequences([list(encoder_text)])
    # print(encoder_sequences)
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

def decoder_output_creater(decoder_out_data, num_samples, MAX_LEN, VOCAB_SIZE):
    decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")
    for i, seqs in enumerate(decoder_out_data):
        # print(seqs)
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq] = 1.

    # print('decode output data : {}'.format(decoder_output_data[0]))

    return decoder_output_data

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
            # encoder_input_data, decoder_output_data,decoder_input_data = padding(encoder_sequences, decoder_sequences, squence_length)

            num_samples = len(encoder_sequences)
            # decoder_output_data = decoder_output_creater(decoder_out_sequences, num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET)
            encoder_input,decoder_input,decoder_output = model_data_flow_creator(encoder_sequences,decoder_in_sequences,decoder_out_sequences, num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)

            # print(encoder_input.shape)
            # print(decoder_input.shape)
            # print(decoder_output.shape)

            # yield ({"encodinput":encoder_input_data,"decodinput":decoder_input_data},{"output":decoder_output_data})
            yield ({"encodinput":encoder_input,"decodinput":decoder_input},{"output":decoder_output})




######################## model ##############################

def simpleModel_embedding():

    encoder_inputs = Input(shape=(max_encoder_seq_length,VOCAB_SIZE_SOURCE),name="encodinput")
    # encoder_inputs = keras.Input(shape=(max_encoder_seq_length,),name='encodinput')
    # emb_inp = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_SOURCE)(encoder_inputs)
    encoder_h1, encoder_state_h1, encoder_state_c1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(encoder_inputs)
    encoder_h2, encoder_state_h2, encoder_state_c2 = keras.layers.LSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)

    # encode_model = keras.Model(encoder_inputs, encoder_h2)
    # encode_model.save('encode_model.h5')


    decoder_inputs = Input(shape=(max_decoder_seq_length,VOCAB_SIZE_TARGET),name="decodinput")
    # decoder_inputs = keras.Input(shape=(max_decoder_seq_length,),name='decodinput')
    # emb_target = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_TARGET, mask_zero=True)(decoder_inputs)
    lstm1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    lstm2 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(VOCAB_SIZE_TARGET, activation='softmax',name="output")

    decoder_h1, _, _ = lstm1(decoder_inputs, initial_state=[encoder_state_h1, encoder_state_c1])
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
    # emb_target = Embedding(output_dim=EMBEDDING_DIM, input_dim=VOCAB_SIZE_TARGET, mask_zero=True)(decoder_inputs)
    decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
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


##按照原数据序列长度 生成训练数据
#all datasets

encoder_sequences, decoder_in_sequences,decoder_out_sequences = text2seq(tokenizer_source,tokenizer_target,corpusX, corpusYin,corpusYout)
encoder_input, decoder_input, decoder_output = padding(encoder_sequences, decoder_in_sequences,decoder_out_sequences, max_encoder_seq_length,max_decoder_seq_length)
num_samples = len(encoder_sequences)

# decoder_output = decoder_output_creater(decoder_output_data, num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET)

# encoder_input,decoder_input,decoder_output = model_data_flow_creator(encoder_sequences,decoder_in_sequences,decoder_out_sequences, num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)
# print(encoder_input.shape)
# print(decoder_input.shape)
# print(decoder_output.shape)

model.fit([encoder_input,decoder_input], decoder_output,
          # batch_size=1,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          # validation_split=0.05
          )


# 训练方式： 分批代入计算
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
    '又一国军事基地被袭击了，都是替美国出头惹的祸！',
    '关于中央八项规定 总书记是怎样带头执行的？',

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
    # print(test_data)
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
    print(''.join([id2word_target.get(i,'None') for i in outputs]))
