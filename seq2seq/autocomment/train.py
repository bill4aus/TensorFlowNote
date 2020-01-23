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


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


n_simples =70000
# n_simples =76000
corpus = []
# squence_length = 16
BATCH_SIZE = 130
# maxFeature=180
maxFeature=10000

lr = 0.008

EPOCHS = 300

stag = 'start'
etag = 'end'
ptag = ' '
btag = ' '

configpath = 'config'






# # 贴吧 任务
# with open('../../datasets/tieba.dialogues', 'r', encoding='utf-8') as f:
#     data = f.read()
#     data = data.split('\n')
# for line in data:
#     try:
#         english = line.split('\t')[0]
#         chinese = line.split('\t')[1]
#         corpus.append((english,chinese))
#     except Exception as e:
#         # raise e
#         pass
    
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

# 网易 任务
corpusdict = dict()
with open('../../datasets/163/163-newsid-title.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    data = data.split('\n')
for line in data:
    try:
        linejson=json.loads(line)
        newsid = linejson.get('newsid:')
        newstitle = linejson.get('title')
        # print(newsid)
        # print(corpusdict.get(newsid))


        if corpusdict.get(newsid)==None:
            corpusdict[newsid]=dict()
            corpusdict[newsid]['title']=newstitle
            corpusdict[newsid]['comments']=list()
    except Exception as e:
        # raise e
        print('.....................error.....................')
        pass

# print(len(corpusdict))

with open('../../datasets/163/163-newsid-comments.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    data = data.split('\n')
for line in data:
    try:
        linejson=json.loads(line)
        newsid = linejson.get('newsid:')
        usercomment = linejson.get('usercomment')

        # and len(usercomment)<50
        if corpusdict.get(newsid)!=None :
            corpusdict.get(newsid).get('comments').append(usercomment)
    except Exception as e:
        # raise e
        print('.....................error.....................')
        pass


print(len(corpusdict))

for newsid in corpusdict:
    newsbody = corpusdict.get(newsid)
    # print(newsid)
    # print(newsbody)
    if newsbody != None:
        ucomts = newsbody.get('comments')
        if ucomts != None:
            for comt in ucomts:
                corpus.append((newsbody.get('title'),comt))





random.shuffle(corpus)
corpus = corpus[:n_simples]

print(corpus[1032:1064])
print(len(corpus))
# exit()











# filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘'
tokenizer_source = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~（）；’‘',num_words=maxFeature)  #创建一个Tokenizer对象
tokenizer_target = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~（）；’‘',num_words=maxFeature)  #创建一个Tokenizer对象


corpusX=[]
corpusYin=[]
corpusYout=[]
for content,title in corpus:
    corpusX.append(list(content))
    corpusYin.append(stag +btag+ ' '.join(list(title)) )#+btag+ etag 
    corpusYout.append(' '.join(list(title))+btag+ etag)

corpus=None

max_encoder_seq_length = int(np.mean([len(wlist) for wlist in corpusX]))
max_decoder_seq_length = int(np.mean([len(txt) for txt in corpusYin]))

print(max_encoder_seq_length)
print(max_decoder_seq_length)



print(corpusX[1032:1064])
# exit()

tokenizer_source.fit_on_texts(corpusX)
tokenizer_target.fit_on_texts(corpusYin+corpusYout)

word2id_source={ k:tokenizer_source.word_index[k]-1 for k in tokenizer_source.word_index} #得到每个词的编号
id2word_source={ k-1:tokenizer_source.index_word[k] for k in tokenizer_source.index_word} #得到每个编号对应的词

word2id_target={ k:tokenizer_target.word_index[k]-1 for k in tokenizer_target.word_index} #得到每个词的编号
id2word_target={ k-1:tokenizer_target.index_word[k] for k in tokenizer_target.index_word} #得到每个编号对应的词
# print(vocab)


def doc2v(tokenizer_source,encoder_text,MAX_LEN,VOCAB_SIZE_SOURCE):
    encoder_sequences = tokenizer_source.texts_to_sequences([list(encoder_text)])
    print(encoder_sequences)
    # encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32',)# padding='post', truncating='post'
    encoder_input = np.zeros((1, MAX_LEN, VOCAB_SIZE_SOURCE), dtype="float32")

    # for i, seqs in enumerate(encoder_sequences):
    #     for j, seq in enumerate(seqs):
    #         if j > 0:
    #             encoder_input[i][j][seq-1] = 1.

    for seqs in encoder_sequences:
        for j, seq in enumerate(seqs):
            # print(j,seq)
            encoder_input[0][j][seq-1] = 1.

    return encoder_input


# print(word2id_source)
# print(id2word_source)

# print(word2id_target)
# print(id2word_target)

VOCAB_SIZE_SOURCE = len(word2id_source)
VOCAB_SIZE_TARGET = len(word2id_target)
print(VOCAB_SIZE_SOURCE)
print(VOCAB_SIZE_TARGET)

# print(word2id_target.get(stag))
# print(word2id_target.get(etag))

# testd = doc2v(tokenizer_source,'hi ',max_encoder_seq_length,VOCAB_SIZE_SOURCE)
# print(testd)
# print(testd.shape)


# exit()

# saving
with open(configpath+'/tokenizer_source.pickle', 'wb') as handle:
    pickle.dump(tokenizer_source, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(configpath+'/tokenizer_target.pickle', 'wb') as handle:
    pickle.dump(tokenizer_target, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(configpath+"/config.file", "wb") as f:
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


def model_data_flow_creator_padding(encoder_sequences,decoder_input_sequences,decoder_output_sequences,num_samples, max_encoder_seq_length,max_decoder_seq_length, VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET):
    pass

    encoder_input_data, decoder_input_data,decoder_output_data = padding(encoder_sequences, decoder_input_sequences, decoder_output_sequences, max_encoder_seq_length,max_decoder_seq_length)

    # encoder_input = encoder_input_data.Reshape()
    # decoder_input = decoder_input_data
    # decoder_output = decoder_output_data

    encoder_input = np.zeros((num_samples, max_encoder_seq_length, VOCAB_SIZE_SOURCE), dtype="float16") #float32
    decoder_input = np.zeros((num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET), dtype="float16")
    decoder_output = np.zeros((num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET), dtype="float16")
    # decoder_output = np.array([wlist[1:] for wlist in decoder_input_data])


    # -1  因为 所有的字的索引 统一减了1
    for i in range(num_samples):
        # print(i)
        for t, j in enumerate(encoder_input_data[i]):
            encoder_input[i, t, j-1] = 1.
        for t, j in enumerate(decoder_input_data[i]):
            # print(i,t,j-1)
            decoder_input[i, t, j-1] = 1.
        for t, j in enumerate(decoder_output_data[i]):
            # print(i,t,j-1)
            # print(decoder_input_data[i][0])
            decoder_output[i, t, j-1] = 1.

    return encoder_input,decoder_input,decoder_output


def model_data_flow_creator_nopadding(encoder_input_data,decoder_input_data,decoder_output_data,num_samples, max_encoder_seq_length,max_decoder_seq_length, VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET):
    pass


    # encoder_input = encoder_input_data
    # decoder_input = decoder_input_data

    encoder_input = np.zeros((num_samples, max_encoder_seq_length, VOCAB_SIZE_SOURCE), dtype="float32")
    decoder_input = np.zeros((num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET), dtype="float32")
    decoder_output = np.zeros((num_samples, max_decoder_seq_length, VOCAB_SIZE_TARGET), dtype="float32")
    # decoder_output = np.array([wlist[1:] for wlist in decoder_input_data])

    print(decoder_input.shape)

    # -1  因为 所有的字的索引 统一减了1
    for i in range(num_samples):
        # print(i)
        for t, j in enumerate(encoder_input_data[i]):
            encoder_input[i, t, j-1] = 1.
        for t, j in enumerate(decoder_input_data[i]):
            decoder_input[i, t, j-1] = 1.
        for t, j in enumerate(decoder_output_data[i]):
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

            # ##按照原数据序列长度 生成训练数据
            # encoder_input,decoder_input,decoder_output = model_data_flow_creator_nopadding(encoder_sequences,decoder_in_sequences,decoder_out_sequences, num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)

            # #根据 MAX_LEN 对序列长度进行裁剪后  生成训练数据
            encoder_input,decoder_input,decoder_output = model_data_flow_creator_padding(encoder_sequences,decoder_in_sequences,decoder_out_sequences,num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)


            # print(encoder_input.shape)
            # print(decoder_input.shape)
            # print(decoder_output.shape)

            # yield ({"encodinput":encoder_input_data,"decodinput":decoder_input_data},{"output":decoder_output_data})
            yield ({"encodinput":encoder_input,"decodinput":decoder_input},{"output":decoder_output})




######################## model ##############################

def simpleModel():
    HIDDEN_SIZE = 256
    dtypset = tf.float32


    encoder_inputs = keras.Input(shape=(None,VOCAB_SIZE_SOURCE),name='encodinput')
    # encoder_inputs = keras.Input(shape=(max_encoder_seq_length,),name='encodinput')
    #emb_inp = Embedding(output_dim=HIDDEN_SIZE, input_dim=EN_VOCAB_SIZE)(encoder_inputs)
    encoder_h1, encoder_state_h1, encoder_state_c1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True,dtype=dtypset)(encoder_inputs)
    encoder_h2, encoder_state_h2, encoder_state_c2 = keras.layers.LSTM(HIDDEN_SIZE, return_state=True,dtype=dtypset)(encoder_h1)

    # encode_model = keras.Model(encoder_inputs, encoder_h2)
    # encode_model.save('encode_model.h5')

    decoder_inputs = keras.Input(shape=(None,VOCAB_SIZE_TARGET),name='decodinput')
    # decoder_inputs = keras.Input(shape=(max_decoder_seq_length,),name='decodinput')
    #emb_target = Embedding(output_dim=HIDDEN_SIZE, input_dim=CH_VOCAB_SIZE, mask_zero=True)(decoder_inputs)
    lstm1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True,dtype=dtypset)
    lstm2 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True,dtype=dtypset)
    decoder_dense = keras.layers.Dense(VOCAB_SIZE_TARGET, activation='softmax',name="output",dtype=dtypset)

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
# model = bidirectional_lstm()
model,encoder_model,decoder_model = simpleModel()



# opt = keras.optimizers.RMSprop(lr=lr, clipnorm=1.0)
opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()


# # 训练方式： 整体代入计算
# #all datasets

# encoder_sequences, decoder_in_sequences,decoder_out_sequences = text2seq(tokenizer_source,tokenizer_target,corpusX, corpusYin,corpusYout)
# num_samples = len(encoder_sequences)

# ##按照原数据序列长度 生成训练数据
# encoder_input,decoder_input,decoder_output = model_data_flow_creator_nopadding(encoder_sequences,decoder_in_sequences,decoder_out_sequences, num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)

# #根据 MAX_LEN 对序列长度进行裁剪后  生成训练数据
# # encoder_input,decoder_input,decoder_output = model_data_flow_creator_padding(encoder_sequences,decoder_in_sequences,decoder_out_sequences,num_samples, max_encoder_seq_length,max_decoder_seq_length,VOCAB_SIZE_SOURCE,VOCAB_SIZE_TARGET)

# # print(encoder_input.shape)
# # print(decoder_input.shape)
# # print(decoder_output.shape)

# model.fit([encoder_input,decoder_input], decoder_output,
#           # batch_size=1,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCHS,
#           # validation_split=0.05
#           )



# 训练方式： 分批代入计算
# generator

# batches = (len(corpusX)+BATCH_SIZE-1)//BATCH_SIZE
batches = (len(corpusX))//BATCH_SIZE
print('len corpusX :{}'.format(len(corpusX)))
print('batches:{}'.format(batches))

# workers=4
model.fit_generator(datagenerator(),steps_per_epoch=batches,epochs=EPOCHS)



model.save(configpath+'/s2s.h5')



#predict


# model=None
# corpus =None
# s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
# s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'


task =[
    # '又一国军事基地被袭击了，都是替美国出头惹的祸！',
    # '关于中央八项规定 总书记是怎样带头执行的？',

    '王毅谈习近平主席对缅甸进行国事访问',
    '【中国稳健前行】加快推进社会治理共同体建设',
    '人均1万美元 了不起 新春走基层  别样"春运"',
    '白金汉宫宣布：哈里梅根将放弃王室头衔',
    '伊朗称失事客机黑匣子将被送往乌克兰：伊朗无法读取内容',
    '是否应该废除死刑?约8成日本人say no 理由是…',
    '世界首富易主！LV总裁取代亚马逊创始人成新首富',
    '谁来接任国民党主席？党内出现提议郭台铭参选呼声',
    '惨烈！亚洲首富家族内斗，有钱人狠起来真可',
    '春节前肉菜供应量增加，肉价下降2块多',
    '检察官建议免除交易员Navinder的牢狱之灾',
    '蛋壳公寓赴美上市，看头部企业如何突围？',
    '央行年内“补水”已超万亿 下周LPR大概率下调',
    '拓新型阅读空间 广东首间“粤书吧”办新春国乐沙龙',
    '21分钟砍21+6！又一广东旧将在CBA挑大梁 配..',
    '“儿子你擦干眼泪去相亲吧！”33岁男子凌晨收',
    '“烂尾楼”变新居 600余户居民搬进新家过新年',
    '凉凉，五家量子波动速读机构被查处',
    '蔡英文还不知道，台湾已陷入四大危机',
    '省工商联组织召开浙江省民营企业家“强信心、增动能”..',
    '浙江卫视给高以翔赔偿金已谈妥，3月份在金宝山',
    '赵忠祥最后一次录制节目的视频曝光：需要靠两人搀扶才..',
    '程潇穿紫色毛绒外套现身机场 踩长靴秀纤细美腿',




    # '看到环球，内容都没看，直接奔着评论来了',
    # '不是暗杀，是就地正法。中立的说法叫击毙。',
    # '国外是真难呀，不论做什么都有一帮人吹毛求疵！',
    # '吓的把客机当无人机直接击落。',
    # '我特想听听专家的看法。',
    # '她们还是孩子！',
    # '如果让我碰到这些兔崽子，我保证不打死他们',
    # '你现在多大了，她多大了？',
    # '这些个坏孩子',
    # '霸凌的怎么都是女生',
    # '霸凌现象难除，是什么原因？',
    # '全部抓来剃头',
    # '社会的悲衰',
    # '心理教育 呵呵',
    # '麻痹的',
    # '你就是专家了。',
    # '用浓硫酸一个个的泼在它们脸上。',

]


for enchar in task:
    test_data = doc2v(tokenizer_source,enchar,max_encoder_seq_length,VOCAB_SIZE_SOURCE)
    # print(test_data.shape)
    h1, c1, h2, c2 = encoder_model.predict(test_data)
    target_seq = np.zeros((1, 1, VOCAB_SIZE_TARGET))
    target_seq[0, 0, word2id_target[stag]] = 1
    outputs = []
    while True:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        outputs.append(sampled_token_index)
        target_seq = np.zeros((1, 1,VOCAB_SIZE_TARGET))
        target_seq[0, 0, sampled_token_index] = 1.
        if sampled_token_index == word2id_target[etag] or len(outputs) > 20: break

    # print(en_data[k])
    print(enchar)
    print(''.join([id2word_target.get(i,'None') for i in outputs]))
