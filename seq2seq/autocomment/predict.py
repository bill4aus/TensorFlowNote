# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import pickle
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import jieba
import jieba.posseg as pseg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# # exit()
#
# en_data = [line.split('\t')[0] for line in data]
# ch_data = ['\t' + line.split('\t')[1] + '\n' for line in data]
#
# print('英文数据:\n', en_data[:10])
# print('\n中文数据:\n', ch_data[:10])

'''
en_file ={}
with open("en.file", "rb") as f:
    en_file = pickle.load(f)

zh_file ={}
with open("zh.file", "rb") as f:
    zh_file = pickle.load(f)
m_config = {}
with open("config.file", "rb") as f:
    m_config = pickle.load(f)
# print(m_config)
# 分别生成中英文字典
# en_vocab = en_file['id2en']
id2en = en_file['id2en']
en2id = en_file['en2id'] #{c:i for i,c in enumerate(id2en)}

# print(id2en)
# exit()
# ch_vocab = zh_file['vocab']
id2ch = zh_file['id2ch']
ch2id = zh_file['ch2id'] #{c:i for i,c in enumerate(id2ch)}

print('\n英文字典:\n', en2id)
print('\n中文字典共计\n:', ch2id)




'''
configpath = 'config'


# loading
with open(configpath+'/tokenizer_source.pickle', 'rb') as handle:
    tokenizer_source = pickle.load(handle)
with open(configpath+'/tokenizer_target.pickle', 'rb') as handle:
    tokenizer_target = pickle.load(handle)
with open(configpath+"/config.file", "rb") as handle:
    config = pickle.load(handle)


# word2id=tokenizer.word_index #得到每个词的编号
# id2word=tokenizer.index_word #得到每个编号对应的词


word2id_source={ k:tokenizer_source.word_index[k]-1 for k in tokenizer_source.word_index} #得到每个词的编号
id2word_source={ k-1:tokenizer_source.index_word[k] for k in tokenizer_source.index_word} #得到每个编号对应的词

word2id_target={ k:tokenizer_target.word_index[k]-1 for k in tokenizer_target.word_index} #得到每个词的编号
id2word_target={ k-1:tokenizer_target.index_word[k] for k in tokenizer_target.index_word} #得到每个编号对应的词


# print(id2word)

######################## model ##############################

VOCAB_SIZE_SOURCE = len(word2id_source)
VOCAB_SIZE_TARGET = len(word2id_target)

print(VOCAB_SIZE_SOURCE)
print(VOCAB_SIZE_TARGET)

HIDDEN_SIZE = 256

LEARNING_RATE = 0.003
BATCH_SIZE = 100
EPOCHS = 200


# max_encoder_seq_length = max([len(wlist) for wlist in corpusX])
# max_decoder_seq_length = max([len(txt) for txt in corpusYin])

max_encoder_seq_length= config['max_encoder_seq_length'] 
max_decoder_seq_length = config['max_decoder_seq_length']

print(max_encoder_seq_length)
print(max_decoder_seq_length)

stag = 'start'
etag = 'end'
ptag = ' '
btag = ' '

#
#
# encoder_inputs = keras.Input(shape=(None, EN_VOCAB_SIZE))
# #emb_inp = Embedding(output_dim=HIDDEN_SIZE, input_dim=EN_VOCAB_SIZE)(encoder_inputs)
# encoder_h1, encoder_state_h1, encoder_state_c1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(encoder_inputs)
# encoder_h2, encoder_state_h2, encoder_state_c2 = keras.layers.LSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)
#
#
# decoder_inputs = keras.Input(shape=(None, CH_VOCAB_SIZE))
# #emb_target = Embedding(output_dim=HIDDEN_SIZE, input_dim=CH_VOCAB_SIZE, mask_zero=True)(decoder_inputs)
# lstm1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
# lstm2 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
# decoder_dense = keras.layers.Dense(CH_VOCAB_SIZE, activation='softmax')
#
# decoder_h1, _, _ = lstm1(decoder_inputs, initial_state=[encoder_state_h1, encoder_state_c1])
# decoder_h2, _, _ = lstm2(decoder_h1, initial_state=[encoder_state_h2, encoder_state_c2])
# decoder_outputs = decoder_dense(decoder_h2)
#


def build_model():
    model = keras.models.load_model('s2s.h5')

    print(model.summary())

    # for layer in model.layers:
    #     print(layer._name)
    #     print(layer)
    #     print('-')

    # print(model.layers[0].output)
    # print(model.layers[1].output)
    # print(model.layers[2].output)
    # print(model.layers[3].output)

    # exit()

    encoder_inputs = model.input[0]   # input_1

    # encoder_h1, encoder_state_h1, encoder_state_c1 = keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(encoder_inputs)
    # encoder_h2, encoder_state_h2, encoder_state_c2 = keras.layers.LSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)

    # encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    # encoder_states = [state_h_enc, state_c_enc]

    encoder_h1, encoder_state_h1, encoder_state_c1 = model.get_layer('lstm').output #model.layers[2].output   # lstm
    encoder_h2, encoder_state_h2, encoder_state_c2 = model.get_layer('lstm_1').output #model.layers[3].output   # lstm_1




    # encoder_model = keras.Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])
    # encoder_model = keras.Model(encoder_inputs, encoder_states)
    # encoder_model = keras.Model(encoder_inputs, [])
    encoder_model = keras.Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])


    #
    #
    # decoder_inputs = model.input[1]   # input_2
    # decoder_state_input_h = keras.Input(shape=(HIDDEN_SIZE,), name='input_3')
    # decoder_state_input_c = keras.Input(shape=(HIDDEN_SIZE,), name='input_4')
    # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # decoder_lstm = model.layers[3]
    # decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    #     decoder_inputs, initial_state=decoder_states_inputs)
    # decoder_states = [state_h_dec, state_c_dec]
    # decoder_dense = model.layers[4]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = keras.Model(
    #     [decoder_inputs] + decoder_states_inputs,
    #     [decoder_outputs] + decoder_states)

    decoder_inputs = model.input[1]   # input_2
    # print(decoder_inputs)

    decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_1')
    decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_2')
    # decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_3')
    # decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_4')



    # 使用传入的值来初始化当前模型的输入状态
    lstm1 = model.get_layer('lstm_2') #model.layers[4]
    # lstm2 = model.get_layer('lstm_3') #model.layers[5]
    embedding1 = model.get_layer('embedding_1') #model.layers[4]

    decoder_dense = model.layers[9]
    decoder_h1, state_h1, state_c1 = lstm1(embedding1(decoder_inputs), initial_state=[decoder_state_input_h1, decoder_state_input_c1])
    # decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
    decoder_outputs = decoder_dense(decoder_h1)

    decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1],
                          [decoder_outputs, state_h1, state_c1])

    # decoder_model = keras.Model([decoder_inputs, encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2],
    #                       [decoder_outputs, state_h1, state_c1, state_h2, state_c2])

    return encoder_model,decoder_model



def build_model_1(pathstr):
    pass
    model = keras.models.load_model(pathstr+'/s2s.h5')
    print(model.summary())



    encoder_inputs = model.input[0]   # input_1

    encoder_h1, encoder_state_h1, encoder_state_c1 = model.get_layer('lstm').output #model.layers[2].output   # lstm
    encoder_h2, encoder_state_h2, encoder_state_c2 = model.get_layer('lstm_1').output #model.layers[3].output   # lstm_1

    encoder_model = keras.Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])


    decoder_inputs = model.input[1]   # input_2
    # print(decoder_inputs)

    decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,))
   
    # 使用传入的值来初始化当前模型的输入状态
    lstm1 = model.get_layer('lstm_2') #model.layers[4]
    lstm2 = model.get_layer('lstm_3') #model.layers[5]
    # embedding1 = model.get_layer('embedding_1') #model.layers[4]

    decoder_dense = model.layers[6]
    decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
    decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
    decoder_outputs = decoder_dense(decoder_h2)

    decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
                          [decoder_outputs, state_h1, state_c1, state_h2, state_c2])
    return encoder_model,decoder_model


# encoder_model,decoder_model = build_model()
encoder_model,decoder_model = build_model_1(configpath)
################################################### predict ############################################################
#预测模型中的encoder和训练中的一样，都是输入序列，输出几个状态。而decoder和训练中稍有不同，因为训练过程中的decoder端的输入是可以确定的，因此状态只需要初始化一次，而预测过程中，需要多次初始化状态，因此将状态也作为模型输入。

# encoder模型和训练相同
#
# encoder_h1 = keras.Input(shape=(HIDDEN_SIZE,))
# encoder_state_h1 = keras.Input(shape=(HIDDEN_SIZE,))
# encoder_state_c1 = keras.Input(shape=(HIDDEN_SIZE,))
# encoder_state_h2 = keras.Input(shape=(HIDDEN_SIZE,))
# encoder_state_c2 = keras.Input(shape=(HIDDEN_SIZE,))
#
# # encoder_model = keras.Model(encoder_inputs_, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])
# encoder_model = keras.models.load_model('encode_model.h5')
#



# 预测模型中的decoder的初始化状态需要传入新的状态
# decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_1')
# decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_2')
# decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_3')
# decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_4')
#
# # 使用传入的值来初始化当前模型的输入状态
# decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs_, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
# decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
# decoder_outputs = decoder_dense(decoder_h2)
#
# decoder_model = keras.Model([decoder_inputs_, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
#                       [decoder_outputs, state_h1, state_c1, state_h2, state_c2])


# exit()
random.seed(234)
# idx = random.randint(0,len(id2en))
# print(idx)
task =[

    '又一国军事基地被袭击了，都是替美国出头惹的祸！',
    '关于中央八项规定 总书记是怎样带头执行的？',
    '70后官员被双开:曾是微博大V 前任因集体嫖娼被敲诈',
    '巴基斯坦法院撤销死刑判决 穆沙拉夫回应：真好',
    '美取消对中国＂汇率操纵国＂认定 外交部:本来就不是',
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




    '国外是真难呀，不论做什么都有一帮人吹毛求疵！',
    '吓的把客机当无人机直接击落。',
    '我特想听听专家的看法。',
    '她们还是孩子！',
    '如果让我碰到这些兔崽子，我保证不打死他们',
    '你现在多大了，她多大了？',
    '这些个坏孩子',
    '霸凌的怎么都是女生',
    '霸凌现象难除，是什么原因？',
    '全部抓来剃头',
    '社会的悲衰',
    '心理教育 呵呵',
    '麻痹的',
    '你就是专家了。',
    '用浓硫酸一个个的泼在它们脸上。',

]




def doc2v(tokenizer_source,encoder_text,MAX_LEN,VOCAB_SIZE_SOURCE):
    encoder_sequences = tokenizer_source.texts_to_sequences([list(encoder_text)])
    # print(encoder_sequences)
    # encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32',)# padding='post', truncating='post'
    encoder_input = np.zeros((1, MAX_LEN, VOCAB_SIZE_SOURCE), dtype="float32")

    # for i, seqs in enumerate(encoder_sequences):
    #     for j, seq in enumerate(seqs):
    #         if j > 0:
    #             encoder_input[i][j][seq-1] = 1.

    for seqs in encoder_sequences:
        for j, seq in enumerate(seqs):
            # print(j,seq)
            try:
                encoder_input[0][j][seq-1] = 1.
            except Exception as e:
                # raise e
                pass
            

    return encoder_input


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
        if sampled_token_index == word2id_target[etag] or len(outputs) > 40: break

    # print(en_data[k])
    resptext = ' '.join([id2word_target.get(i,'None') for i in outputs]).replace('end','')
    print('网友发言：{} <----> 机器回复：{}'.format(enchar,resptext))
    # print(resptext)

    # words = pseg.cut(resptext)
    # for word, flag in words:
    #     print('%s %s' % (word, flag))

    # if enchar!=resptext:
    #     task.append(resptext)
    # time.sleep(1)
