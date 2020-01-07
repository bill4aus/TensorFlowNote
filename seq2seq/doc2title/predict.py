# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import pickle
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
#
# with open('../../datasets/cmn.txt', 'r', encoding='utf-8') as f:
#     data = f.read()
#     data = data.split('\n')
#     # print(data)
#     # data = data[:100]
# print(data[-500:])
# print(len(data))
# data = data[0:1000]
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
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
######################## model ##############################

EN_VOCAB_SIZE = len(tokenizer.word_index)
CH_VOCAB_SIZE = len(tokenizer.word_index)
HIDDEN_SIZE = 256

LEARNING_RATE = 0.003
BATCH_SIZE = 100
EPOCHS = 200
squence_length = 100
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

maxFeature=10000
# tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~！，。（）；’‘',split=" ",num_words=maxFeature)  #创建一个Tokenizer对象

# exit()
random.seed(234)
# idx = random.randint(0,len(id2en))
# print(idx)
task =[
    # '夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。',
    '中国古代玺印的出现，可以上溯至商代。印者，信也，印章的主要功能就是示信，随着佩印风气的形成，吉语印、肖形印等彰显审美功能的多种印式逐渐盛行。唐宋元时期文人用印功能扩大，渐成体系，但中国印章出现新的发展方向。明代中叶以后，石章应用渐盛，便利了文人独立完成创作，篆刻成为新的艺术样式。文人寄情金石，使篆刻逐渐脱离实用功能，重在抒情写意，向着艺术化方向发展。以文彭为代表的文人群体推动风气，一时风从者众，继而衍为流派。明代晚期的文人篆刻家大都受到文彭的影响，后人以文彭为首的吴门派、何震为首的雪渔派和苏宣为首的泗水派，尊为开风气之先的早期流派，文人篆刻进入盛期。明末汪关，清初程邃、林皋、许容等亦各创新风，形成新的流派。清中期的丁敬、邓石如奇峰突起，浙派、邓派从者如云，印风播及南北。高凤翰、高翔、巴慰祖、鞠履厚、乔林等一批区域性名家活跃于印坛，扬州、云间、齐鲁印人群体各呈风貌，派系繁多，争奇斗艳。晚清印人除传承浙、邓二派印风以外，亦受金石学振兴的影响，取资广泛。吴让之师法邓派而面目一新，赵之谦、徐三庚融邓、浙二家而自出机杼，黄士陵独立黟山派，印坛千峰竞秀。印人张扬个性的理念更为自觉，文人篆刻艺术的发展进入新的阶段，对近现代中国篆刻艺术的影响极为深刻。 文彭刻 画隐、梁袠刻 东山草堂珍玩 兩面章由上海博物馆、无锡博物院联合主办的《云心石面——上海博物馆、无锡博物院藏明清文人篆刻特展》将于10月29日在无锡博物院西区二层临展厅隆重展出。此次特展全面系统地展示了明清流派印的发展过程，更融入了无锡地域文化的特征。展出的印章作品，主要包括上海博物馆藏明清流派印和无锡博物院藏明代顾林墓出土的一组明代流派印，共计150余件。 苏宣刻 顾林之印 石章此次展出的印章实物，除上海博物馆藏明清文人篆刻作品以外，另有无锡博物院藏明代顾林墓、锡山区文管会藏明代华师伊墓出土的两组印章，均为首次面世。上海博物馆藏明清篆刻作品中，有相当一部分为无锡籍近代实业家华笃安先生和毛明芬女士捐赠。华氏为无锡望族，元代华幼武，明代华夏、华云、华叔阳，清代华翼纶，近代华绎之等，素有雅好收藏之传统。此次展出也是华笃安旧藏首次回归故里，观众在品味明清篆刻艺术魅力的同时，亦可从中感受到收藏、捐赠者的文化情怀，以及无锡华氏悠久深厚的家族文化传承。 乔林刻 何可一日无此君竹根章 展览详情 展览名称：《云心石面——上海博物馆、无锡博物院藏明清文人篆刻特展》 展览时间：2016.10.29-2017.2.19 展览地点：无锡博物院西区二层临展厅 主办单位：上海博物馆、无锡博物院 图文自：无锡博物院｜编辑：小仙(声明：本文仅代表作者观点，不代表文博圈立场)往期精选阅读（直接点击进入）博物馆该如何迎接IP运营时代？国家文物局：净收入50%奖励文创有功人员，企业享受资金税收扶持政策一个馆长的魄力：所有博物馆都必须改变看完考古学家吃饭，很多人服了怎样才能成为一个出色的讲解员？博物馆数字技术的现在和未来博物馆旅游功能日益突出，如何打造博物馆奇妙之旅？他把博物馆当做文化旅游项目来运作从一座馆，看一座城我们有多少博物馆，能让公众产生文化依赖博物馆数字化的可持续发展博物馆如何应对新科技的挑战？博物馆文物修复行业，为啥留不住人？他对文物界来了一次“拨乱反正”视频：海昏侯墓的考古故事，居然这么有趣陈列部主任每天在干什么？敦煌原创动画《降魔成道》引起一片叫好！文博圈，qq群 149299743'
]




def doc2v(tokenizer,encoder_text,MAX_LEN,VOCAB_SIZE):
    encoder_sequences = tokenizer.texts_to_sequences([encoder_text])
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_output_data = np.zeros((1, MAX_LEN, VOCAB_SIZE), dtype="float32")
    for i, seqs in enumerate(encoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq-1] = 1.
    return decoder_output_data


for enchar in task:
    # test_data = encoder_input_data[k:k + 1]
    test_data = doc2v(tokenizer,enchar,squence_length,len(tokenizer.index_word))
    test_input = test_data[0].T
    print(test_input.shape)
    # exit()



    h1, c1, h2, c2 = encoder_model.predict(test_input)
    target_seq = np.zeros((squence_length, CH_VOCAB_SIZE))
    target_seq[0, tokenizer.word_index['\t']] = 1
    outputs = []
    while True:
        output_tokens, h1, c1 = decoder_model.predict([target_seq.T, h1, c1])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        outputs.append(sampled_token_index)
        target_seq = np.zeros(( squence_length, CH_VOCAB_SIZE))
        target_seq[0, sampled_token_index] = 1.
        if sampled_token_index == tokenizer.word_index['\n'] or len(outputs) > 20: break

    # print(en_data[k])
    print(enchar)
    print(outputs)
    print(''.join([tokenizer.index_word[i] for i in outputs]))
