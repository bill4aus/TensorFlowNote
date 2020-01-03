# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import pickle
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
en_file ={}
with open("en.file", "rb") as f:
    en_file = pickle.load(f)

zh_file ={}
with open("zh.file", "rb") as f:
    zh_file = pickle.load(f)
m_config = {}
with open("config.file", "rb") as f:
    m_config = pickle.load(f)
print(m_config)
# 分别生成中英文字典
# en_vocab = en_file['id2en']
id2en = en_file['id2en']
en2id = en_file['en2id'] #{c:i for i,c in enumerate(id2en)}

# ch_vocab = zh_file['vocab']
id2ch = zh_file['id2ch']
ch2id = zh_file['ch2id'] #{c:i for i,c in enumerate(id2ch)}

print('\n英文字典:\n', en2id)
print('\n中文字典共计\n:', ch2id)
#
# # 建立字典,将文本数据映射为数字数据形式。
# en_num_data = [[en2id[en] for en in line ] for line in en_data]
# ch_num_data = [[ch2id[ch] for ch in line] for line in ch_data]
# de_num_data = [[ch2id[ch] for ch in line][1:] for line in ch_data]

#
# print('char:', en_data[1])
# print('index:', en_num_data[1])



#one hot 数据格式改为onehot的格式
#
# # 获取输入输出端的最大长度
# max_encoder_seq_length = max([len(txt) for txt in en_num_data])
# max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
# print('max encoder length:', max_encoder_seq_length)
# print('max decoder length:', max_decoder_seq_length)

# encoder_input_data = np.zeros((len(en_num_data), max_encoder_seq_length, len(en2id)), dtype='float32')
# decoder_input_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
# decoder_target_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
# #
# for i in range(len(ch_num_data)):
#     for t, j in enumerate(en_num_data[i]):
#         encoder_input_data[i, t, j] = 1.
#     for t, j in enumerate(ch_num_data[i]):
#         decoder_input_data[i, t, j] = 1.
#     for t, j in enumerate(de_num_data[i]):
#         decoder_target_data[i, t, j] = 1.
#
# print('index data:\n', en_num_data[1])
# # print('one hot data:\n', encoder_input_data[1])
# print(encoder_input_data.shape)
# print(decoder_input_data.shape)
#

def entext2token(enchar):
    tokenarray = [[en2id[en] for en in enchar]]
    encoder_input_data = np.zeros((1, m_config['max_encoder_seq_length'], len(en2id)), dtype='float32')
    for i in range(len(tokenarray)):
        for t, j in enumerate(tokenarray[i]):
            encoder_input_data[i, t, j] = 1.
    return encoder_input_data

######################## model ##############################

EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)
HIDDEN_SIZE = 256

LEARNING_RATE = 0.003
BATCH_SIZE = 100
EPOCHS = 200
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



model = keras.models.load_model('ss2ss_test.h5')

print(model.summary())
print(ch2id['\n'])
print(ch2id['\t'])
# for layer in model.layers:
#     print(layer._name)
#     print(layer)
#     print('-')

# print(model.layers[0].output)
# print(model.layers[1].output)
# print(model.layers[2].output)
# print(model.layers[3].output)

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

decoder_state_input_h1 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_1')
decoder_state_input_c1 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_2')
decoder_state_input_h2 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_3')
decoder_state_input_c2 = keras.Input(shape=(HIDDEN_SIZE,),name='ipt_4')

# 使用传入的值来初始化当前模型的输入状态
lstm1 = model.get_layer('lstm_2') #model.layers[4]
lstm2 = model.get_layer('lstm_3') #model.layers[5]
decoder_dense = model.layers[6]
decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
# decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[encoder_state_h1, encoder_state_c1])
decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
# decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[encoder_state_h2, encoder_state_c2])
decoder_outputs = decoder_dense(decoder_h2)

decoder_model = keras.Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
                      [decoder_outputs, state_h1, state_c1, state_h2, state_c2])

# decoder_model = keras.Model([decoder_inputs, encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2],
#                       [decoder_outputs, state_h1, state_c1, state_h2, state_c2])


# print(model.layers[2])


# print(model.get_layer('lstm_1'))
# print(model.get_layer('lstm_2'))
# print(model.get_layer('lstm_3'))
#
#
# encoder_inputs_ = model.inputs[0] #model.get_layer('input_1').input  #model.inputs[0] #model.layers[0].outputs #keras.Input(shape=(None, EN_VOCAB_SIZE))
# # https://stackoverflow.com/questions/57611085/how-to-save-and-reload-hidden-states-of-keras-encoder-decoder-model-for-inferenc
# # the Graph disconnected error occurred due to referencing an incorrect instance of the stacked LSTM layers, should have been newmodel.layers[6].output
# encoder_h1, encoder_state_h1, encoder_state_c1 = model.layers[2].output
# encoder_h2, encoder_state_h2, encoder_state_c2 = model.layers[3].output
#
# decoder_inputs_ = model.inputs[1] #model.get_layer('input_2').input  #model.inputs[1] #model.layers[1].inputs #keras.Input(shape=(None, CH_VOCAB_SIZE))
# print(decoder_inputs_)
# # exit()
# decoder_outputs = model.output[0] #model.get_layer('dense').output #model.output[0] #model.layers[6].outputs
# # exit()
#

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

#
# # print(ch2id['\n'])
# # print(ch2id['\t'])
# task =[
#     'i love you',
#     'i hate you',
#     'i miss you',
#     'i run ',
#     'i kill you',
#     'i see you',
# ]
#
#
# for enchar in task:
#     test_data = entext2token(enchar)
#     print(test_data.shape)
#     # test_data = encoder_input_data[k:k + 1]
#     # print(test_data)
#
#     # print(encoder_model.predict(test_data))
#     # h1, c1, h2, c2 = encoder_model.predict(test_data)
#     # target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
#     # target_seq[0, 0, ch2id['\t']] = 1
#     # outputs = []
#     # while True:
#     #     output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
#     #     sampled_token_index = np.argmax(output_tokens[0, -1, :])
#     #     outputs.append(sampled_token_index)
#     #     target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
#     #     target_seq[0, 0, sampled_token_index] = 1
#     #     if sampled_token_index == ch2id['\n'] or len(outputs) > 20: break
#     #
#     # print(en_data[k])
#     # print(''.join([id2ch[i] for i in outputs]))
#
#     # Encode the input as state vectors.
#     # states_value = encoder_model.predict(test_data)
#     h1, c1, h2, c2 = encoder_model.predict(test_data)
#     # print(len(states_value))
#     # Generate empty target sequence of length 1.
#     target_seq = np.zeros((1, 1,CH_VOCAB_SIZE))
#     # Populate the first character of target sequence with the start character.
#     target_seq[0, 0, ch2id['\t']] = 1.
#
#     # print(target_seq.shape)
#
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = ''
#     outputs=[]
#     # while not stop_condition:
#     while True:
#         # print(len(decoder_model.predict([target_seq , states_value[0] , states_value[1]])))
#         # output_tokens, h1, c1, h2, c2
#         output_tokens, h1, c1, h2, c2  = decoder_model.predict([target_seq,h1, c1, h2, c2]) #返回一个list，和输入对应
#         # to_split  = decoder_model.predict([target_seq,h1, c1]) #返回一个list，和输入对应
#         # for xx in  to_split:
#         #     print(xx.shape)
#         # exit()
#
#         # print(output_tokens)
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         # sampled_token_index = np.argmax(output_tokens[-1,:])
#         # print(sampled_token_index)
#         # print(sampled_token_index.shape)
#         # print(sampled_token_index)
#
#         outputs.append(sampled_token_index)
#         # sampled_char = reverse_target_char_index[sampled_token_index]
#         # decoded_sentence += sampled_char
#         target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
#         target_seq[0, 0, sampled_token_index] = 1.
#         # or len(outputs) > 20
#         if sampled_token_index == ch2id['\n'] or len(outputs) > 20:
#             break
#             # stop_condition = True
#
#
#         # # Exit condition: either hit max length
#         # # or find stop character.
#         # if (sampled_char == '\n' or
#         #         len(decoded_sentence) > max_decoder_seq_length):
#         #     stop_condition = True
#         #
#         # # Update the target sequence (of length 1).
#         # target_seq = np.zeros((1, 1, num_decoder_tokens))
#         # target_seq[0, 0, sampled_token_index] = 1.
#
#         # Update states
#         # states_value = [h, c]
#     # print(''.join(outputs))
#     print(enchar)
#     # print(en_data[k])
#     print(''.join([ id2ch[idx] for idx in outputs]))
#




task =[
    'Have a good Christmas.',
    'Do you believe in God?',
    'You should apologize.',
    'Where is the mailbox?',
    'We are having dinner.',
    'We got to be friends.',
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
        target_seq[0, 0, sampled_token_index] = 1.
        if sampled_token_index == ch2id['\n'] or len(outputs) > 20: break

    # print(en_data[k])
    print(enchar)
    print(''.join([id2ch[i] for i in outputs]))
