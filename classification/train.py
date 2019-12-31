# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import config



x_train = np.random.random((config.corpus_size,config.embedding_size))
y_train = np.random.randint(config.classnum, size=config.corpus_size)
y_train = tf.one_hot(y_train,depth=config.classnum)
print(x_train)
print(y_train)



input_tensor = keras.Input(shape=(config.embedding_size,),dtype=tf.int32,name='input_text')

# input_embdded = keras.layers.Embedding(64,config.vob_size)(input_tensor) # the corpus we generated is already a embeded like data
input_dense = keras.layers.Dense(64,activation='relu')(input_tensor)
output_tensor = keras.layers.Dense(config.classnum,activation='softmax')(input_dense)

tfModel = keras.models.Model(input_tensor,output_tensor)
tfModel.summary()



# train
tfModel.compile(optimizer=tf.keras.optimizers.Adam(config.lr),loss='categorical_crossentropy')
tfModel.fit(x_train,y_train,epochs=config.epoch_size,batch_size=config.batch_size)

# tf.keras.models.save_model(tfModel, "FastText.hp5", save_format="h5")
tfModel.save('model.h5')




