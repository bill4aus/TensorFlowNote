# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import config

new_tfModel = keras.models.load_model('model.h5')


x_test = np.random.random((1,config.embedding_size))
# print(x_test)
print(x_test.shape)

res = new_tfModel.predict(x_test)
print(res.ndim) #2
resclass = np.argmax(res,axis=1) # res[0][x] so axis =1
# resclass = new_tfModel.predict_classes(x_test)   # Sequential model only
print(res)
print(resclass)





