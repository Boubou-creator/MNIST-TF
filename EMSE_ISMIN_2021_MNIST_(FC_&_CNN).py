#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[ ]:


# Type of training set (x and y)
print('Type of x_train:', type(x_train))
print('Type of y_train:', type(y_train))


# In[ ]:


# Shape of training set (x and y)
print('Shape of x_train:', x_train.shape)
print('Shape of y_train:', y_train.shape)


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


# Plot images from training set
rndperm = np.random.permutation(x_train.shape[0])
# rndperm = np.arange(12)
nbimgx, nbimgy = 4, 3
plt.figure()
for i in range(nbimgx * nbimgy):
    plt.subplot(nbimgy, nbimgx, i + 1)
    plt.tick_params(which='both',
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.imshow(x_train[rndperm[i]])
plt.show()
print('Associated labels:', y_train[rndperm[0:nbimgx * nbimgy]])


# In[ ]:


#creat model Auto-Encoder
lin = tf.keras.layers.Input((28,28,1))
lact = lin
lact = tf.keras.layers.ZeroPadding2D(1)(lact)  #(30,30,1)
lact = tf.keras.layers.Conv2D(64,3, activation='relu')(lact) #(28,28,64)
#lact = tf.keras.layers.Conv2D(64,3, use_bias=False)(lact)
#lact = tf.keras.layers.BatchNormalisation()(lact)
#lact = tf.keras.layers.Activation(relu)(lact)
lact = tf.keras.layers.AvgPool2D(2)(lact) #(14,14,64)

lact = tf.keras.layers.Conv2D(32,3, activation='relu')(lact) #(12,12,32)
lact = tf.keras.layers.AvgPool2D(2)(lact)#(6,6,32)

lact = tf.keras.layers.Conv2D(32,3, activation='relu')(lact) #(4,4,32)
lact = tf.keras.layers.AvgPool2D(2)(lact)#(2,2,32)

lact = tf.keras.layers.Conv2D(16,2, activation='relu')(lact) #(1,1,16)

lact = tf.keras.layers.Conv2DTranspose(32,2,activation='relu')(lact)

lact = tf.keras.layers.UpSampling2D(2)(lact)
lact = tf.keras.layers.Conv2DTranspose(32,3,activation='relu')(lact)

lact = tf.keras.layers.UpSampling2D(2)(lact)
lact = tf.keras.layers.Conv2DTranspose(64,3,activation='relu')(lact)

lact = tf.keras.layers.UpSampling2D(2)(lact)
lact = tf.keras.layers.Conv2DTranspose(1,1,activation='relu')(lact)

lout = tf.keras.layers.Activation('tanh')(lact)
model_ae = tf.keras.models.Model(lin,lout)
model_ae.summary()


# In[ ]:


unique, count = np.unique(np.max(x_train, axis=(1, 2)), return_counts=True)
print('Max. per image:', unique, '| Count:', count)
unique, count = np.unique(np.min(x_train, axis=(1, 2)), return_counts=True)
print('Min. per image:', unique, '| Count:', count)
x_mean = np.mean(x_train, axis=0).astype(np.uint8)
plt.figure()
plt.tick_params(which='both',
                bottom=False, left=False,
                labelbottom=False, labelleft=False)
plt.imshow(x_mean)
plt.show()
print('avg. of mean per pixel: %.2f' % np.mean(x_mean))


# In[ ]:


def pre_process(x_train, y_train, x_test, y_test, pp='None'):
  y_train_pp = tf.keras.utils.to_categorical(y_train.astype('float32'))
  y_test_pp = tf.keras.utils.to_categorical(y_test.astype('float32'))
  if pp == 'None':
    return x_train, y_train_pp, x_test, y_test_pp
  if pp == '255':
    return x_train / 255., y_train_pp, x_test / 255., y_test_pp
  if pp == 'Samp':
    x_train_pp = x_train - np.expand_dims(np.mean(x_train, axis=(1 ,2)),
                                          axis=(1, 2))
    x_train_pp /= np.expand_dims(np.maximum(np.ones(x_train.shape[0]),
                                            np.max(x_train, axis=(1, 2))),
                                 axis=(1, 2))
    x_test_pp = x_test - np.expand_dims(np.mean(x_test, axis=(1 ,2)),
                                        axis=(1, 2))
    x_test_pp /= np.expand_dims(np.maximum(np.ones(x_test.shape[0]),
                                           np.max(x_test, axis=(1 ,2))),
                                axis=(1, 2))
    return x_train_pp, y_train_pp, x_test_pp, y_test_pp
  raise Exception('Unknown pre-process')


# In[ ]:


xtrain, ytrain, xtest, ytest = pre_process(x_train, y_train, x_test, y_test,
                                           pp='255')


# In[ ]:


print(ytrain.shape)


# In[ ]:


print(ytrain[0])


# In[ ]:


# FC ANN
lin = tf.keras.layers.Input(shape=(28,28))
lact =lin
lact = tf.keras.layers.Flatten()(lact)
lact = tf.keras.layers.Dense(50)(lact)
lact = tf.keras.layers.Activation('sigmoid')(lact)
lact = tf.keras.layers.Dense(10)(lact)
lout = tf.keras.layers.Activation('softmax')(lact)
model = tf.keras.Model(lin, lout)
model.summary()


# In[ ]:


def mish(x):
  return x*tf.math.tanh(tf.math.softplus(x))
# CNN ANN

lin = tf.keras.layers.Input(shape=(28,28))
lact =lin
lact = tf.keras.layers.Reshape((28,28,1))(lact)
lact = tf.keras.layers.Conv2D(16, (3,3))(lact)
lact = tf.keras.layers.Activation(mish)(lact)

lact = tf.keras.layers.MaxPool2D(2)(lact)
lact = tf.keras.layers.Conv2D(32, (3,3))(lact)
lact = tf.keras.layers.Activation('sigmoid')(lact)
#lact = tf.keras.layers.Flatten()(lact)
lact = tf.keras.layers.GlobalMaxPool2D()(lact)
lact = tf.keras.layers.Dense(10)(lact)
lout = tf.keras.layers.Activation('softmax')(lact)
model = tf.keras.Model(lin, lout)
model.summary()


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])


# In[ ]:


log_dir = 'tb'
get_ipython().system('rm -Rf {log_dir}')
get_ipython().run_line_magic('load_ext', 'tensorboard')
# %reload_ext tensorboard
get_ipython().run_line_magic('tensorboard', '--logdir {log_dir}')
# !kill 97


# In[ ]:


cb = list()
cb.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch'))
model.fit(x=xtrain, y=ytrain, batch_size=4000, epochs=600, validation_split=.1,
          callbacks=cb, verbose=1)


# In[ ]:




