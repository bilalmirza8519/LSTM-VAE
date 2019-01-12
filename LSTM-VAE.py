import numpy as np
import pandas as pd
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from keras.layers.recurrent import LSTM
from keras.layers import Input, LSTM, RepeatVector, Masking, TimeDistributed, Lambda
from keras.losses import mse, binary_crossentropy, mean_squared_error
from keras.models import Model
from keras import backend as K


#define variables
latent_dim=3
Intermediate_dim=6
nb_epoch=1000
batch_size=100
optimizer='adadelta'
#X is the data matrix

#encoder LSTM
inputs = Input(shape=(7, 1), name='InputTimeSeries')  #(timesteps, input_dim)
encoded = LSTM(Intermediate_dim, name='EncoderLSTM')(inputs) # intermediate dimension

#Creating mean and sigma vectors
z_mean = Dense(latent_dim, name='MeanVector' )(encoded)
z_log_sigma = Dense(latent_dim,name='SigmaVector')(encoded)

#latent vector sampling
def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

z = Lambda(sampling, name='LatentVector', output_shape=(latent_dim,))([z_mean, z_log_sigma])  

#VAE Loss
def vae_loss(inputs, decoded):
   
    xent_loss = K.sum(K.binary_crossentropy(inputs, decoded), axis=1)
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1) 
    return K.mean(xent_loss + kl_loss)


#decoder LSTM
decoded = RepeatVector(7, name='EmbeddingtoTimeSeries')(z) #timesteps
decoded = LSTM(Intermediate_dim,name='DecoderLSTM1', return_sequences=True)(decoded) #intermediate dimensions
decoded = LSTM(1,name='DecoderLSTM2', return_sequences=True)(decoded) #input_dim

#decoded=TimeDistributed(Dense(1, name='Wrapper'), name='TimeDistributed')(decoded)  

v_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, z_mean)  
v_autoencoder.summary()

v_autoencoder.compile(optimizer=optimizer, loss=vae_loss)
v_autoencoder.fit(X,X,nb_epoch=nb_epoch,batch_size=batch_size)  
