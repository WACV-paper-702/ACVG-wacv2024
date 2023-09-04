# from typing import Any
import tensorflow as tf
tf.random.set_seed(77)
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras as K
# from tensorflow import keras
from keras.layers import *
# from K.layers import *
from keras.layers.rnn.base_conv_lstm import ConvLSTMCell
from keras.layers import LSTMCell
# import utils
# from loss import *
# from ops import *

SEED=77
# class conv2d(tf.Module):
class conv2d(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,name,activation='relu', padding='same',initializer=None,**kwargs):
        super().__init__(name=name,**kwargs)
        # self.name=name
        self.filters=filters
        self.initializer_w=tf.keras.initializers.GlorotUniform(seed=SEED)
        self.initializer_b=tf.keras.initializers.Constant(0.0)
        self.activation=activation
        self.kernel_size=kernel_size
        # with self.name_scope:
            # if initializer == None:
            #     self.conv2d=Conv2D(filters=self.filters,kernel_size=self.kernel_size,name=self.name,padding=padding,
            #                     kernel_initializer=self.initializer_w,bias_initializer= self.initializer_b,
            #                     activation=self.activation,data_format="channels_last",**kwargs)
            # else:
            #     self.conv2d=Conv2D(filters=self.filters,kernel_size=self.kernel_size,name=self.name,padding="same",
            #                     kernel_initializer=initializer,bias_initializer= 'zeros',
            #                     activation=self.activation,data_format="channels_last",**kwargs)
        if initializer == None:
            self.conv2d=Conv2D(filters=self.filters,kernel_size=self.kernel_size,padding=padding,
                            kernel_initializer=self.initializer_w,bias_initializer= self.initializer_b,
                            activation=self.activation,data_format="channels_last",**kwargs)
        else:
            self.conv2d=Conv2D(filters=self.filters,kernel_size=self.kernel_size,padding="same",
                            kernel_initializer=initializer,bias_initializer= 'zeros',
                            activation=self.activation,data_format="channels_last",**kwargs)
    # @tf.Module.with_name_scope
    def call(self,input):
        return self.conv2d(input)
    
    
class deconv2d(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,name,activation='relu', padding='same',initializer=None,**kwargs):
        super().__init__(name=name,**kwargs)
        # self.name=name
        self.filters=filters
        self.initializer_w=tf.keras.initializers.GlorotUniform(seed=SEED)
        self.initializer_b=tf.keras.initializers.Constant(0.0)
        self.activation=activation
        self.kernel_size=kernel_size
        # with self.name_scope:
        if initializer == None:
            self.deconv2d=Conv2DTranspose(filters=self.filters,kernel_size=self.kernel_size,padding=padding,
                            kernel_initializer=self.initializer_w,bias_initializer= self.initializer_b,
                            activation=self.activation,data_format="channels_last",**kwargs)
        else:
            self.deconv2d=Conv2DTranspose(filters=self.filters,kernel_size=self.kernel_size,padding="same",
                            kernel_initializer=initializer,bias_initializer= 'zeros',
                            activation=self.activation,data_format="channels_last",**kwargs)
    # @tf.Module.with_name_scope
    def call(self,input):
        return self.deconv2d(input)

class maxpool2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=[2,2],strides=2, padding='valid',name=None, **kwargs):
        super().__init__(name=name,**kwargs)
        self.pool_size=pool_size
        self.strides=strides
        # with self.name_scope:
        self.maxpool2D=MaxPool2D( pool_size=self.pool_size,strides=self.strides,padding=padding,name=name,**kwargs)
    # @tf.Module.with_name_scope    
    def call(self,input):
        return self.maxpool2D(input)   


class upsampling2D(tf.keras.layers.Layer):
    def __init__(self,size=[2,2],interpolation='bilinear',name=None, **kwargs): 
        super().__init__(name=name,**kwargs)
        self.size=size
        # with self.name_scope:
        self.upsampling2D=UpSampling2D(size=self.size, interpolation=interpolation, **kwargs)
    # @tf.Module.with_name_scope
    def call(self,input):
        return self.upsampling2D(input) 
    
class convLSTM2Dcell(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,name,activation='relu', padding='same',initializer=None,**kwargs):
        super().__init__(name=name,**kwargs)
        self.rank=2   ##2D convolution is hard coded for convLSTMcell 
        self.filters=filters
        self.kernel=kernel_size
        # self.name=name
        self.activation=activation
        self.kernel_initializer=K.initializers.GlorotUniform(seed=SEED)
        self.initializer_b= K.initializers.Constant(0.0)
        # with self.name_scope:
        if initializer == None:
            self.convLSTM2Dcell=ConvLSTMCell(rank=self.rank,filters=self.filters,kernel_size=self.kernel,
                                    padding=padding,kernel_initializer=self.kernel_initializer,
                                    bias_initializer= self.initializer_b,recurrent_initializer="orthogonal",
                            activation=self.activation,data_format="channels_last",**kwargs)
        else:
            self.convLSTM2Dcell=ConvLSTMCell(rank=self.rank,filters=self.filters,kernel_size=self.kernel,padding="same",
                            kernel_initializer=initializer,bias_initializer= 'zeros',recurrent_initializer="orthogonal",
                            activation=self.activation,data_format="channels_last",**kwargs)
    # @tf.Module.with_name_scope
    def call(self,input,state):
        return self.convLSTM2Dcell(input,state)
    
class LSTMcell(tf.keras.layers.Layer):
    def __init__(self,units,name,activation='tanh',initializer=None,**kwargs):
        super().__init__(name=name,**kwargs)
        # self.rank=2   ##2D convolution is hard coded for convLSTMcell 

        self.units=units
        self.activation=activation
        self.kernel_initializer=K.initializers.GlorotUniform(seed=SEED)
        self.initializer_b= K.initializers.Constant(0.0)
        # with self.name_scope:
        if initializer == None:
            self.LSTMcell=LSTMCell(units=self.units,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer= self.initializer_b,recurrent_initializer="orthogonal",
                            activation=self.activation,**kwargs)
        else:
            self.LSTMcell=LSTMCell(units=self.units,
                            kernel_initializer=initializer,bias_initializer= 'zeros',recurrent_initializer="orthogonal",
                            activation=self.activation,**kwargs)
    # @tf.Module.with_name_scope
    def call(self,input,state):
        return self.LSTMcell(input,state)
    
class dropout(tf.keras.layers.Layer):
    def __init__(self,rate,name=None ,seed=SEED,**kwargs):
        super().__init__(name=name,**kwargs)
        self.rate=rate
        self.dropout=Dropout(rate=self.rate,seed=seed)
    def call(self,input):
        return self.dropout(input)


    
class avgpool2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=[2,2],strides=2, padding='valid',name=None ,**kwargs):
        super().__init__(name=name,**kwargs)
        self.pool_size=pool_size
        self.strides=strides
        # with self.name_scope:
        self.avgpool2D=AveragePooling2D( pool_size=self.pool_size,strides=self.strides,padding=padding,**kwargs)
    # @tf.Module.with_name_scope    
    def call(self,input):
        return self.avgpool2D(input) 
    
class input(tf.keras.layers.Layer):
    def __init__(self,input_shape,batch_size,name,dtype=tf.float32,**kwargs):
        super().__init__(name=name,**kwargs)
        self.in_shape=input_shape
        self.batch_size=batch_size
        # self.name=name
        # with self.name_scope:
        self.inputLayer=InputLayer(input_shape=self.in_shape, batch_size=self.batch_size,dtype=dtype,**kwargs)
    # @tf.Module.with_name_scope
    def call(self,input):
        return self.inputLayer(input)
    
class batchNorm(tf.keras.layers.Layer):
    def __init__(self,axis=-1,scale=True,beta_initializer='zeros',gamma_initializer='ones',moving_mean_initializer='zeros',moving_variance_initializer='ones',name=None,**kwargs):
        super().__init__(name=name,**kwargs)
        # with self.name_scope:
        self.batchNorm=BatchNormalization(axis=axis,scale=scale,beta_initializer=beta_initializer,
                                        gamma_initializer=gamma_initializer,moving_mean_initializer=moving_mean_initializer,
                                        moving_variance_initializer=moving_variance_initializer,synchronized=True,**kwargs ) ##synched for distributed strategies
    # @tf.Module.with_name_scope
    def call(self, input):
        return self.batchNorm(input)
    
class dense(tf.keras.layers.Layer):
    def __init__(self,output_shape,name,activation=None, initializer=None,**kwargs):
        super().__init__(name=name,**kwargs)
        self.out_shape=output_shape
        # self.name=name
        self.initializer_w=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=SEED)
        self.initializer_b=tf.keras.initializers.Constant(0.0)
        # with self.name_scope:
        if initializer == None:
            self.dense=Dense(units=self.out_shape,kernel_initializer=self.initializer_w,
                        bias_initializer=self.initializer_b, activation=activation,**kwargs )
        else:
            self.dense=Dense(units=self.out_shape,kernel_initializer=initializer,
                        bias_initializer=self.initializer_b, activation=activation,**kwargs )
    # @tf.Module.with_name_scope        
    def call(self,input):
        return self.dense(input)
    
class flatten(tf.keras.layers.Layer):
    def __init__(self,name=None ,**kwargs):
        super().__init__(name=name,**kwargs)
        # with self.name_scope:
        self.flatten=Flatten()
    # @tf.Module.with_name_scope    
    def call(self,input):
        return self.flatten(input) 

        