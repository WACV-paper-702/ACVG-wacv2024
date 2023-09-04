""" this is file where all the configuration for training the network including the image sizes, batch size, past timestep and future
 or prediction time steps are provided. We also provide the dat path, model save paths here only. The main file and model file creats 
 an instance of the config object """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf
tf.random.set_seed(77)
# import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

import os
import numpy as np
from os.path import exists
from os import makedirs


def define_network_flags():
  flags.DEFINE_integer('buffer_size', 50000, 'Shuffle buffer size')
  flags.DEFINE_integer('batch_size',os.environ.get('BATCH_SZ'), 'Batch Size')
  flags.DEFINE_integer('image_h', os.environ.get('IMG_H'), 'image_h')
  flags.DEFINE_integer('image_w', os.environ.get('IMG_W'), 'image_w')
  flags.DEFINE_float('lr', 0.0001, 'lr')
  flags.DEFINE_float('lr_acvg', 0.00000001, 'lr_acvg')
  flags.DEFINE_float('alpha', 1.0, 'alpha')
  flags.DEFINE_float('beta', 0.0001, 'beta')
  flags.DEFINE_integer('c_dim', os.environ.get('C_DIM'), 'c_dim')
  flags.DEFINE_integer('a_dim', os.environ.get('A_DIM'), 'a_dim')
  flags.DEFINE_integer('past_TS', os.environ.get('PAST_TS'), 'past_TS')
  flags.DEFINE_integer('future_TS', os.environ.get('FUTURE_TS'), 'future_TS')
  flags.DEFINE_integer('test_future_TS', os.environ.get('TEST_FUTURE_TS'), 'test_future_TS')
  flags.DEFINE_integer('epochs', os.environ.get('EPOCHS'), 'Number of epochs')
  flags.DEFINE_integer('epochs_actor', os.environ.get('EPOCHS_ACTOR'), 'Number of epochs for actor')
  flags.DEFINE_integer('epochs_acvg', os.environ.get('EPOCHS_ACVG'), 'Number of epochs for acvg')
  flags.DEFINE_float('beta1', 0.5, 'beta1')
  flags.DEFINE_integer('gpu', os.environ.get('GPU_ID'), 'GPU')
  flags.DEFINE_string('ckpt_G', os.environ.get('CKPT_G'), 'CKPT_G')
  flags.DEFINE_string('ckpt_D', os.environ.get('CKPT_D'), 'CKPT_D')
  flags.DEFINE_string('ckpt_A', os.environ.get('CKPT_A'), 'CKPT_A')
  flags.DEFINE_integer('filters', os.environ.get('FILTERS'), 'filters')
  flags.DEFINE_integer('a_filters', os.environ.get('A_FILTERS'), 'a_filters')
  flags.DEFINE_integer('df_dim', 32, 'df_dim')
  flags.DEFINE_float('margin', 0.3, 'margin')
  flags.DEFINE_boolean('tf_enable', True, 'tf_Enable?')
  flags.DEFINE_boolean('G_train',os.environ.get('G_TRAIN'), 'G_train')
  flags.DEFINE_boolean('A_train', os.environ.get('A_TRAIN'), 'A_train')
  flags.DEFINE_boolean('test_G', os.environ.get('TEST_G'), 'test_G')
  # flags.DEFINE_string('datapath', "../../../catkin_ws/RoAM_dataset/processed/final", 'Directory to  the dataset')
  # flags.DEFINE_string('datapath', "../../../catkin_ws/RoAM_dataset/processed/final/test", 'Directory to  the dataset')
  # flags.DEFINE_string('datapath', "../../VANET/data/RoAM_dataset/processed/final", 'Directory to  the dataset')
  flags.DEFINE_string('datapath', os.environ.get('DATAPATH'), 'Directory to  the dataset')
  flags.DEFINE_string('model_name', 'ACPNet', 'Deciding model name')
  
  # flags.DEFINE_string('train_mode', 'custom_loop',
  #                     'Use either "keras_fit" or "custom_loop"')

def flags_dict():
  """Define the flags.

  Returns:
    Command line arguments as Flags.
  """

  kwargs = {
      'epochs': FLAGS.epochs,
      'epochs_actor': FLAGS.epochs_actor,
      'epochs_acvg': FLAGS.epochs_acvg,
      'tf_enable': FLAGS.tf_enable,
      'buffer_size': FLAGS.buffer_size,
      'batch_size': FLAGS.batch_size,
      'image_h': FLAGS.image_h,
      'image_w': FLAGS.image_w,
      'model_name': FLAGS.model_name,
      'lr': FLAGS.lr,
      'lr_acvg': FLAGS.lr_acvg,
      'alpha': FLAGS.alpha,
      'beta': FLAGS.beta,
      'beta1': FLAGS.beta1,
      'c_dim': FLAGS.c_dim,
      'a_dim': FLAGS.a_dim,
      'past_TS': FLAGS.past_TS,
      'future_TS': FLAGS.future_TS,
      'test_future_TS': FLAGS.test_future_TS,
      'gpu': FLAGS.gpu,
      'filters': FLAGS.filters,
      'a_filters': FLAGS.a_filters,
      'df_dim': FLAGS.df_dim,
      'margin': FLAGS.margin,
      'G_train': FLAGS.G_train,
      'A_train': FLAGS.A_train,
      'test_G': FLAGS.test_G,
      'ckpt_G': FLAGS.ckpt_G,
      'ckpt_D': FLAGS.ckpt_D,
      'ckpt_A': FLAGS.ckpt_A,
      'datapath': FLAGS.datapath
  }
  return kwargs

class Config(object):
    def __init__(self,model_name="ACPNet"):
        self.image_h=64
        self.image_w=64
        self.image_size=[self.image_h, self.image_w]
        self.batch_size=16
        self.lr=0.0001
        self.alpha=1.0
        self.beta=0.0001
        self.c_dim=3
        self.a_dim=2
        self.past_TS=5
        self.future_TS=10
        self.epochs=150000
        self.gpu=0
        self.beta1=0.5
        self.iter_start=0
        self.filters=32
        self.df_dim=32
        self.margin=0.3
        self.is_train=True
        self.datapath="../../VANET/data/RoAM_dataset/processed/final"
        if not exists(self.datapath):
            raise Exception("Sorry, invalid datapath")
        self.model_name=model_name
        self.prefix = ("Roam_Full-v1_{}".format(self.model_name)
              + "_GPU_id="+str(self.gpu)
              + "_image_w="+str(self.image_size)
              + "_K="+str(self.past_TS)
              + "_T="+str(self.future_TS)
              + "_batch_size="+str(self.batch_size)
              + "_alpha="+str(self.alpha)
              + "_beta="+str(self.beta)
              + "_lr="+str(self.lr)
              +"_no_epochs="+str(self.epochs)+"_beta1"+str(self.beta1))
        self.checkpoint_dir = "../models/"+self.prefix+"/"
        self.samples_dir = "../samples/"+self.prefix+"/"
        self.logs_dir = "../logs/"+self.prefix+"/"
        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)
        if not exists(self.samples_dir):
            makedirs(self.samples_dir)
        if not exists(self.logs_dir):
            makedirs(self.logs_dir)
        

