import sys
import math
import numpy as np 
import tensorflow as tf
tf.random.set_seed(77)
from tensorflow import keras
from keras.losses import Loss
# tf.compat.v1.enable_eager_execution()
# from joblib import Parallel, delayed

from tensorflow.python.framework import ops

from utils import *

class sigmoid_cross_entropy_with_logits(Loss):
  def __init__(self,name="cross_entropy",**kwargs):
      super().__init__(name=name, **kwargs)


  def call(self, logits=None, labels=None):
     loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels))
     return loss
  
  
  

def stgdl_v2(gen_frames, gt_frames, gen_frames_dt, gt_frames_dt, alpha, image_size, channel_no):
  """
  Arguments:
    gen_frames: Predicted frames or temporal difference of predicted frames (predicted velocity map).
    gt_frames: Ground truth predicted frames or the temporal difference of the ground truth frames (true velocity map).
    gen_frames_dt: The predicted second order temporal difference velocity map if gen_frames is predicted frames or acceleration map
                    if gen_frames is velocity map
    gt_frames_dt: The ground truth second order temporal difference velocity map if gt_frames is predicted frames or acceleration map
                    if gt_frames is velocity map
    alpha: The power to which each gradient term is raised.
    image_size: tuple for image shape (h,w)
  
  Returns: 
    The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(channel_no), dtype=tf.float32)
  neg = -1 * pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
  # [[1],[-1]]
  filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
  # [[[-1]],[[1]]]
  strides1 = [1, 1, 1, 1]  # stride of (1, 1)
  padding = 'SAME'

  gen_dx = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  
  

  grad_diff_x = tf.abs(gt_dx - gen_dx)
  grad_diff_y = tf.abs(gt_dy - gen_dy)
  grad_diff_t = tf.abs(gen_frames_dt - gt_frames_dt)
  grad_diff_t = tf.reshape(grad_diff_t,[-1, image_size[0], image_size[1], channel_no])
  zero_fill_shape = [grad_diff_x.shape[0].value - grad_diff_t.shape[0].value, image_size[0], image_size[1], channel_no]
  zero_fill = tf.zeros(zero_fill_shape, dtype=tf.float32)
  grad_diff_t_ext = tf.concat([grad_diff_t, zero_fill], axis=0)

  gdl_loss = (tf.reduce_mean((grad_diff_t_ext**alpha + grad_diff_x ** alpha + grad_diff_y ** alpha)))
  return gdl_loss


def cgdl(gen_frames, gt_frames,  alpha, image_size, channel_no):
  """
  Arguments:
    gen_frames: Predicted frames or temporal difference of predicted frames (predicted velocity map).
    gt_frames: Ground truth predicted frames or the temporal difference of the ground truth frames (true velocity map).
    gen_frames_dt: The predicted second order temporal difference velocity map if gen_frames is predicted frames or acceleration map
                    if gen_frames is velocity map
    gt_frames_dt: The ground truth second order temporal difference velocity map if gt_frames is predicted frames or acceleration map
                    if gt_frames is velocity map
    alpha: The power to which each gradient term is raised.
    image_size: tuple for image shape (h,w)
  
  Returns: 
    The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(channel_no), dtype=tf.float32)
  neg = -1 * pos
  centre=0*pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg,centre, pos]), 0)
  # [[[1],[-1]]]
  filter_y = tf.stack([tf.expand_dims(pos, 0),tf.expand_dims(centre, 0), tf.expand_dims(neg, 0)])
  # [[[-1]],[[1]]]
  strides1 = [1, 1, 1, 1]  # stride of (1, 1)
  padding = 'SAME'

  gen_dx = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  
  

  grad_diff_x =0.5* tf.abs(gt_dx - gen_dx)
  grad_diff_y =0.5* tf.abs(gt_dy - gen_dy)
  # grad_diff_t = tf.abs(gen_frames_dt - gt_frames_dt)
  # grad_diff_t = tf.reshape(grad_diff_t,[-1, image_size[0], image_size[1], channel_no])
  # zero_fill_shape = [grad_diff_x.shape[0].value - grad_diff_t.shape[0].value, image_size[0], image_size[1], channel_no]
  # zero_fill = tf.zeros(zero_fill_shape, dtype=tf.float32)
  # grad_diff_t_ext = tf.concat([grad_diff_t, zero_fill], axis=0)

  gdl_loss = (tf.reduce_mean(( grad_diff_x ** alpha + grad_diff_y ** alpha)))
  return gdl_loss
class bgdl(Loss):
  def __init__(self,alpha=1,name="bgdl",**kwargs):
      super().__init__(name=name, **kwargs)
      self.alpha=alpha


  def call(self, gen_frames, gt_frames):
    """
      Arguments:
      gen_frames: Predicted frames or temporal difference of predicted frames (predicted velocity map).
      gt_frames: Ground truth predicted frames or the temporal difference of the ground truth frames (true velocity map).
      gen_frames_dt: The predicted second order temporal difference velocity map if gen_frames is predicted frames or acceleration map
                      if gen_frames is velocity map
      gt_frames_dt: The ground truth second order temporal difference velocity map if gt_frames is predicted frames or acceleration map
                      if gt_frames is velocity map
      alpha: The power to which each gradient term is raised.
      image_size: tuple for image shape (h,w)

    Returns: 
      The GDL loss.
    """
    _input_shape=gen_frames.shape
    image_h=_input_shape[2]
    image_w=_input_shape[3]
    channel_no=_input_shape[4]
    # create filters [-1, 1] and [[1],[-1]]
    # for diffing to the left and down respectively.
    pos = tf.constant(np.identity(channel_no), dtype=tf.float32)
    neg = -1 * pos
    # centre=0*pos
    # [-1, 1]
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
    # [[1],[-1]]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
    # [[[-1]],[[1]]]
    strides1 = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_h,image_w,channel_no]),
                                                                filter_x, strides1, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_h,image_w,channel_no]),
                                                                filter_y, strides1, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_h,image_w,channel_no]),
                                                                filter_x, strides1, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_h,image_w,channel_no]),
                                                                filter_y, strides1, padding=padding))


    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y =tf.abs(gt_dy - gen_dy)
    # grad_diff_t = tf.abs(gen_frames_dt - gt_frames_dt)
    # grad_diff_t = tf.reshape(grad_diff_t,[-1, image_size[0], image_size[1], channel_no])
    # zero_fill_shape = [grad_diff_x.shape[0].value - grad_diff_t.shape[0].value, image_size[0], image_size[1], channel_no]
    # zero_fill = tf.zeros(zero_fill_shape, dtype=tf.float32)
    # grad_diff_t_ext = tf.concat([grad_diff_t, zero_fill], axis=0)

    gdl_loss = (tf.reduce_mean(( grad_diff_x ** self.alpha + grad_diff_y ** self.alpha)))
    return gdl_loss
  
class recon_loss_l2(Loss):
  def __init__(self,name="recon_loss_l2",**kwargs):
      super().__init__(name=name, **kwargs)

  def call(self,gt,output):
     loss = tf.reduce_mean(tf.square(gt - output))
     return loss

class recon_loss_l1(Loss):
  def __init__(self,name="recon_loss_l1",**kwargs):
      super().__init__(name=name, **kwargs)
  def call(self,gt,output):
    loss = tf.reduce_mean(
                    tf.abs(gt - output))
    return loss
  


def optical_flow_loss(pred_flow, gt_flow,  alpha):
  """
  Arguments:
    gen_frames: Predicted frames or temporal difference of predicted frames (predicted velocity map).
    gt_frames: Ground truth predicted frames or the temporal difference of the ground truth frames (true velocity map).
    gen_frames_dt: The predicted second order temporal difference velocity map if gen_frames is predicted frames or acceleration map
                    if gen_frames is velocity map
    gt_frames_dt: The ground truth second order temporal difference velocity map if gt_frames is predicted frames or acceleration map
                    if gt_frames is velocity map
    alpha: The power to which each gradient term is raised.
    image_size: tuple for image shape (h,w)
  
  Returns: 
    The GDL loss.
  """
  if alpha==2:
    optical_loss = tf.reduce_mean(
                tf.square(gt_flow - pred_flow))
  else:
    optical_loss = tf.reduce_mean(
                tf.abs(gt_flow - pred_flow))
  return optical_loss




def sgdl(gen_frames, gt_frames, gen_frames_dt, gt_frames_dt, alpha, image_size, channel_no):
  """
  Arguments:
    gen_frames: Predicted frames or temporal difference of predicted frames (predicted velocity map).
    gt_frames: Ground truth predicted frames or the temporal difference of the ground truth frames (true velocity map).
    gen_frames_dt: The predicted second order temporal difference velocity map if gen_frames is predicted frames or acceleration map
                    if gen_frames is velocity map
    gt_frames_dt: The ground truth second order temporal difference velocity map if gt_frames is predicted frames or acceleration map
                    if gt_frames is velocity map
    alpha: The power to which each gradient term is raised.
    image_size: tuple for image shape (h,w)
  
  Returns: 
    The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(channel_no), dtype=tf.float32)
  neg = -1 * pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
  # [[1],[-1]]
  filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
  # [[[-1]],[[1]]]
  strides1 = [1, 1, 1, 1]  # stride of (1, 1)
  padding = 'SAME'

  gen_dx = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  
  

  grad_diff_x = tf.abs(gt_dx - gen_dx)
  grad_diff_y = tf.abs(gt_dy - gen_dy)
  gdl_loss = (tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha)))
  # grad_diff_t = tf.abs(gen_frames_dt - gt_frames_dt)
  # grad_diff_t = tf.reshape(grad_diff_t,[-1, image_size[0], image_size[1], channel_no])
  # zero_fill_shape = [grad_diff_x.shape[0].value - grad_diff_t.shape[0].value, image_size[0], image_size[1], channel_no]
  # zero_fill = tf.zeros(zero_fill_shape, dtype=tf.float32)
  # grad_diff_t_ext = tf.concat([grad_diff_t, zero_fill], axis=0)

  
  return gdl_loss