import tensorflow as tf
tf.random.set_seed(77)
import sys
# tf.compat.v1.enable_eager_execution()
# from joblib import Parallel, delayed
import time
import utils as U
import numpy as np
import loss as Ls
# from tensorflow import keras
from tensorflow.keras.metrics import Mean
# from config import Config as cfg
# import tensorflow.contrib.eager as tfe

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    if percent_done==100:
      print('[%s] %f%s\r' % (bar, percent_done, '%'))

class Train(object):
   def __init__(self,generator,actor ,optimizer_act,cfg):
    self.gen = generator
    self.act=actor
    self.cfg=cfg
    # self.epochs = self.gen.cfg.epochs
    # self.batch_size = self.gen.cfg.batch_size
    # self.checkpoint_dir = self.gen.cfg.checkpoint_dir
    # self.cfg['is_train']=train
    # self.dis_checkpoint_dir = self.gen.cfg.dis_checkpoint_dir
    self.optimizer_act=optimizer_act
    # self.gradient_loss=Ls.bgdl(name='bgdl',alpha=1)
    self.loss_L2= Ls.recon_loss_l2()
    # self.cross_entropy_loss= Ls.sigmoid_cross_entropy_with_logits()
    ##instantiate all the losses as individual loss object 
    # self.loss_dict={1: Ls.bgdl(name='bgdl',alpha=self.gen.cfg), 2: Ls.recon_loss_l1 }
    
    self.L2_norm_metric= Mean(name="L2_loss")
    # self.L_p_metric =Mean(name="L_p")
    # self.L_sgdl_metric = Mean(name="L_sgdl")
    # self.L_Gen_metric = Mean(name="L_gen")
    # self.d_loss_metric = Mean(name="d_loss")
    # self.d_loss_real_metric = Mean(name="d_loss_real")
    # self.d_loss_fake_metric = Mean(name="d_loss_fake")
   # @tf.function()
   def compute_loss(self,input):    # input is a dictionary of predict,gen_vel_map, gt,gt_vel_map,D_real,D_logits_real,D_fake, D_logits_fake )
      Loss_l2=self.loss_L2(input['predict'],input['gt'])
    #   Loss_p1=self.recon_loss(input['gt_vel_map'],input['gen_vel_map'])
    #   gdl=self.gradient_loss(input['predict'],input['gt'])
    #   vgdl=self.gradient_loss(input['gen_vel_map'], input['gt_vel_map'])
    #   d_loss_real=self.cross_entropy_loss(input['D_logits_real'], tf.ones_like(input['D_real']))
    #   d_loss_fake=self.cross_entropy_loss(input['D_logits_fake'], tf.zeros_like(input['D_fake']))
    #   L_gen=self.cross_entropy_loss(input['D_logits_fake'], tf.ones_like(input['D_fake']))
      loss={'L2': Loss_l2}
      return loss
   

   def train_step(self, input): ## input is a disctionary having input of in_seq,xt,vel_seq,action_seq, target):
      at_target=input['at_target']
      _shape=at_target.shape.as_list()
      batch_size=_shape[0]
      vel_state=tf.zeros([batch_size,self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['filters']*4])
      state=tf.zeros([batch_size,self.cfg['a_dim']])

      with tf.GradientTape() as tape0:
      
        img_feature=[]
        predict = self.gen([input['xt'], input['vel_seq'],input['action_past_seq'],vel_state], training=self.cfg['G_train']) 
        for step in range(self.cfg['future_TS']):
           en_feature=self.gen.content_enc(predict[:,step,:,:,:],training=self.cfg['G_train'])
           feature=en_feature[0]
           img_feature.append(feature)
        img_feature=tf.stack(img_feature,axis=1)
        at_predict=self.act([input['at_in_seq'],img_feature,state], training=self.cfg['A_train'])


        
        
        # gt_vel_map = target[:, 1:, ...] - target[:, :-1, ...]
        # gen_vel_map = predict[:, 1:, ...] -predict[:, :-1, ...]

        #predict,gen_vel_map, gt,gt_vel_map,D_real,D_logits_real,D_fake, D_logits_fake
        model_output_dict={'predict': at_predict,'gt':at_target  }
        loss_dict=self.compute_loss(model_output_dict)
        # L_p=loss_dict['L_p0']+loss_dict['L_p1']
        # L_sgdl = loss_dict['L_gdl'] + loss_dict['L_vgdl']
        # reconst_loss= L_p+1*L_sgdl
        # tape.watch(reconst_loss)

        
            
        self.act_vars = self.act.trainable_variables  
        tape0.watch(self.act_vars) 
        grads_act = tape0.gradient(loss_dict['L2'], self.act_vars)
        self.optimizer_act.apply_gradients(zip(grads_act, self.act_vars))
    
      self.L2_norm_metric.update_state(loss_dict['L2'])
      # self.L_p_metric.update_state(L_p)
      # self.L_sgdl_metric.update_state(L_sgdl)
      # self.L_Gen_metric.update_state(loss_dict['D_L_gen'])
      # self.d_loss_metric.update_state(d_loss)
      # self.d_loss_real_metric.update_state( loss_dict['D_L_real'])
      # self.d_loss_fake_metric.update_state( loss_dict['D_L_fake'])
      return {"distance_loss": loss_dict['L2']}
   
   def reset_metrics(self):
      self.L2_norm_metric.reset_state()
   
   def custom_loop(self,batched_dataset):
      self.gen.build(input_shape=[( None,self.cfg['image_h'], self.cfg['image_w'],self.cfg['c_dim']),
                                  (None, self.cfg['past_TS'],self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']),
                                  (None,self.cfg['past_TS'],self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['a_dim']),
                                  (None,self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['filters']*4)])
      self.act.build(input_shape=[( None,self.cfg['past_TS'],self.cfg['a_dim']),
                                  (None, self.cfg['future_TS'],self.cfg['image_h']//8, self.cfg['image_w']//8,self.cfg['filters']*4),
                                  (None,self.cfg['a_dim'])])
      self.gen.summary(expand_nested=True)
      self.act.summary(expand_nested=True)
      self.tf_enable=self.cfg['tf_enable']
      epochs = self.cfg['epochs']
      # image_h=self.cfg['image_h']
      # image_w=self.cfg['image_w']
      past_TS=self.cfg['past_TS']
      self.act.compile(optimizer=self.optimizer_act)


      checkpoint_gen = tf.train.Checkpoint(model=self.gen)
      # manager_gen = tf.train.CheckpointManager(checkpoint_gen, directory=self.cfg['checkpoint_dir_G'])
      checkpoint_path_gen=self.cfg['checkpoint_dir_G']+'/ckpt-'+self.cfg['ckpt_G']  #63'
      restore_gen=checkpoint_gen.restore(checkpoint_path_gen).expect_partial()
      
      # checkpoint_gen.restore(manager_gen.latest_checkpoint)
      # if manager_gen.latest_checkpoint:
      if restore_gen:
       print("Restored from {}".format(checkpoint_path_gen))
      else:
       raise Exception("No valid check point found")
      checkpoint_act = tf.train.Checkpoint(optimizer=self.optimizer_act, model=self.act)
      manager_act = tf.train.CheckpointManager(checkpoint_act, directory=self.cfg['checkpoint_dir_A'], max_to_keep=200)


      if self.tf_enable:
         self.train_step = tf.function(self.train_step)
      for epoch in range(epochs):
        step=0
        for data_batch in batched_dataset.as_numpy_iterator():
          outputs= U.parallel_data(data_batch,past_TS=self.cfg['past_TS'],future_TS=self.cfg['future_TS'],
                                  image_h=self.cfg['image_h'],image_w=self.cfg['image_w'] )
          seq_batch=[]
          diff_batch=[]
          action_batch=[]
          for output in outputs:
            seq_frames = output[0]
            seq_frames= np.expand_dims(seq_frames, axis=0)
            seq_batch.append(seq_frames)

            diff_frames = output[1]
            diff_frames= np.expand_dims(diff_frames, axis=0)
            diff_batch.append(diff_frames)

            action_frames = output[3]
            action_frames= np.expand_dims(action_frames, axis=0)
            action_batch.append(action_frames)
            # diff_batch = output[1]
            # accel_batch = output[2]
            # action_batch=output[3]
        #   output=np.expand_dims(output, axis=0)
          gt_frames= np.concatenate(seq_batch, axis=0)
          vel_seq= np.concatenate(diff_batch, axis=0)
          action_seq= np.concatenate(action_batch, axis=0)
          action_past_seq= action_seq[:,:self.cfg['past_TS'],:,:,:]
          at_in_seq=tf.squeeze(action_past_seq[:,:,0,0,:])
          action_future_seq= action_seq[:,self.cfg['past_TS']:,:,:,:]
          at_target=tf.squeeze(action_future_seq[:,:,0,0,:])

          xt= gt_frames[:,self.cfg['past_TS']-1,...]
        #   gt_frames=output[:]['vid']
        #   xt= gt_frames[:,self.cfg['past_TS'],...]
        #   vel_seq=output['diff']
        #   action_seq=output['actions']
          in_seq=gt_frames[:,:past_TS,:,:,:]
         #  target=gt_frames[:,past_TS:,:,:,:]
          input={'in_seq': in_seq, 'xt': xt,'vel_seq':vel_seq,
                 'action_past_seq':action_past_seq, 'at_in_seq':at_in_seq, 'at_target':at_target}
          self.train_step(input)
          progress_bar(( step+ 1) / len(batched_dataset) * 100, 60)
          
          step=step+1
        template = ('Epoch: {},  L2-norm Loss: {}')

        print(template.format(epoch, self.L2_norm_metric.result()))
        
        if epoch != self.cfg['epochs'] - 1:
          self.reset_metrics()
        if epoch%10==0:
           manager_act.save()
        if epoch%50==0:
           tf.saved_model.save(self.gen,self.cfg['checkpoint_dir_A'])

      return (self.L2_norm_metric.result().numpy())
      

          

          
         


      

