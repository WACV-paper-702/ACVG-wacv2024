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

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    if percent_done==100:
      print('[%s] %f%s\r' % (bar, percent_done, '%'))


class Train(object):
   def __init__(self, acvg, discriminator, optimizer_gen, optimizer_dis, optimizer_act, cfg):
    self.model = acvg
    self.dis=discriminator
    self.cfg=cfg
    # self.epochs = self.gen.cfg.epochs
    # self.batch_size = self.gen.cfg.batch_size
    # self.checkpoint_dir = self.gen.cfg.checkpoint_dir
    # self.cfg['G_train']=train
    # self.dis_checkpoint_dir = self.gen.cfg.dis_checkpoint_dir
    self.optimizer_gen=optimizer_gen
    self.optimizer_dis=optimizer_dis
    self.optimizer_act=optimizer_act
    self.gradient_loss=Ls.bgdl(name='bgdl',alpha=1)
    self.recon_loss= Ls.recon_loss_l1()
    self.cross_entropy_loss= Ls.sigmoid_cross_entropy_with_logits()
    self.loss_L2_at= Ls.recon_loss_l2()
    ##instantiate all the losses as individual loss object 
    # self.loss_dict={1: Ls.bgdl(name='bgdl',alpha=self.gen.cfg), 2: Ls.recon_loss_l1 }
    
    self.L_recon_metric= Mean(name="reconst_loss")
    self.L_p_metric =Mean(name="L_p")
    self.L_sgdl_metric = Mean(name="L_sgdl")
    self.L_Gen_metric = Mean(name="L_gen")
    self.d_loss_metric = Mean(name="d_loss")
    self.d_loss_real_metric = Mean(name="d_loss_real")
    self.d_loss_fake_metric = Mean(name="d_loss_fake")
    self.L2_norm_metric_at= Mean(name="L2_loss_at")
   # @tf.function()
   def compute_loss(self,input):    # input is a dictionary of predict,gen_vel_map, gt,gt_vel_map,D_real,D_logits_real,D_fake, D_logits_fake )
      Loss_p0=self.recon_loss(input['frame_predict'],input['frame_gt'])
      Loss_p1=self.recon_loss(input['gt_vel_map'],input['gen_vel_map'])
      gdl=self.gradient_loss(input['frame_predict'],input['frame_gt'])
      vgdl=self.gradient_loss(input['gen_vel_map'], input['gt_vel_map'])
      d_loss_real=self.cross_entropy_loss(input['D_logits_real'], tf.ones_like(input['D_real']))
      d_loss_fake=self.cross_entropy_loss(input['D_logits_fake'], tf.zeros_like(input['D_fake']))
      L_gen=self.cross_entropy_loss(input['D_logits_fake'], tf.ones_like(input['D_fake']))
      Loss_l2_at=self.loss_L2_at(input['at_predict'],input['at_gt'])
      loss={'L_p0': Loss_p0, 'L_p1': Loss_p1, 'L_gdl': gdl, 'L_vgdl': vgdl,
            'D_L_real': d_loss_real, 'D_L_fake':d_loss_fake, 'D_L_gen': L_gen,
            'L2_at': Loss_l2_at}
      return loss
   
   def train_step(self, input): ## input is a disctionary having input of in_seq,xt,vel_seq,action_seq, target):
      target=input['target']
      at_target=input['at_target']
      _shape=target.shape.as_list()
      batch_size=_shape[0]
      vel_state=tf.zeros([batch_size,self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['filters']*4])
      at_state=tf.zeros([batch_size,self.cfg['a_dim']])


      with tf.GradientTape() as tape0, tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        predict = self.model([input['xt'], input['vel_seq'],input['action_past_seq'],vel_state,
                            input['at_in_seq'],at_state], training=True)
        frames_predict=predict[0]
        at_predict=predict[1]

        input_Dis=tf.reshape(tf.transpose(input['in_seq'], [0,2,3,1,4]), 
                             shape=[batch_size,self.cfg['image_h'],self.cfg['image_w'],self.cfg['past_TS']*self.cfg['c_dim']])
        target_Dis=tf.reshape(tf.transpose(input['target'], [0,2,3,1,4]), 
                             shape=[batch_size,self.cfg['image_h'],self.cfg['image_w'],self.cfg['future_TS']*self.cfg['c_dim']])
        predict_Dis=tf.reshape(tf.transpose(frames_predict, [0,2,3,1,4]), 
                             shape=[batch_size,self.cfg['image_h'],self.cfg['image_w'],self.cfg['future_TS']*self.cfg['c_dim']])
        
        D_real_, D_logits_real_= self.dis([input_Dis,target_Dis], training=self.cfg['G_train']) 
        D_fake_, D_logits_fake_ = self.dis([input_Dis,predict_Dis], training=self.cfg['G_train'])
      

        # D_real_, D_logits_real_= self.dis([tf.transpose(input['in_seq'], [0,2,3,1,4]),
        #                                    tf.transpose(input['target'], [0,2,3,1,4])], training=self.cfg['G_train']) 
        # D_fake_, D_logits_fake_ = self.dis([tf.transpose(input['in_seq'], [0,2,3,1,4]),
        #                                     tf.transpose(predict, [0,2,3,1,4])], training=self.cfg['G_train'])

        D_real= tf.concat(axis=1, values=D_real_)
        D_logits_real= tf.concat(axis=1, values= D_logits_real_)
        D_fake= tf.concat(axis=1, values= D_fake_)
        D_logits_fake= tf.concat(axis=1, values= D_logits_fake_)
        
        gt_vel_map = target[:, 1:, ...] - target[:, :-1, ...]
        gen_vel_map = frames_predict[:, 1:, ...] -frames_predict[:, :-1, ...]

        model_output_dict={'frame_predict': frames_predict,'gen_vel_map': gen_vel_map,'frame_gt':target ,
                  'gt_vel_map':gt_vel_map,'D_real':D_real, 'D_logits_real':D_logits_real,
                      'D_fake':D_fake, 'D_logits_fake':D_logits_fake,
                          'at_predict': at_predict,'at_gt':at_target}
        loss_dict=self.compute_loss(model_output_dict)
        L_p=loss_dict['L_p0']+loss_dict['L_p1']
        L_sgdl = loss_dict['L_gdl'] + loss_dict['L_vgdl']
        reconst_loss= L_p+1*L_sgdl
        ################# Generative and adversarial losses
        d_loss= loss_dict['D_L_real']+loss_dict['D_L_fake']
        updateD = True
        updateG = True
        if  loss_dict['D_L_fake'] < self.cfg['margin'] or loss_dict['D_L_real'] < self.cfg['margin']:
            updateD = False
            # print("Not! updating Discriminator")
        if loss_dict['D_L_fake'] > (1.-self.cfg['margin']) or loss_dict['D_L_real'] > (1.-self.cfg['margin']):
            updateG = False
            # print("Not! updating generator")
        if not updateD and not updateG:
            updateD = True
            updateG = True
            
        self.gen_vars = self.model.gen.trainable_variables  
        tape0.watch(self.gen_vars) 
        self.dis_vars = self.dis.trainable_variables 
        tape1.watch(self.dis_vars)
        self.act_vars = self.model.act.trainable_variables  
        tape2.watch(self.act_vars) 

        if updateG:
              grads_gen = tape0.gradient(self.cfg['alpha']*reconst_loss+
                                        self.cfg['beta']*loss_dict['D_L_gen'], self.gen_vars)
              self.optimizer_gen.apply_gradients(zip(grads_gen, self.gen_vars))
              grads_act = tape2.gradient(loss_dict['L2_at'], self.act_vars)
              self.optimizer_act.apply_gradients(zip(grads_act, self.act_vars))
        if updateD: 
              grads_dis = tape1.gradient(d_loss, self.dis_vars)
              self.optimizer_dis.apply_gradients(zip(grads_dis, self.dis_vars))
        
      self.L_recon_metric.update_state(reconst_loss)
      self.L_p_metric.update_state(L_p)
      self.L_sgdl_metric.update_state(L_sgdl)
      self.L_Gen_metric.update_state(loss_dict['D_L_gen'])
      self.d_loss_metric.update_state(d_loss)
      self.d_loss_real_metric.update_state( loss_dict['D_L_real'])
      self.d_loss_fake_metric.update_state( loss_dict['D_L_fake'])
      self.L2_norm_metric_at.update_state(loss_dict['L2_at'])
      # return {"recon_loss": reconst_loss, "Generative_loss": loss_dict['D_L_gen'],
      #          "Dis_loss": d_loss, "Action loss": loss_dict['L2_at']}
      return frames_predict




   def reset_metrics(self):
      self.L_recon_metric.reset_state()
      self.L_p_metric.reset_state()
      self.L_sgdl_metric.reset_state()
      self.L_Gen_metric.reset_state()
      self.d_loss_metric.reset_state()
      self.d_loss_real_metric.reset_state()
      self.d_loss_fake_metric.reset_state()
      self.L2_norm_metric_at.reset_state()
   
   def custom_loop(self,batched_dataset):
      self.model.build(input_shape=[( None,self.cfg['image_h'], self.cfg['image_w'],self.cfg['c_dim']),
                                  (None, self.cfg['past_TS'],self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']),
                                  (None,self.cfg['past_TS'],self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['a_dim']),
                                  (None,self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['filters']*4),
                                  ( None,self.cfg['past_TS'],self.cfg['a_dim']),
                                  (None,self.cfg['a_dim'])])
      self.dis.build(input_shape=[( None,self.cfg['image_h'], self.cfg['image_w'],self.cfg['past_TS']* self.cfg['c_dim']),
                                  (None, self.cfg['image_h'], self.cfg['image_w'], self.cfg['future_TS']*self.cfg['c_dim'])])
      self.model.summary(expand_nested=True)
      self.dis.summary(expand_nested=True)
      self.tf_enable=self.cfg['tf_enable']
      epochs = self.cfg['epochs']
      past_TS=self.cfg['past_TS']
      self.model.gen.compile(optimizer=self.optimizer_gen)
      self.dis.compile(optimizer=self.optimizer_dis)
      self.model.act.compile(optimizer=self.optimizer_act)

      checkpoint_gen = tf.train.Checkpoint(optimizer=self.optimizer_gen, model=self.model.gen)
      manager_gen = tf.train.CheckpointManager(checkpoint_gen, directory=self.cfg['acvg_checkpoints_G'], max_to_keep=250)
      checkpoint_path_gen=self.cfg['checkpoint_dir_G']+'/ckpt-'+self.cfg['ckpt_G']  #63'
      restore_gen=checkpoint_gen.restore(checkpoint_path_gen).expect_partial()
      # checkpoint_gen.restore(manager_gen.latest_checkpoint).expect_partial()
      if restore_gen:
       print("Restored from {}".format(checkpoint_path_gen))
      # if manager_gen.latest_checkpoint:
      #  print("Restored from {}".format(manager_gen.latest_checkpoint))
      else:
       raise Exception("No valid check point for generator found")
      
      checkpoint_act = tf.train.Checkpoint(optimizer=self.optimizer_act, model=self.model.act)
      manager_act = tf.train.CheckpointManager(checkpoint_act, directory=self.cfg['acvg_checkpoints_A'], max_to_keep=250)
      checkpoint_path_act=self.cfg['checkpoint_dir_A']+'/ckpt-'+self.cfg['ckpt_A'] #25'
      restore_act=checkpoint_act.restore(checkpoint_path_act).expect_partial()
      # checkpoint_act.restore(manager_act.latest_checkpoint).expect_partial()
      # restore_act=checkpoint_act.restore(checkpoint_path_act).expect_partial()
      # if restore_act:
      if restore_act:
       print("Restored from {}".format(checkpoint_path_act))
      # if checkpoint_act.restore(manager_act.latest_checkpoint):
      # #  print("Restored from {}".format(checkpoint_path_act))
      #  print("Restored from {}".format(manager_act.latest_checkpoint))
      else:
       raise Exception("No valid check point for actor found")
      
      checkpoint_dis = tf.train.Checkpoint(optimizer=self.optimizer_dis, model=self.dis)
      manager_dis = tf.train.CheckpointManager(checkpoint_dis, directory=self.cfg['acvg_checkpoints_D'], max_to_keep=250)
      checkpoint_path_dis=self.cfg['checkpoint_dir_D']+'/ckpt-'+self.cfg['ckpt_D'] #25'
      restore_dis=checkpoint_dis.restore(checkpoint_path_dis).expect_partial()
      if restore_dis:
       print("Restored from {}".format(checkpoint_path_dis))
      # checkpoint_dis.restore(manager_dis.latest_checkpoint).expect_partial()
      # if manager_dis.latest_checkpoint:
      #  print("Restored from {}".format(manager_dis.latest_checkpoint))
      else:
       raise Exception("No valid check point for discriminator found")
      

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
          target=gt_frames[:,past_TS:,:,:,:]
          input={'in_seq': in_seq, 'xt': xt,'vel_seq':vel_seq,
                 'action_past_seq':action_past_seq, 'target':target,
                 'at_in_seq':at_in_seq, 'at_target':at_target}
          predict=self.train_step(input)
          progress_bar(( step+ 1) / len(batched_dataset) * 100, 60)
          
          step=step+1
        template = ('Epoch: {}, Reconstruction Loss: {}, L-norm Loss: {}, Gradient Loss: {}, '
                'Generative Loss: {}, Discriminator loss for fake: {}, Discriminator loss for real: {}'
                'L2-norm Loss for at: {}')

        print(template.format(epoch, self.L_recon_metric.result(),
                          self.L_p_metric.result(),
                          self.L_sgdl_metric.result(),
                          self.L_Gen_metric.result(),
                          self.d_loss_fake_metric.result(),
                          self.d_loss_real_metric.result(),
                          self.L2_norm_metric_at.result()))
        
        if epoch != self.cfg['epochs'] - 1:
          self.reset_metrics()
        if epoch%10==0:
           manager_gen.save()
           manager_act.save()
           manager_dis.save()
           samples = predict[0,:,:,:,:]
           sbatch = target[0,:,:, :, :]
           samples = np.concatenate((samples, sbatch), axis=0)
           print("Saving sample ...")
           U.save_images(samples[:, :, :, ::-1], [2, self.cfg['future_TS']],
                      self.cfg['acvg_samples_dir']+"train_%s.png" % (epoch))
        if epoch%500==0:
           tf.saved_model.save(self.model.gen,self.cfg['acvg_checkpoints_G'])
           tf.saved_model.save(self.dis, self.cfg['acvg_checkpoints_D'])
           tf.saved_model.save(self.model.act,self.cfg['acvg_checkpoints_A'])
      return (self.L_recon_metric.result().numpy(),
                          self.L_p_metric.result().numpy(),
                          self.L_sgdl_metric.result().numpy(),
                          self.L_Gen_metric.result().numpy(),
                          self.d_loss_fake_metric.result().numpy(),
                          self.d_loss_real_metric.result().numpy(),
                          self.L2_norm_metric_at.result().numpy())

      
    