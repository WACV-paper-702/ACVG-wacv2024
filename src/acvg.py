import tensorflow as tf
tf.random.set_seed(77)
import os
from tensorflow import keras
from tensorflow import keras as K
import layers as L
from acpnet import Generator, Action_predictor

class acvg(tf.keras.Model):
    def __init__(self, cfg,cfg_gen,cfg_act,name="acvg",**kwargs):
      super().__init__(name=name,**kwargs)
      self.cfg=cfg
      self.gen=Generator(cfg_gen,name="ACVG_Generator")
      self.act=Action_predictor(cfg_act, name= "ACVG_Actor")
      
        
    # @tf.function (input_signature=([tf.TensorSpec(shape=[None,], dtype=tf.float32,name='xt')],[tf.TensorSpec(shape=[None,None,None,None,None], dtype=tf.float32,name='vel_seq')],
                                  #[tf.TensorSpec(shape=[None,None,None,None,None], dtype=tf.float32,name='action_seq')]))
    # @tf.function
    def call(self,input):
        # _shape=_action_seq.shape.as_list()
        xt=self.gen.xt(input[0])
        vel_seq=self.gen.vel_seq(input[1])
        action_seq_G=self.gen.action_seq(input[2])
        # action_seq=self.action_seq(in_action_seq)
        vel_state_values=self.gen.state(input[3])
        vel_state= [vel_state_values, vel_state_values]

        action_seq_A=self.act.action_seq(input[4])
        at_state_values=self.act.state(input[5])
        at_state= [at_state_values, at_state_values]
        for step in range(self.cfg['past_TS']):
            if step < self.cfg['past_TS']-1:
                motion_out=self.gen.motion_enc([vel_seq[:, step,:,:,:],action_seq_G[:, step,:,:,:],
                                                              vel_state],training=self.cfg['G_train'] )
                h_vel_out=motion_out[0]
                vel_state=motion_out[1]
                vel_res_in=motion_out[2]
            at_hat,at_state=self.act.action_LSTMcell(action_seq_A[:, step,:],
                                         at_state,training=self.cfg['A_train'] )
            
            # h_vel_out, vel_state, vel_res_in=self.motion_enc(vel_seq[:, step,:,:,:],action_seq[:, step,:,:,:],
            #                                                   vel_state,training=self.cfg['is_train'] )

        frame_predict = []
        at_predict=[]
        vel_in = vel_seq[:,self.cfg['past_TS']-2,:,:,:]
        for t in range(self.cfg['future_TS']):
            if t>0:
                # h_vel_out, vel_state, vel_res_in = self.motion_enc(vel_in, action_in, vel_state,training=self.cfg['is_train'])
                # h_con_state, con_res_in= self.content_enc(xt,training=self.cfg['is_train'])
                motion_out=self.gen.motion_enc([vel_in, action_in_G, vel_state],training=self.cfg['G_train'] )
                h_vel_out=motion_out[0]
                vel_state=motion_out[1]
                vel_res_in=motion_out[2]
                at_hat, at_state=self.act.action_LSTMcell(at,
                                         at_state,training=self.cfg['A_train'])
            content_out=self.gen.content_enc(xt,training=self.cfg['G_train'])
            h_con_state= content_out[0]
            con_res_in=content_out[1]
            comb_conv=self.gen.combination_layer([h_con_state, h_vel_out],training=self.cfg['G_train'])
            res_conv= self.gen.res_comb_layer([con_res_in, vel_res_in],training=self.cfg['G_train'])
            x_tilda= self.gen.decoder([xt,comb_conv,res_conv],training=self.cfg['G_train'])

            acvg_conv1=self.act.conv1(h_con_state,training=self.cfg['A_train'])
            acvg_maxpool1=self.act.max_pool1(acvg_conv1,training=self.cfg['A_train'])    
            img_ft_flat=self.act.flat1(acvg_maxpool1,training=self.cfg['A_train'])
            at_hat_aug=self.act.concat([at_hat,img_ft_flat])
            at_tilde=self.act.at_decoder(at_hat_aug,training=self.cfg['A_train'])
            at=at_tilde
            at_G=tf.expand_dims(tf.expand_dims(at,axis=1),axis=1)
            _tensor=tf.ones([1,self.cfg['image_h']//8, self.cfg['image_w']//8, self.cfg['a_dim']])
            action_in_G=tf.multiply(_tensor,at_G)


            vel_in_past= vel_in
            vel_in= x_tilda- xt
            # vel_in=vel 
            # acc_in= vel_in - vel_in_past
            xt=x_tilda
            # predict.append(tf.expand_dims(x_tilda,axis=1))
            frame_predict.append(x_tilda)
            at_predict.append(at_tilde)
            # predict.append(tf.reshape(x_tilda,[batch_size,1, self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]))
        # print('shape={}'.format(tf.stack(predict,axis=1).shape.as_list))
        return [tf.stack(frame_predict,axis=1) , tf.stack(at_predict,axis=1)]