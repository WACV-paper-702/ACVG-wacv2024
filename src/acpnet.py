import tensorflow as tf
tf.random.set_seed(77)
import os
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow import keras as K
# from keras import layers
# import utils
# from loss import *
import layers as L
# from config import Config
# import loss as ls
# from keras.base_conv_lstm import ConvLSTMCell
# from keras.layers.rnn.base_conv_lstm import ConvLSTMCell

IMG_H=int(os.environ.get('IMG_H'))
IMG_W=int(os.environ.get('IMG_W'))
C_DIM=int(os.environ.get('C_DIM'))
A_DIM=int(os.environ.get('A_DIM'))
PAST_TS=int(os.environ.get('PAST_TS'))
FUTURE_TS=int(os.environ.get('FUTURE_TS'))
FILTERS=int(os.environ.get('FILTERS'))
                     


        


class motion_enc(tf.keras.Model):
    def __init__(self,filters, name="motion_enc",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.filters=filters
        self.mo_conv1=L.conv2d(filters=self.filters,kernel_size=[5,5],activation="relu",name="mo_conv1",padding="same")
        self.mo_max_pool1 = L.maxpool2D(pool_size=[2, 2],strides=2,name="mo_max_pool1")
        self.mo_conv2=L.conv2d(filters=self.filters*2,kernel_size=[5,5],activation="relu",name="mo_conv2",padding="same")
        self.mo_max_pool2 =  L.maxpool2D(pool_size=[2, 2],strides=2,name="mo_max_pool2")
        self.mo_conv3=L.conv2d(filters=self.filters*4,kernel_size=[3,3],activation="relu",name="mo_conv3",padding="same")
        self.mo_max_pool3 =  L.maxpool2D(pool_size=[2, 2],strides=2,name="mo_max_pool3")
        self.motion_LSTM=L.convLSTM2Dcell(filters=self.filters*4,kernel_size=[3,3],name='convlstm',padding="same")
    # def call(self, mon_in,action_in,mon_state):
    # @tf.function  #(input_signature=[tf.TensorSpec(shape=[None,PAST_TS,IMG_H,IMG_W,C_DIM],dtype=tf.float32),
                                  # tf.TensorSpec(shape=[None,PAST_TS+FUTURE_TS,IMG_H//8,IMG_W//8,A_DIM],dtype=tf.float32),
                                  # tf.TensorSpec(shape=[None,IMG_H//8,IMG_W//8,FILTERS*4],dtype=tf.float32)])
    def call(self, input):
         mon_in=input[0]
         action_in=input[1]
         mon_state=input[2]

         mon_res_in = []
         conv1=self.mo_conv1(mon_in)
         mon_res_in.append(conv1)
         pool1=self.mo_max_pool1(conv1)

         conv2=self.mo_conv2(pool1)
         mon_res_in.append(conv2)
         pool2=self.mo_max_pool2(conv2)

         conv3=self.mo_conv3(pool2)
         mon_res_in.append(conv3)
         pool3=self.mo_max_pool3(conv3)
         pool3= tf.concat([pool3,action_in], axis=-1)
         h1_state, mon_state = self.motion_LSTM(pool3, mon_state)
         return [h1_state, mon_state, mon_res_in]
    
class action_decoder(tf.keras.Model):
    def __init__(self,a_dim,a_filters, name="action_decoder",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.a_dim=a_dim
        self.a_filters=a_filters
        self.linear1=L.dense(output_shape=self.a_filters*4, name='act_dense1')
        self.batch_norm1=L.batchNorm(name='act_bn1')
        self.linear2=L.dense(output_shape=self.a_filters*2, name='act_dense2')
        self.batch_norm2=L.batchNorm(name='act_bn2')
        self.linear3=L.dense(output_shape=self.a_filters*4, name='act_dense3')
        self.droput1=L.dropout(rate=0.2,name='act_drop1')
        self.batch_norm3=L.batchNorm(name='act_bn3')
        self.linear4=L.dense(output_shape=self.a_filters, name='act_dense4')
        self.droput2=L.dropout(rate=0.2,name='act_drop2')
        self.batch_norm4=L.batchNorm(name='act_bn4')
        self.linear5=L.dense(output_shape=self.a_dim, name='act_dense5')

    def call(self,input):
        linear1=self.linear1(input)
        batch_norm1=self.batch_norm1(linear1)
        linear2=self.linear2(batch_norm1)
        batch_norm2=self.batch_norm2(linear2)
        linear3=self.linear3(batch_norm2)
        drop1=self.droput1(linear3)
        batch_norm3=self.batch_norm3(drop1)
        linear4=self.linear4(batch_norm3)
        drop2=self.droput2(linear4)
        batch_norm4=self.batch_norm4(drop2)
        action=self.linear5(batch_norm4)
        return action

class Action_predictor(tf.keras.Model):
    def __init__(self,cfg, name="action_predictor",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.cfg=cfg
        at_shape=[ self.cfg['past_TS'], self.cfg['a_dim']]
        img_shape=[  self.cfg['image_h']//8, self.cfg['image_w']//8, self.cfg['filters']*4]
        state_shape=[  self.cfg['a_dim']]
        self.action_seq= L.input(input_shape=at_shape,batch_size=None, name="at_seq")
        self.img_seq= L.input(input_shape=img_shape,batch_size=None, name="img_seq")
        self.state= L.input(input_shape=state_shape,batch_size=None, name="state")
        self.action_LSTMcell=L.LSTMcell(units=self.cfg['a_dim'],name='lstmcell')
        self.conv1=L.conv2d(filters=self.cfg['a_filters'],kernel_size=[3,3],activation="relu",name="at_conv1",padding="same")
        self.max_pool1 = L.maxpool2D(pool_size=[2, 2],strides=2,name="at_max_pool1")
        self.flat1=L.flatten(name='flatten1')
        self.concat=keras.layers.Concatenate(axis=-1)
        self.at_decoder=action_decoder(a_dim=self.cfg['a_dim'],a_filters=self.cfg['a_filters'],
                                       name='action_decoder')

   
    # @tf.function  #(input_signature=[tf.TensorSpec(shape=[None,PAST_TS,IMG_H,IMG_W,C_DIM],dtype=tf.float32),
                                  # tf.TensorSpec(shape=[None,PAST_TS+FUTURE_TS,IMG_H//8,IMG_W//8,A_DIM],dtype=tf.float32),
                                  # tf.TensorSpec(shape=[None,IMG_H//8,IMG_W//8,FILTERS*4],dtype=tf.float32)])
    def call(self, input):
         action_seq=self.action_seq(input[0])
         image_state =self.img_seq(input[1])
         state_values=self.state(input[2])

         at_state= [state_values, state_values]
         for step in range(self.cfg['past_TS']):
            at_hat,at_state=self.action_LSTMcell(action_seq[:, step,:],
                                         at_state,training=self.cfg['is_train'] )
         at_predict=[]
         for t in range(self.cfg['future_TS']):
            if t>0:
                at_hat, at_state=self.action_LSTMcell(at,
                                         at_state,training=self.cfg['is_train'])
            conv1=self.conv1(image_state[:,t,:,:,:],training=self.cfg['is_train'])
            maxpool1=self.max_pool1(conv1,training=self.cfg['is_train'])    
            img_ft_flat=self.flat1(maxpool1,training=self.cfg['is_train'])
            at_hat_aug=self.concat([at_hat,img_ft_flat])
            at_tilde=self.at_decoder(at_hat_aug,training=self.cfg['is_train'])
            at=at_tilde
            at_predict.append(at_tilde)

         return tf.stack(at_predict,axis=1) 
    
# class content_enc(keras.layers.Layer):
class content_enc(tf.keras.Model):
    def __init__(self,filters, name="content_enc",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.filters=filters
        self.con_conv1=L.conv2d(filters=self.filters,kernel_size=[5,5],activation="relu",name="con_conv1",padding="same")
        self.con_max_pool1 = L.maxpool2D(pool_size=[2, 2],strides=2,name="con_max_pool1")
        self.con_conv2=L.conv2d(filters=self.filters*2,kernel_size=[5,5],activation="relu",name="con_conv2",padding="same")
        self.con_max_pool2 = L.maxpool2D(pool_size=[2, 2],strides=2,name="con_max_pool2")
        self.con_conv3=L.conv2d(filters=self.filters*4,kernel_size=[3,3],activation="relu",name="con_conv3",padding="same")
        self.con_max_pool3 =L.maxpool2D(pool_size=[2, 2],strides=2,name="con_max_pool3")
    # @tf.function 
    def call(self,xt):
        content_res_in = []
        conv1=self.con_conv1(xt)
        content_res_in.append(conv1)
        pool1=self.con_max_pool1(conv1)

        conv2=self.con_conv2(pool1)
        content_res_in.append(conv2)
        pool2=self.con_max_pool2(conv2)

        conv3=self.con_conv3(pool2)
        content_res_in.append(conv3)
        pool3=self.con_max_pool3(conv3)
        return [pool3, content_res_in]

# class combination_layer(keras.layers.Layer):
class combination_layer(tf.keras.Model):
    def __init__(self,filters, name="combination_layer",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.filters=filters
        self.comb_conv1=L.conv2d(filters=self.filters*4,kernel_size=[3,3],activation="relu",name="comb_conv1",padding="same")
        self.comb_conv2=L.conv2d(filters=self.filters*2,kernel_size=[3,3],activation="relu",name="comb_conv2",padding="same")
        self.comb_conv3=L.conv2d(filters=self.filters*4,kernel_size=[3,3],activation="relu",name="comb_conv3",padding="same")
    
    # @tf.function 
    def call(self,input):
        h_con_state=input[0]
        h_motion_kernal=input[1]
        motion_in= tf.concat([h_con_state, h_motion_kernal], axis= 3 )
        motion_emb1=self.comb_conv1(motion_in)
        motion_emb2=self.comb_conv2(motion_emb1)
        motion_emb3=self.comb_conv3(motion_emb2)

        return motion_emb3
    

# class res_comb_layer(keras.layers.Layer):
class res_comb_layer(tf.keras.Model):
    def __init__(self,filters, name="res_comb_layer",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.filters=filters
        self.res_cont_conv1_1=L.conv2d(filters=self.filters,kernel_size=[3,3],activation="relu",name="res_cont_conv1_1",padding="same")
        self.res_cont_conv1_2=L.conv2d(filters=self.filters,kernel_size=[3,3],activation="relu",name="res_cont_conv1_2",padding="same")

        self.res_cont_conv2_1=L.conv2d(filters=self.filters*2,kernel_size=[3,3],activation="relu",name="res_cont_conv2_1",padding="same")
        self.res_cont_conv2_2=L.conv2d(filters=self.filters*2,kernel_size=[3,3],activation="relu",name="res_cont_conv2_2",padding="same")

        self.res_cont_conv3_1=L.conv2d(filters=self.filters*4,kernel_size=[3,3],activation="relu",name="res_cont_conv3_1",padding="same")
        self.res_cont_conv3_2=L.conv2d(filters=self.filters*4,kernel_size=[3,3],activation="relu",name="res_cont_conv3_2",padding="same")

    # @tf.function 
    def call(self,input):
        con_res_in=input[0]
        mon_res_in=input[1]
        res_conv_out=[]
        res_motion_in1= tf.concat([con_res_in[0], mon_res_in[0]], axis= 3 )
        image_res_emb1_1=self.res_cont_conv1_1(res_motion_in1)
        image_res_emb1_2=self.res_cont_conv1_2(image_res_emb1_1)
        res_conv_out.append(image_res_emb1_2)

        res_motion_in2= tf.concat([con_res_in[1], mon_res_in[1]], axis= 3 )
        image_res_emb2_1=self.res_cont_conv2_1(res_motion_in2)
        image_res_emb2_2=self.res_cont_conv2_2(image_res_emb2_1)
        res_conv_out.append(image_res_emb2_2)

        res_motion_in3= tf.concat([con_res_in[2], mon_res_in[2]], axis= 3 )
        image_res_emb3_1=self.res_cont_conv3_1(res_motion_in3)
        image_res_emb3_2=self.res_cont_conv3_2(image_res_emb3_1)
        res_conv_out.append(image_res_emb3_2)

        return res_conv_out
         

class decoder(keras.Model):
    def __init__(self,filters,c_dim=3, name="decoder",**kwargs):
        super().__init__(name=name,  **kwargs )
        self.filters=filters
        # self.batch_size=batch_size
        self.c_dim=c_dim
        self.up_samp1=L.upsampling2D(size=[2,2])
        self.deconv1=L.deconv2d(filters=self.filters*4,
                                 kernel_size=[3,3], name='dec_deconv1', activation='relu')
        self.up_samp2=L.upsampling2D(size=[2,2])
        self.deconv2=L.deconv2d(filters=self.filters*2,
                                 kernel_size=[3,3], name='dec_deconv2', activation='relu')
        self.up_samp3=L.upsampling2D(size=[2,2])
        self.deconv3=L.deconv2d(filters=self.filters,
                                 kernel_size=[3,3], name='dec_deconv3', activation='relu')
        self.deconv4=L.deconv2d(filters=self.filters,
                                 kernel_size=[3,3], name='dec_deconv4', activation='relu')
        self.deconv5=L.deconv2d(filters=self.c_dim,
                                 kernel_size=[1,1], name='dec_deconv5', activation='tanh')
    # @tf.function 
    def call(self,input):
        xt=input[0]
        cont_conv=input[1]
        res_conv=input[2]
        # shape1 = [self.batch_size, self.image_size[0]//4,
        #                                 self.image_size[1]//4, self.filters*4]
        up_samp1 = self.up_samp1(cont_conv)
        deconv1=self.deconv1(up_samp1)
        decode1 = tf.math.add(deconv1, res_conv[2])

        up_samp2 = self.up_samp2(decode1)
        deconv2=self.deconv2(up_samp2)
        decode2 = tf.math.add(deconv2, res_conv[1])

        up_samp3 = self.up_samp3(decode2)
        deconv3=self.deconv3(up_samp3)
        decode3 = tf.math.add(deconv3, res_conv[0])

        deconv4=self.deconv4(decode3)
        decode4 = tf.concat(axis=3, values=[deconv4, xt])
        decode_out=self.deconv5(decode4)

        return decode_out
    
class Discriminator(tf.keras.Model):
    def __init__(self, cfg,name="Discriminator",**kwargs):
      super().__init__(name=name,  **kwargs)
      self.df_dim=cfg['df_dim']
      self.future_timestep=cfg['future_TS']
      input_shape = [ cfg['image_h'], cfg['image_w'],cfg['past_TS']* cfg['c_dim']]
      target_shape = [cfg['image_h'], cfg['image_w'],cfg['future_TS']* cfg['c_dim']]
      self.in_seq= L.input(input_shape=input_shape,batch_size=None, name="in_seq")
      self.target_seq= L.input(input_shape=target_shape,batch_size=None, name="target_seq")
      self.concat=keras.layers.Concatenate(axis=-1)
      self.conv1=L.conv2d(filters=self.df_dim, kernel_size=[5,5],name='dis_conv1', activation='leaky_relu') #,
                        #   input_shape=[cfg['image_h'], cfg['image_w'],(cfg['future_TS']+cfg['past_TS'])* cfg['c_dim']]) 
      self.batch_norm1=L.batchNorm(name='dis_bn1')
      self.maxpool1=L.maxpool2D(pool_size=[2,2],strides=2, name='dis_maxpool1') 
      self.conv2=L.conv2d(filters=self.df_dim*2, kernel_size=[5,5],name='dis_conv2', activation='leaky_relu') 
      self.batch_norm2=L.batchNorm(name='dis_bn2')
      self.maxpool2=L.maxpool2D(pool_size=[2,2],strides=2,name='dis_maxpool2') 
      self.conv3=L.conv2d(filters=self.df_dim, kernel_size=[5,5],name='dis_conv3', activation='leaky_relu') 
      self.batch_norm3=L.batchNorm(name='dis_bn3')
      self.maxpool3=L.maxpool2D(pool_size=[2,2],strides=2,name='dis_maxpool3') 

      self.conv4=L.conv2d(filters=self.df_dim, kernel_size=[3,3],name='dis_conv4', activation='leaky_relu') 
      self.batch_norm4=L.batchNorm(name='dis_bn4')
      self.maxpool4=L.maxpool2D(pool_size=[2,2],strides=2,name='dis_maxpool4') 
      self.linear1=L.dense(output_shape=self.future_timestep, name='dis_dense1')

    # @tf.function (input_signature=[[tf.TensorSpec(shape=[None,None,None,None, None], dtype=tf.float32,name='Dis_input')],[tf.TensorSpec(shape=[None,None,None,None, None], dtype=tf.float32,name='Dis_target')]] )
    # def call(self,input_d, target_d):
    #     input=self.in_seq(input_d)
    #     target=self.target_seq(target_d)
    # @tf.function
    def call(self,input_d):
        # input=self.in_seq(input_d[0])
        # target=self.target_seq(input_d[1])
        _Dis_in_img=self.in_seq(input_d[0])
        _Dis_target_img=self.target_seq(input_d[1])
        image_array=self.concat([_Dis_in_img,_Dis_target_img])
        dis_conv1=self.conv1(image_array)
        dis_batch_norm1=self.batch_norm1(dis_conv1)
        dis_maxpool1=self.maxpool1(dis_batch_norm1)

        dis_conv2=self.conv2(dis_maxpool1)
        dis_batch_norm2=self.batch_norm2(dis_conv2)
        dis_maxpool2=self.maxpool2(dis_batch_norm2)

        dis_conv3=self.conv3(dis_maxpool2)
        dis_batch_norm3=self.batch_norm3(dis_conv3)
        dis_maxpool3=self.maxpool3(dis_batch_norm3)

        dis_conv4=self.conv4(dis_maxpool3)
        dis_batch_norm4=self.batch_norm4(dis_conv4)
        dis_maxpool4=self.maxpool4(dis_batch_norm4)

        output= self.linear1(dis_maxpool4)
        return tf.nn.sigmoid(output), output
    
class Generator(tf.keras.Model):
    def __init__(self, cfg,name="Generator",**kwargs):
      super().__init__(name=name,**kwargs)
      self.cfg=cfg
      xt_shape = [self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]
      vel_shape = [self.cfg['past_TS']-1, self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]
      at_shape=[ self.cfg['past_TS'], self.cfg['image_h']//8, self.cfg['image_w']//8, self.cfg['a_dim']]
      state_shape=[  self.cfg['image_h']//8, self.cfg['image_w']//8, self.cfg['filters']*4]
    #   self.target_shape = [self.cfg['future_TS'], self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]
    #   self.in_seq= L.input(input_shape=self.input_shape,batch_size=None, name="in_seq")
      self.xt= L.input(input_shape=xt_shape,batch_size=None, name="xt")
      self.vel_seq= L.input(input_shape=vel_shape,batch_size=None, name="vel_seq")
      self.action_seq= L.input(input_shape=at_shape,batch_size=None, name="at_seq")
      self.state= L.input(input_shape=state_shape,batch_size=None, name="state")
      self.motion_enc=motion_enc(filters=self.cfg['filters'], name="motion_enc")
      self.content_enc=content_enc(filters=self.cfg['filters'], name="content_enc")
      self.combination_layer=combination_layer(filters=self.cfg['filters'], name = "combination_layer")
      self.res_comb_layer=res_comb_layer(filters=self.cfg['filters'], name = "res_comb_layer")
      self.decoder=decoder(filters=self.cfg['filters'],  
                            c_dim=self.cfg['c_dim'], name= "decoder")
        
    # @tf.function (input_signature=([tf.TensorSpec(shape=[None,], dtype=tf.float32,name='xt')],[tf.TensorSpec(shape=[None,None,None,None,None], dtype=tf.float32,name='vel_seq')],
                                  #[tf.TensorSpec(shape=[None,None,None,None,None], dtype=tf.float32,name='action_seq')]))
    # @tf.function
    def call(self,input_g):
        # _shape=_action_seq.shape.as_list()
        xt=self.xt(input_g[0])
        vel_seq=self.vel_seq(input_g[1])
        action_seq=self.action_seq(input_g[2])
        # action_seq=self.action_seq(in_action_seq)
        state_values=self.state(input_g[3])
        # state_initializer = keras.initializers.Zeros()
        # state_values=state_initializer(shape=[self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['filters']*4])
        # batch_size=_shape[0]
        # _state_=tf.zeros([batch_size,self.cfg['image_h']// 8, self.cfg['image_w'] // 8,self.cfg['filters']*4])
        # vel_state= [_state_, _state_]
        vel_state= [state_values, state_values]
        # vel_state= [_vel_state0, _vel_state0]
        for step in range(self.cfg['past_TS']-1):
            motion_out=self.motion_enc([vel_seq[:, step,:,:,:],action_seq[:, step,:,:,:],
                                                              vel_state],training=self.cfg['is_train'] )
            h_vel_out=motion_out[0]
            vel_state=motion_out[1]
            vel_res_in=motion_out[2]
            # h_vel_out, vel_state, vel_res_in=self.motion_enc(vel_seq[:, step,:,:,:],action_seq[:, step,:,:,:],
            #                                                   vel_state,training=self.cfg['is_train'] )

        predict = []
        for t in range(self.cfg['future_TS']):
            if t==0:
                content_out=self.content_enc(xt,training=self.cfg['is_train'])
                h_con_state= content_out[0]
                con_res_in=content_out[1]
                # h_con_state, con_res_in= self.content_enc(xt,training=self.cfg['is_train'])
                comb_conv=self.combination_layer([h_con_state, h_vel_out],training=self.cfg['is_train'])
                res_conv= self.res_comb_layer([con_res_in, vel_res_in],training=self.cfg['is_train'])
                x_tilda= self.decoder([xt,comb_conv,res_conv],training=self.cfg['is_train'])
                vel_in = vel_seq[:,self.cfg['past_TS']-2,:,:,:]
                action_in=action_seq[:,self.cfg['past_TS']-1,:,:,:]
            else:
                # h_vel_out, vel_state, vel_res_in = self.motion_enc(vel_in, action_in, vel_state,training=self.cfg['is_train'])
                # h_con_state, con_res_in= self.content_enc(xt,training=self.cfg['is_train'])
                motion_out=self.motion_enc([vel_in, action_in, vel_state],training=self.cfg['is_train'] )
                h_vel_out=motion_out[0]
                vel_state=motion_out[1]
                vel_res_in=motion_out[2]
                content_out=self.content_enc(xt,training=self.cfg['is_train'])
                h_con_state= content_out[0]
                con_res_in=content_out[1]
                comb_conv=self.combination_layer([h_con_state, h_vel_out],training=self.cfg['is_train'])
                res_conv= self.res_comb_layer([con_res_in, vel_res_in],training=self.cfg['is_train'])
                x_tilda= self.decoder([xt,comb_conv,res_conv],training=self.cfg['is_train'])
            vel_in_past= vel_in
            vel_in= x_tilda- xt
            # vel_in=vel 
            # acc_in= vel_in - vel_in_past
            xt=x_tilda
            # predict.append(tf.expand_dims(x_tilda,axis=1))
            predict.append(x_tilda)
            # predict.append(tf.reshape(x_tilda,[batch_size,1, self.cfg['image_h'], self.cfg['image_w'], self.cfg['c_dim']]))
        # print('shape={}'.format(tf.stack(predict,axis=1).shape.as_list))
        return tf.stack(predict,axis=1)  ##tf.stack might be helpful

#####################################################################################################    

