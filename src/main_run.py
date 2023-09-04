from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from absl import app
import tensorflow as tf
tf.random.set_seed(77)
import utils as U

import config 
from acpnet import Generator, Discriminator, Action_predictor
from acvg import acvg
# import train as Tr
from os.path import exists
from os import makedirs
import json
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def run_main(argv):
  """Passes the flags to main.

  Args:
    argv: argv
  """
  del argv
  kwargs = config.flags_dict()
  main(**kwargs)


def main(epochs=2500,
      epochs_actor=1000,
      epochs_acvg=1000,
      tf_enable=True,
      buffer_size=5000,
      batch_size=8,
      image_h=64,
      image_w=64,
      model_name="ACPNet",
      lr=0.0001,
      lr_acvg=0.00000001,
      alpha=1,
      beta=0.0001,
      beta1=0.5,
      c_dim=3,
      a_dim=2,
      past_TS=5,
      future_TS=10,
      test_future_TS=20,
      gpu=0,
      filters=32,
      a_filters=32,
      df_dim=32,
      margin=0.3,
      ckpt_G=63,
      ckpt_D=63,
      ckpt_A=25,
      G_train=True,
      A_train=True,
      test_G=False,
      datapath=None):
  if not exists(datapath):
            raise Exception("Sorry, invalid datapath")
  
  # prefix = ("Roam_Full-v1_{}".format(model_name)
  #         + "_GPU_id="+str(gpu)
  #         + "_image_w="+str(image_w)
  #         + "_K="+str(past_TS)
  #         + "_T="+str(future_TS)
  #         + "_batch_size="+str(batch_size)
  #         + "_alpha="+str(alpha)
  #         + "_beta="+str(beta)
  #         + "_lr="+str(lr)
  #         +"_no_epochs="+str(epochs)+"_beta1"+str(beta1))
  
  # checkpoint_dir_G = "../models/"+prefix+"/Generator"
  # checkpoint_dir_D = "../models/"+prefix+"/Discriminator"
  # checkpoint_dir_A = "../models/"+prefix+"/Actor"
  # samples_dir = "../samples/"+prefix+"/"
  # logs_dir = "../logs/"+prefix+"/"
  # if not exists(checkpoint_dir_G):
  #     makedirs(checkpoint_dir_G)
  # if not exists(checkpoint_dir_D):
  #     makedirs(checkpoint_dir_D)
  # if not exists(checkpoint_dir_A):
  #     makedirs(checkpoint_dir_A)
  # if not exists(samples_dir):
  #     makedirs(samples_dir)
  # if not exists(logs_dir):
  #     makedirs(logs_dir)

  gen_config={ 'image_h':image_h, 'image_w':image_w,'c_dim':c_dim,'a_dim':a_dim,
              'filters':filters, 'past_TS':past_TS, 'future_TS':future_TS, 'is_train':G_train}
  
  act_config={ 'image_h':image_h, 'image_w':image_w,'c_dim':c_dim,'a_dim':a_dim,'a_filters':a_filters,
              'filters':filters, 'past_TS':past_TS, 'future_TS':future_TS, 'is_train':A_train}
  
  dis_config={'df_dim':df_dim, 'future_TS': future_TS,'past_TS':past_TS,
              'image_h':image_h, 'image_w':image_w,'c_dim':c_dim}
  
  acvg_config={'image_h':image_h, 'image_w':image_w,'c_dim':c_dim,'a_dim':a_dim, 'A_train':A_train,
              'filters':filters, 'past_TS':past_TS, 'future_TS':future_TS, 'G_train':G_train}
  
  # train_config={'tf_enable':tf_enable, 'alpha':alpha, 'beta':beta, 'epochs':epochs, 'G_train':G_train,'A_train':A_train,'filters':filters, 'ckpt_G': ckpt_G, 'ckpt_D': ckpt_D,
  #               'image_h':image_h, 'image_w':image_w, 'past_TS':past_TS, 'future_TS': future_TS, 'margin': margin,'batch_size': batch_size,'c_dim':c_dim,'a_dim':a_dim,
  #               'checkpoint_dir_G':checkpoint_dir_G,'checkpoint_dir_D':checkpoint_dir_D,'checkpoint_dir_A':checkpoint_dir_A,'samples_dir': samples_dir}
  
  # test_config={'tf_enable':tf_enable, 'G_train':G_train,'A_train':A_train,'filters':filters,'batch_size':1, 
  #               'image_h':image_h, 'image_w':image_w, 'past_TS':past_TS, 'future_TS': test_future_TS, 'model_name': 'ACVG', 'ckpt_G': ckpt_G, 'ckpt_A': ckpt_A,
  #               'checkpoint_dir_G':checkpoint_dir_G,'checkpoint_dir_D':checkpoint_dir_D,'checkpoint_dir_A':checkpoint_dir_A,'c_dim':c_dim,'a_dim':a_dim}
  with tf.device("/gpu:{}".format(gpu)):
    
    
    
    optimizer_gen= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
    optimizer_dis= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
    optimizer_act= tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
    
    if G_train==True or A_train==True:
      prefix = ("Roam_Full-v1_{}".format(model_name)
          + "_GPU_id="+str(gpu)
          + "_image_w="+str(image_w)
          + "_K="+str(past_TS)
          + "_T="+str(future_TS)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr)
          +"_no_epochs="+str(epochs)+"_beta1"+str(beta1))
  
      checkpoint_dir_G = "../models/"+prefix+"/Generator"
      checkpoint_dir_D = "../models/"+prefix+"/Discriminator"
      checkpoint_dir_A = "../models/"+prefix+"/Actor"
      samples_dir = "../samples/"+prefix+"/"
      logs_dir = "../logs/"+prefix+"/"
      if not exists(checkpoint_dir_G):
          makedirs(checkpoint_dir_G)
      if not exists(checkpoint_dir_D):
          makedirs(checkpoint_dir_D)
      if not exists(checkpoint_dir_A):
          makedirs(checkpoint_dir_A)
      if not exists(samples_dir):
          makedirs(samples_dir)
      if not exists(logs_dir):
          makedirs(logs_dir)
      train_config={'tf_enable':tf_enable, 'alpha':alpha, 'beta':beta, 'epochs':epochs, 'G_train':G_train,'A_train':A_train,'filters':filters, 'ckpt_G': ckpt_G, 'ckpt_D': ckpt_D,
                'image_h':image_h, 'image_w':image_w, 'past_TS':past_TS, 'future_TS': future_TS, 'margin': margin,'batch_size': batch_size,'c_dim':c_dim,'a_dim':a_dim,
                'checkpoint_dir_G':checkpoint_dir_G,'checkpoint_dir_D':checkpoint_dir_D,'checkpoint_dir_A':checkpoint_dir_A,'samples_dir': samples_dir}


      batched_dataset=U.load_roam_dict(datapath,buffer_size,batch_size=train_config['batch_size'])
      if G_train==True and A_train==False:
        print('Training Generator...')
        import train as Tr
        discriminator=Discriminator(dis_config, name= "ACP_Discriminator")
        generator=Generator(gen_config,name="ACP_Generator")
        train_config['ckpt_G']=ckpt_G
        train_config['ckpt_D']=ckpt_D
        train_obj=Tr.Train(generator,discriminator,optimizer_gen,optimizer_dis,train_config)
      elif G_train==False and A_train==True:
        print('Training Actor...')
        import train_actor as Tr
        generator=Generator(gen_config,name="ACP_Generator")
        actor=Action_predictor(act_config, name= "ACP_Actor")
        train_config['epochs']=epochs_actor
        train_config['ckpt_G']=ckpt_G
        train_config['ckpt_A']=ckpt_A
        train_obj=Tr.Train(generator,actor,optimizer_act,train_config)

      elif G_train==True and A_train==True:
        print('Training ACVG...')
        import train_acvg as Tr
        model_name='ACVG'
        prefix = ("Roam_Full-v1_{}".format(model_name)
        + "_GPU_id="+str(gpu)
        + "_image_w="+str(image_w)
        + "_K="+str(past_TS)
        + "_T="+str(future_TS)
        + "_batch_size="+str(batch_size)
        + "_alpha="+str(alpha)
        + "_beta="+str(beta)
        + "_lr="+str(lr)
        +"_no_epochs="+str(epochs_acvg)+"_beta1"+str(beta1))

        train_config['acvg_checkpoints_G'] = "../models/"+prefix+"/Generator"
        train_config['acvg_checkpoints_D'] = "../models/"+prefix+"/Discriminator"
        train_config['acvg_checkpoints_A'] = "../models/"+prefix+"/Actor"
        train_config['acvg_samples_dir'] = "../samples/"+prefix+"/"
        train_config['acvg_logs_dir'] = "../logs/"+prefix+"/"
        if not exists(train_config['acvg_checkpoints_G']):
            makedirs(train_config['acvg_checkpoints_G'])
        if not exists(train_config['acvg_checkpoints_D']):
            makedirs(train_config['acvg_checkpoints_D'])
        if not exists(train_config['acvg_checkpoints_A']):
            makedirs(train_config['acvg_checkpoints_A'])
        if not exists(train_config['acvg_samples_dir']):
            makedirs(train_config['acvg_samples_dir'])
        if not exists(train_config['acvg_logs_dir']):
            makedirs(train_config['acvg_logs_dir'])

        discriminator=Discriminator(dis_config, name= "ACVG_Discriminator")
        model=acvg(acvg_config,gen_config,act_config,name= "ACVG")
        train_config['epochs']=epochs_acvg
        train_config['ckpt_G']=ckpt_G
        train_config['ckpt_D']=ckpt_D
        train_config['ckpt_A']=ckpt_A
        train_obj=Tr.Train(model,discriminator,optimizer_gen,optimizer_dis,optimizer_act,train_config) 
      
      json_file = open("../models/"+prefix+"/training_config.json","w")
      train_json = json.dumps(train_config, indent=2)
      json_file.write(train_json)
      json_file.close()

      # pickle_file = open("../models/"+prefix+"/training_config.pkl","wb")
      # pickle.dump(train_config,pickle_file)
      # pickle_file.close()
      train_obj.custom_loop(batched_dataset)
      
    else:
      if test_G==True:
        import test_acpnet as Ts
        model_name='ACPNet'
        print("Testing {}....".format(model_name))
        # test_config['model_name']='ACPNet'
        # gen_config['future_TS']= test_config['future_TS']
        gen_config['future_TS']=test_future_TS
        model=Generator(gen_config,name="ACP_Generator")
      else:
        import test_acvg as Ts
        model_name='ACVG'
        print("Testing {}....".format(model_name))
        acvg_config['future_TS']=test_future_TS
        epochs=epochs_acvg
        # acvg_config['future_TS']=test_config['future_TS']
      # act_config['future_TS']=test_config['future_TS']
        model=acvg(acvg_config,gen_config,act_config)



      prefix = ("Roam_Full-v1_{}".format(model_name)
          + "_GPU_id="+str(gpu)
          + "_image_w="+str(image_w)
          + "_K="+str(past_TS)
          + "_T="+str(future_TS)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr)
          +"_no_epochs="+str(epochs)+"_beta1"+str(beta1))
  
      checkpoint_dir_G = "../models/"+prefix+"/Generator"
      checkpoint_dir_D = "../models/"+prefix+"/Discriminator"
      checkpoint_dir_A = "../models/"+prefix+"/Actor"

      test_config={'tf_enable':tf_enable, 'G_train':G_train,'A_train':A_train,'filters':filters,'batch_size':1, 
                'image_h':image_h, 'image_w':image_w, 'past_TS':past_TS, 'future_TS': test_future_TS, 'model_name': model_name, 'ckpt_G': ckpt_G, 'ckpt_A': ckpt_A,
                'checkpoint_dir_G':checkpoint_dir_G,'checkpoint_dir_D':checkpoint_dir_D,'checkpoint_dir_A':checkpoint_dir_A,'c_dim':c_dim,'a_dim':a_dim}
          


      results_prefix= ("half_fps_Roam_Full-v1_{}".format(test_config['model_name'])
          + "_GPU_id="+str(gpu)
          + "_image_w="+str(image_w)
          + "_K="+str(past_TS)
          + "_T="+str(test_future_TS)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr)
          +"_no_epochs="+str(epochs_acvg)+"_beta1"+str(beta1))
      test_config['results_prefix']=results_prefix
      batched_dataset=U.load_roam_dict(datapath,buffer_size,batch_size=test_config['batch_size'])        
      test_obj=Ts.Test(model,test_config)
      test_obj.custom_loop(batched_dataset)
      test_json = json.dumps(test_config)
      json_file = open("../results/quantitative/RoAM/"+results_prefix+"/testing_config.json","w")
      json_file.write(test_json)
      json_file.close()


    



if __name__ == '__main__':
  config.define_network_flags()
  app.run(run_main)
