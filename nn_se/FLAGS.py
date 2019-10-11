class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'val'
  MODEL_INFER_KEY = 'infer'

  # dataset name
  train_name="train"
  validation_name="validation"
  test_name="test"

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/speech_enhancement_2019_exp/'
  '''
  # dir to store log, model and results files:
  $root_dir/datasets: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/test_records: test results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"

  # _1_preprocess param
  n_train_set_records = 72000
  n_val_set_records = 7200
  n_test_set_records = 3600
  train_val_snr = [-5, 15]
  train_val_wav_seconds = 3.0

  sampling_rate = 8000

  n_processor_gen_tfrecords = 16
  tfrecords_num_pre_set = 160
  shuffle_records = True
  batch_size = 64
  n_processor_tfdata = 4

  """
  @param model_name:
  CNN_RNN_REAL_MASK_MODEL, COMPLEX_CNN_COMPLEX_MASK_MODEL, REAL_CNN_COMPLEX_MASK_MODEL
  """
  model_name = "CNN_RNN_REAL_MASK_MODEL"

  """
  @param loss_name:
  real_net_mag_mse, real_net_spec_mse, real_net_wav_L1, real_net_wav_L2
  """
  loss_name = ["real_net_mag_mse"]
  frame_length = 256
  frame_step = 64
  no_cnn = False
  blstm_layers = 2
  fft_dot = 129
  max_keep_ckpt = 30
  learning_rate = 0.001
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  s_epoch = 1
  max_epoch = 100
  batches_to_logging = 300

  max_model_abandon_time = 3
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 4000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)


class debug(BaseConfig):
  blstm_layers = 2
  no_cnn = True

class nn_se_lr0001(BaseConfig): # running 15123
  """
  cnn2blstm
  """
  learning_rate = 0.0001

class nn_se_lr0003(BaseConfig): # running 15123
  """
  cnn2blstm
  """
  learning_rate = 0.0003

class nn_se_lr001(BaseConfig): # pendding 15123
  """
  cnn2blstm
  """
  learning_rate = 0.001

class nn_se_lr003(BaseConfig): # pendding 15123
  """
  cnn2lstm
  """
  learning_rate = 0.003


# add loss weight

"""####################################################################################
#######################################################################################
####################################################################################"""

class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.225
  root_dir = '/home/zhangwenbo5/lihongfeng/speech_enhancement_2019_exp'

class p40_nn_se_cnn1lstm(p40): # runnning p40
  """
  cnn1lstm, same to "nn_se_warmup"
  """
  pass

class p40_nn_se_cnn2lstm(p40): # running p40
  """
  cnn2lstm
  """
  blstm_layers = 2

class p40_nn_se_2lstmonly(p40): # done p40
  """
  2lstm only
  """
  blstm_layers = 2
  no_cnn = True

class p40_nn_se_cnnonly(p40): # running p40
  """
  cnn only
  """
  blstm_layers = 0

PARAM = nn_se_lr0001
