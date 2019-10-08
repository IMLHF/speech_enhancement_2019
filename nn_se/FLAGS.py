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
  $root_dir/exp/$config_name/decode: decode results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"

  n_train_set_records = 72000
  n_val_set_records = 7200
  n_test_set_records = 3600
  train_val_snr = [-5, 15]
  train_val_wav_seconds = 5.0

  sampling_rate = 16000

  n_processor_gen_tfrecords = 16
  tfrecords_num_pre_set = 160
  shuffle_records = True
  batch_size = 64
  n_processor_tfdata = 4

  model_name = "CNN_RNN_REAL_MASK_MODEL"
  loss_name = "mag_mse"
  frame_length = 512
  frame_step = 256
  blstm_layers = 1
  fft_dot = 257
  max_keep_ckpt = 30
  learning_rate = 0.0001
  use_lr_warmup = False
  warmup_steps = 4000
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True

  s_epoch = 1
  max_epoch = 100
  batches_to_logging = 300
  
  max_model_abandon_time = 3
  use_lr_warmup = False # true: lr warmup; false: lr halving
  warmup_step = 4000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)
  





PARAM = BaseConfig
