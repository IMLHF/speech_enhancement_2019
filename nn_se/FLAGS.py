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
  real_net_mag_mse, real_net_spec_mse, real_net_wav_L1, real_net_wav_L2,
  real_net_reMagMse, real_net_reSpecMse, real_net_reWavL2,
  real_net_sdrV1, real_net_sdrV2, real_net_sdrV3, real_net_stSDRV3, real_net_cosSimV1, real_net_cosSimV1WT10, real_net_cosSimV2,
  real_net_specTCosSimV1, real_net_specFCosSimV1, real_net_specTFCosSimV1,
  """
  relative_loss_AFD = 1000.0
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128
  sdrv3_bias = None # float, a bias will be added before vector dot multiply.
  loss_name = ["real_net_mag_mse"]
  loss_weight = []
  use_wav_as_feature = False
  net_out_mask = True
  frame_length = 256
  frame_step = 64
  no_cnn = False
  blstm_layers = 2
  lstm_layers = 0
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
  no_stop = False
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 4000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)


class debug(BaseConfig):
  blstm_layers = 2
  no_cnn = True

class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.225
  root_dir = '/home/zhangwenbo5/lihongfeng/speech_enhancement_2019_exp'

class nn_se_lr0001(BaseConfig): # done 15123
  """
  cnn2blstm
  """
  learning_rate = 0.0001

class nn_se_lr0003(BaseConfig): # done 15123
  """
  cnn2blstm
  """
  learning_rate = 0.0003

class nn_se_lr001(BaseConfig): # done 15123
  """
  cnn2blstm
  """
  learning_rate = 0.001

class nn_se_lr003(p40): # done p40
  """
  cnn2blstm
  """
  learning_rate = 0.003

class nn_se_only2blstm(BaseConfig): # done 15123
  """
  only2blstm
  """
  no_cnn = True

class nn_se_only1blstm(BaseConfig): # done 15123
  """
  only1blstm
  """
  no_cnn = True
  blstm_layers = 1

class nn_se_cnn1blstm(p40): # done p40
  """
  cnn1blstm
  """
  blstm_layers = 1

class nn_se_cnn1blstm1lstm(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1

class nn_se_rSpecMSE(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]

class nn_se_rSpecMSE_noStop(BaseConfig): # running 15041
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse"]
  no_stop = True

class nn_se_rReMagMSE100(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 100.0

class nn_se_rReMagMSE1000(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reMagMse"]
  relative_loss_AFD = 1000.0

class nn_se_rReSpecMSE100(p40): # running p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_AFD = 100.0

class nn_se_rReSpecMSE1000(BaseConfig): # running 15123
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reSpecMse"]
  relative_loss_AFD = 1000.0

class nn_se_rWavL1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_wav_L1"]

class nn_se_rWavL2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_wav_L2"]

class nn_se_rReWavL2_AFD100(p40): # running p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 100
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_AFD = 100.0

class nn_se_rReWavL2_AFD1000(p40): # running p40
  """
  cnn1blstm1lstm
  relative wav mse, AFD 1000
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_reWavL2"]
  relative_loss_AFD = 1000.0

class nn_se_rSDRv1(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_sdrV1"]

class nn_se_rSDRv2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_sdrV2"]

class nn_se_rSDRv3(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2)
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b005(p40): # done p40
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 0.05
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b010(p40): # done p40
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 0.1
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b050(p40): # done p40
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 0.5
  loss_name = ["real_net_sdrV3"]

class nn_se_rSDRv3_b100(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  SDR V3 (1-cos^2), wav add bias to cal loss.
  Avoid over contribution of small value to loss.
  """
  blstm_layers = 1
  lstm_layers = 1
  sdrv3_bias = 1.0
  loss_name = ["real_net_sdrV3"]

class nn_se_rStSDRV3(BaseConfig): # done 15123
  """
  cnn1blstm1lstm
  short time SDR V3 (1-cos^2)
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_stSDRV3"]
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128

class nn_se_rStSDRV3_f1024(p40): # done p40
  """
  cnn1blstm1lstm
  short time SDR V3 (1-cos^2)
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_stSDRV3"]
  st_frame_length_for_loss = 1024
  st_frame_step_for_loss = 256

class nn_se_rCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_cosSimV1"]

class nn_se_rCosSimV1WT10(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_cosSimV1WT10"]

class nn_se_rCosSimV2(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_cosSimV2"]

class nn_se_rSpecTCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_specTCosSimV1"]

class nn_se_rSpecFCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_specFCosSimV1"]

class nn_se_rSpecTFCosSimV1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_specTFCosSimV1"]

class nn_se_rSpecMseSDRv3_1_1(p40): # done p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_spec_mse", "real_net_sdrV3"]
  loss_weight = [1.0, 1.0]

class nn_se_rWavL2SDRv3_1_1(p40): # running p40
  """
  cnn1blstm1lstm
  """
  blstm_layers = 1
  lstm_layers = 1
  loss_name = ["real_net_wav_L2", "real_net_sdrV3"]
  loss_weight = [1.0, 1.0]

class nn_se_rSTWavMSE256(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 256
  frame_step = 64
  fft_dot = 256
  loss_weight = [100.0]

class nn_se_rSTWavMSE512(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 512
  frame_step = 128
  fft_dot = 512
  loss_weight = [100.0]
  GPU_PARTION = 0.3

class nn_se_rSTWavMSE256Map(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 256
  frame_step = 64
  fft_dot = 256
  loss_weight = [100.0]
  net_out_mask = False

class nn_se_rSTWavMSE256Map_noStop(BaseConfig): # running 15041
  """
  cnn1blstm1lstm
  short time wav as feature
  no stop
  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 256
  frame_step = 64
  fft_dot = 256
  loss_weight = [100.0]
  net_out_mask = False
  no_stop = True

class nn_se_rSTWavMSE512Map(p40): # done p40
  """
  cnn1blstm1lstm
  short time wav as feature

  """
  blstm_layers = 1
  lstm_layers = 1
  use_wav_as_feature = True
  frame_length = 512
  frame_step = 128
  fft_dot = 512
  loss_weight = [100.0]
  net_out_mask = False
  GPU_PARTION = 0.3

PARAM = nn_se_rSTWavMSE256Map_noStop

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m xxx._2_train
