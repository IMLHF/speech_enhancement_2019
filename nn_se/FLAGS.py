class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'val'
  MODEL_INFER_KEY = 'infer'

  train_name="train"
  validation_name="validation"
  test_name="test"

  def __class__name__(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/speech_enhancement_2019_exp/'
  '''
  # dir to store log, model and results files:
  $root_dir/datasets: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/summary: tensorboard summary
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/decode: decode results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"

  n_train_set_records = 20
  n_val_set_records = 20
  n_test_set_records = 20
  train_val_snr = [-5, 15]
  train_val_wav_seconds = 5.0

  sampling_rate = 16000

  n_processor_gen_tfrecords = 16
  tfrecords_num_pre_set = 160
  shuffle_records = True
  batch_size = 128
  n_processor_tfdata = 4



PARAM = BaseConfig
