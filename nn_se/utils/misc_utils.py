import tensorflow as tf
import sys
import tensorflow.contrib.slim as slim
import time
from distutils import version
from pathlib import Path

from ..FLAGS import PARAM

def test_records_save_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('test_records')


def hparams_file_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('hparam')


def ckpt_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('ckpt')


def test_log_file_dir():
  log_dir_ = log_dir()
  return log_dir_.joinpath('test.log')


def train_log_file_dir():
  log_dir_ = log_dir()
  return log_dir_.joinpath('train.log')


def log_dir():
  exp_config_name_dir = exp_configName_dir()
  return exp_config_name_dir.joinpath('log')


def exp_configName_dir():
  return Path(PARAM.root_dir).joinpath('exp', PARAM().config_name())


def datasets_dir():
  return Path(PARAM.root_dir).joinpath("datasets")


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
  '''Noam scheme learning rate decay
  init_lr: initial learning rate. scalar.
  global_step: scalar.
  warmup_steps: scalar. During warmup_steps, learning rate increases
      until it reaches init_lr.
  '''
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def show_variables(vars_):
  slim.model_analyzer.analyze_vars(vars_, print_info=True)
  sys.stdout.flush()


def show_all_variables():
  model_vars = tf.trainable_variables()
  show_variables(model_vars)


def print_log(msg, log_file=None, no_time=False):
  if log_file is not None:
    log_file = str(log_file)
  if not no_time:
      time_stmp = "%s | " % time.ctime()
      msg = time_stmp+msg
  print(msg, end='', flush=True)
  if log_file:
    with open(log_file, 'a+') as f:
        f.write(msg)


def check_tensorflow_version():
  # LINT.IfChange
  min_tf_version = PARAM.min_TF_version
  # LINT.ThenChange(<pwd>/nmt/copy.bara.sky)
  if not (version.LooseVersion(tf.__version__) == version.LooseVersion(min_tf_version)):
    raise EnvironmentError("Tensorflow version must be '%s'" % min_tf_version)


def save_hparams(f):
  f = open(f, 'a+')
  from .. import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  f.writelines('FLAGS.PARAM:\n')
  supper_dict = FLAGS.BaseConfig.__dict__
  for key in sorted(supper_dict.keys()):
    if key in self_dict_keys:
      f.write('%s:%s\n' % (key,self_dict[key]))
    else:
      f.write('%s:%s\n' % (key,supper_dict[key]))
  f.write('--------------------------\n\n')

  f.write('Short hparams:\n')
  [f.write("%s:%s\n" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  f.write('--------------------------\n\n')


def print_hparams(short=True):
  from .. import FLAGS
  self_dict = FLAGS.PARAM.__dict__
  self_dict_keys = self_dict.keys()
  print('\n--------------------------\n')
  print('Short hparams:')
  [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
  print('--------------------------\n')
  if not short:
    print('FLAGS.PARAM:')
    supper_dict = FLAGS.BaseConfig.__dict__
    for key in sorted(supper_dict.keys()):
      if key in self_dict_keys:
        print('%s:%s' % (key,self_dict[key]))
      else:
        print('%s:%s' % (key,supper_dict[key]))
    print('--------------------------\n')
    print('Short hparams:')
    [print("%s:%s" % (key, self_dict[key])) for key in sorted(self_dict_keys)]
    print('--------------------------\n')
