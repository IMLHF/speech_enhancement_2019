import tensorflow as tf
import abc
import collections

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils

class Variables(object):
  """
  NN Variables
  """
  def __init__(self):
    with tf.compat.v1.variable_scope("compat.v1.var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # CNN
    conv2d_1 = tf.keras.layers.Conv2D(8, [5,5], padding="same", name='conv2_1') # -> [batch, time, fft_dot, 8]
    conv2d_2 = tf.keras.layers.Conv2D(16, [5,5], dilation_rate=[2,2], padding="same", name='conv2_2') # -> [batch, t, f, 16]
    conv2d_3 = tf.keras.layers.Conv2D(8, [5,5], dilation_rate=[4,4], padding="same", name='conv2_3') # -> [batch, t, f, 8]
    conv2d_4 = tf.keras.layers.Conv2D(1, [5,5], padding="same", name='conv2_4') # -> [batch, t, f, 1]
    self.conv2d_layers = [conv2d_1, conv2d_2, conv2d_3, conv2d_4]
    if PARAM.no_cnn:
      self.conv2d_layers = []

    # BLSTM
    self.N_RNN_CELL = 512
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      forward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm, merge_mode='concat', name='blstm_%d' % i)
      self.blstm_layers.append(blstm)

    # FC
    self.out_fc = tf.keras.layers.Dense(PARAM.fft_dot, name='out_fc')


class Module(object):
  """
  speech enhancement base
  """
  def __init__(self,
               mode,
               variables: Variables,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    del noise_wav_batch
    self.mixed_wav_batch = mixed_wav_batch

    self.variables = variables
    self.mode = mode

    # global_step, lr, vars
    self._global_step = self.variables._global_step
    self._lr = self.variables._lr
    self.save_variables = [self.global_step, self._lr]

    # for lr halving
    self.new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self.assign_lr = tf.compat.v1.assign(self._lr, self.new_lr)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(self._lr, self.global_step, warmup_steps=PARAM.warmup_steps)


    # nn forward
    forward_outputs = self.forward(mixed_wav_batch)
    self._est_clean_wav_batch = forward_outputs[-1]

    trainable_variables = tf.compat.v1.trainable_variables()
    self.save_variables.extend([var for var in trainable_variables])
    self.saver = tf.compat.v1.train.Saver(self.save_variables, max_to_keep=PARAM.max_keep_ckpt, save_relative_paths=True)

    if mode == PARAM.MODEL_INFER_KEY:
      return

    # labels
    self.clean_wav_batch = clean_wav_batch
    self.clean_spec_batch = misc_utils.tf_batch_stft(clean_wav_batch, PARAM.frame_length, PARAM.frame_step) # complex label
    self.clean_mag_batch = tf.math.abs(self.clean_spec_batch) # mag label

    self._loss = self.get_loss(forward_outputs)

    if mode == PARAM.MODEL_VALIDATE_KEY:
      return

    # optimizer
    # opt = tf.keras.optimizers.Adam(learning_rate=self._lr)
    opt = tf.compat.v1.train.AdamOptimizer(self._lr)
    params = tf.compat.v1.trainable_variables()
    gradients = tf.gradients(
      self._loss,
      params,
      colocate_gradients_with_ops=True
    )
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, PARAM.max_gradient_norm)
    self._train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                         global_step=self.global_step)


  def CNN_RNN_FC(self, mixed_mag_batch, training=False):
    mixed_mag_batch = tf.expand_dims(mixed_mag_batch, -1) # [batch, time, fft_dot, 1]
    outputs = mixed_mag_batch
    _batch_size = tf.shape(outputs)[0]

    # CNN
    for conv2d in self.variables.conv2d_layers:
      outputs = conv2d(outputs)
    if len(self.variables.conv2d_layers) > 0:
      outputs = tf.squeeze(outputs, [-1]) # [batch, time, fft_dot]

    # print(outputs.shape.as_list())
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    # BLSTM
    for blstm in self.variables.blstm_layers:
      outputs = blstm(outputs, training=training)

    # FC
    if len(self.variables.blstm_layers) > 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, PARAM.fft_dot])
    outputs = self.variables.out_fc(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])
    return outputs


  def real_networks_forward(self, mixed_wav_batch):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    mixed_mag_batch = tf.math.abs(mixed_spec_batch)
    mixed_angle_batch = tf.math.angle(mixed_spec_batch)
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    mask = self.CNN_RNN_FC(mixed_mag_batch, training)

    est_clean_mag_batch = tf.multiply(mask, mixed_mag_batch) # mag estimated
    est_clean_spec_batch = tf.cast(est_clean_mag_batch, tf.dtypes.complex64) * tf.exp(1j*tf.cast(mixed_angle_batch, tf.dtypes.complex64)) # complex
    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def complex_networks_forward(self, mixed_wav_batch):
    # TODO
    return mixed_wav_batch


  @abc.abstractmethod
  def forward(mixed_wav_batch):
    """
    Returns:
      forward_outputs: pass to get_loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "forward not implement, code: 939iddfoollvoae")


  @abc.abstractmethod
  def get_loss(forward_outputs):
    """
    Returns:
      a tf number: loss
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "get_loss not implement, code: ppqekkgorkkfd")

  def change_lr(self, sess, new_lr):
    sess.run(self.assign_lr, feed_dict={self.new_lr:new_lr})

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_batch

  @property
  def global_step(self):
    return self._global_step

  @property
  def train_op(self):
    return self._train_op

  @property
  def loss(self):
    return self._loss

  @property
  def lr(self):
    return self._lr

  @property
  def est_clean_wav_batch(self):
    return self._est_clean_wav_batch
