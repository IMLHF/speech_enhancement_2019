import tensorflow as tf
import abc
import collections
from typing import Union

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from ..utils import complex_value_ops as c_ops
from .complex_value_layers import ComplexValueConv2d
from .complex_value_layers import ComplexValueLSTMCell
from .complex_value_layers import ComplexValueDense


class ComplexVariables(object):
  """
  Complex Value NN Variables
  """
  def __init__(self):
    with tf.compat.v1.variable_scope("compat.v1.var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # CCNN
    conv2d_1 = ComplexValueConv2d(8, [5,5], padding="same", name='conv2_1') # -> [batch, time, fft_dot, 8]
    conv2d_2 = ComplexValueConv2d(16, [5,5], dilation_rate=[2,2], padding="same", name='conv2_2') # -> [batch, t, f, 16]
    conv2d_3 = ComplexValueConv2d(8, [5,5], dilation_rate=[4,4], padding="same", name='conv2_3') # -> [batch, t, f, 8]
    conv2d_4 = ComplexValueConv2d(1, [5,5], padding="same", name='conv2_4') # -> [batch, t, f, 1]
    self.conv2d_layers = [conv2d_1, conv2d_2, conv2d_3, conv2d_4]
    if PARAM.no_cnn:
      self.conv2d_layers = []

    # CBLSTM
    self.N_RNN_CELL = PARAM.rnn_units
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      complex_lstm_cell_f = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.lstmCell_implementation)
      complex_lstm_cell_b = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.lstmCell_implementation)
      forward_lstm = tf.keras.layers.RNN(complex_lstm_cell_f, return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.keras.layers.RNN(complex_lstm_cell_b, return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='blstm_%d' % i)
      self.blstm_layers.append(blstm)

    # CLSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      complex_lstm_cell = ComplexValueLSTMCell(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                               implementation=PARAM.lstmCell_implementation)
      lstm = tf.keras.layers.RNN(complex_lstm_cell, return_sequences=True, name='lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # CFC
    self.out_fc = ComplexValueDense(PARAM.fft_dot, name='out_fc')


class RealVariables(object):
  """
  Real Value NN Variables
  """
  def __init__(self):
    with tf.compat.v1.variable_scope("compat.v1.var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))

    # CNN
    conv2d_1 = tf.keras.layers.Conv2D(16, [5,5], padding="same", name='conv2_1') # -> [batch, time, fft_dot, 8]
    conv2d_2 = tf.keras.layers.Conv2D(32, [5,5], dilation_rate=[2,2], padding="same", name='conv2_2') # -> [batch, t, f, 16]
    conv2d_3 = tf.keras.layers.Conv2D(16, [5,5], dilation_rate=[4,4], padding="same", name='conv2_3') # -> [batch, t, f, 8]
    conv2d_4 = tf.keras.layers.Conv2D(1, [5,5], padding="same", name='conv2_4') # -> [batch, t, f, 1]
    self.conv2d_layers = [conv2d_1, conv2d_2, conv2d_3, conv2d_4]
    if PARAM.no_cnn:
      self.conv2d_layers = []

    # BLSTM
    self.N_RNN_CELL = PARAM.rnn_units
    self.blstm_layers = []
    for i in range(1, PARAM.blstm_layers+1):
      forward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.lstmCell_implementation,
                                          return_sequences=True, name='fwlstm_%d' % i)
      backward_lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, implementation=PARAM.lstmCell_implementation,
                                           return_sequences=True, name='bwlstm_%d' % i, go_backwards=True)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
                                            merge_mode='concat', name='blstm_%d' % i)
      self.blstm_layers.append(blstm)
    # self.blstm_layers = []
    # if PARAM.blstm_layers > 0:
    #   forward_lstm = tf.keras.layers.RNN(
    #       [tf.keras.layers.LSTMCell(
    #           self.N_RNN_CELL, dropout=0.2, name="lstm_%d" % i) for i in range(PARAM.blstm_layers)],
    #       return_sequences=True, name="fwlstm")
    #   backward_lstm = tf.keras.layers.RNN(
    #       [tf.keras.layers.LSTMCell(
    #           self.N_RNN_CELL, dropout=0.2, name="lstm_%d" % i) for i in range(PARAM.blstm_layers)],
    #       return_sequences=True, name="bwlstm", go_backwards=True)
    #   self.blstm_layers.append(tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm,
    #                                                          merge_mode='concat', name='blstm'))

    #LSTM
    self.lstm_layers = []
    for i in range(1, PARAM.lstm_layers+1):
      lstm = tf.keras.layers.LSTM(self.N_RNN_CELL, dropout=0.2, recurrent_dropout=0.1,
                                  return_sequences=True, implementation=PARAM.lstmCell_implementation,
                                  name='lstm_%d' % i)
      self.lstm_layers.append(lstm)

    # FC
    self.out_fc = tf.keras.layers.Dense(PARAM.fft_dot, name='out_fc')


class Module(object):
  """
  speech enhancement base.
  Discriminate spec and mag:
    spec: spectrum, complex value.
    mag: magnitude, real value.
  """
  def __init__(self,
               mode,
               variables: Union[RealVariables, ComplexVariables],
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
    if PARAM.use_wav_as_feature:
      self.clean_mag_batch = self.clean_spec_batch
    else:
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

    # LSTM
    for lstm in self.variables.lstm_layers:
      outputs = lstm(outputs, training=training)

    # FC
    if len(self.variables.blstm_layers) > 0 and len(self.variables.lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL])
    outputs = self.variables.out_fc(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])
    return outputs


  def real_networks_forward(self, mixed_wav_batch):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    if PARAM.use_wav_as_feature:
      mixed_mag_batch = mixed_spec_batch
    else:
      mixed_mag_batch = tf.math.abs(mixed_spec_batch)
    mixed_angle_batch = tf.math.angle(mixed_spec_batch)
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    mask = self.CNN_RNN_FC(mixed_mag_batch, training)

    if PARAM.net_out_mask:
      est_clean_mag_batch = tf.multiply(mask, mixed_mag_batch) # mag estimated
    else:
      est_clean_mag_batch = mask
    if PARAM.use_wav_as_feature:
      est_clean_spec_batch = est_clean_mag_batch
    else:
      est_clean_spec_batch = tf.cast(est_clean_mag_batch, tf.dtypes.complex64) * tf.exp(1j*tf.cast(mixed_angle_batch, tf.dtypes.complex64)) # complex
    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


  def CCNN_CRNN_CFC(self, mixed_spec_batch, training=False):
    mixed_spec_batch = tf.expand_dims(mixed_spec_batch, -1) # [batch, time, fft_dot, 1]
    outputs = mixed_spec_batch
    _batch_size = tf.shape(outputs)[0]
    # print(outputs.shape.as_list(), 'cnn_shape')

    # CNN
    for conv2d in self.variables.conv2d_layers:
      outputs = conv2d(outputs)
      # print(outputs.shape.as_list(), 'cnn_shape')
    if len(self.variables.conv2d_layers) > 0:
      outputs = tf.squeeze(outputs, [-1]) # [batch, time, fft_dot]

    # print(outputs.shape.as_list())
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])

    # CBLSTM
    for blstm in self.variables.blstm_layers:
      outputs = blstm(outputs, training=training)

    # CLSTM
    for lstm in self.variables.lstm_layers:
      outputs = lstm(outputs, training=training)

    # CFC
    if len(self.variables.blstm_layers) > 0 and len(self.variables.lstm_layers) <= 0:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL*2])
    else:
      outputs = tf.reshape(outputs, [-1, self.variables.N_RNN_CELL])
    outputs = self.variables.out_fc(outputs)
    outputs = tf.reshape(outputs, [_batch_size, -1, PARAM.fft_dot])
    return outputs


  def complex_networks_forward(self, mixed_wav_batch):
    mixed_spec_batch = misc_utils.tf_batch_stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    training = (self.mode == PARAM.MODEL_TRAIN_KEY)
    complex_mask = self.CCNN_CRNN_CFC(mixed_spec_batch, training)
    if PARAM.net_out_mask:
      est_clean_spec_batch = c_ops.tf_complex_multiply(complex_mask, mixed_spec_batch) # mag estimated
    else:
      est_clean_spec_batch = complex_mask
    _mixed_wav_len = tf.shape(mixed_wav_batch)[-1]
    _est_clean_wav_batch = misc_utils.tf_batch_istft(est_clean_spec_batch, PARAM.frame_length, PARAM.frame_step)
    est_clean_wav_batch = tf.slice(_est_clean_wav_batch, [0,0], [-1, _mixed_wav_len]) # if stft.pad_end=True, so est_wav may be longger than mixed.

    est_clean_mag_batch = tf.math.abs(est_clean_spec_batch)

    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch


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


  def get_loss(self, forward_outputs):
    est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch = forward_outputs

    # region losses
    ## frequency domain loss
    self.real_net_mag_mse = losses.batch_time_fea_real_mse(est_clean_mag_batch, self.clean_mag_batch)
    self.real_net_reMagMse = losses.batch_real_relativeMSE(est_clean_mag_batch, self.clean_mag_batch, PARAM.relative_loss_AFD)
    self.real_net_spec_mse = losses.batch_time_fea_complex_mse(est_clean_spec_batch, self.clean_spec_batch)
    self.real_net_reSpecMse = losses.batch_complex_relativeMSE(est_clean_spec_batch, self.clean_spec_batch, PARAM.relative_loss_AFD)
    self.real_net_specTCosSimV1 = losses.batch_complexspec_timeaxis_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specFCosSimV1 = losses.batch_complexspec_frequencyaxis_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specTFCosSimV1 = losses.batch_complexspec_TF_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167

    ## time domain loss
    self.real_net_wav_L1 = losses.batch_wav_L1_loss(est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.real_net_wav_L2 = losses.batch_wav_L2_loss(est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.real_net_reWavL2 = losses.batch_wav_relativeMSE(est_clean_wav_batch, self.clean_wav_batch, PARAM.relative_loss_AFD)
    self.real_net_sdrV1 = losses.batch_sdrV1_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV2 = losses.batch_sdrV2_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV3 = losses.batch_sdrV3_loss(est_clean_wav_batch, self.clean_wav_batch, PARAM.sdrv3_bias) # *0.167
    if PARAM.sdrv3_bias:
      assert PARAM.sdrv3_bias > 0.0, 'sdrv3_bias is constrained larger than zero.'
      self.real_net_sdrV3 *= 1.0 + 60 * PARAM.sdrv3_bias
    self.real_net_cosSimV1 = losses.batch_cosSimV1_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.real_net_cosSimV1WT10 = self.real_net_cosSimV1*0.167 # loss weighted to 10 level
    self.real_net_cosSimV2 = losses.batch_cosSimV2_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.334
    self.real_net_stSDRV3 = losses.batch_short_time_sdrV3_loss(est_clean_wav_batch, self.clean_wav_batch,
                                                               PARAM.st_frame_length_for_loss,
                                                               PARAM.st_frame_step_for_loss)
    # engregion losses

    loss = 0
    loss_names = PARAM.loss_name

    for i, name in enumerate(loss_names):
      loss_t = {
        'real_net_mag_mse': self.real_net_mag_mse,
        'real_net_reMagMse': self.real_net_reMagMse,
        'real_net_spec_mse': self.real_net_spec_mse,
        'real_net_reSpecMse': self.real_net_reSpecMse,
        'real_net_wav_L1': self.real_net_wav_L1,
        'real_net_wav_L2': self.real_net_wav_L2,
        'real_net_reWavL2': self.real_net_reWavL2,
        'real_net_sdrV1': self.real_net_sdrV1,
        'real_net_sdrV2': self.real_net_sdrV2,
        'real_net_sdrV3': self.real_net_sdrV3,
        'real_net_cosSimV1': self.real_net_cosSimV1,
        'real_net_cosSimV1WT10': self.real_net_cosSimV1WT10,
        'real_net_cosSimV2': self.real_net_cosSimV2,
        'real_net_specTCosSimV1': self.real_net_specTCosSimV1,
        'real_net_specFCosSimV1': self.real_net_specFCosSimV1,
        'real_net_specTFCosSimV1': self.real_net_specTFCosSimV1,
        'real_net_stSDRV3': self.real_net_stSDRV3,
      }[name]
      if len(PARAM.loss_weight) > 0:
        loss_t *= PARAM.loss_weight[i]
      loss += loss_t
    return loss


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
