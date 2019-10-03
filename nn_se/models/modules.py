import tensorflow as tf

from ..FLAGS import PARAM

class Module(object):
  def __init__(self, *args, **kwargs):
    pass

  def CNN_RNN_FC(self, mixed_mag_batch):
    mixed_mag_batch = tf.expand_dims(mixed_mag_batch, -1) # [batch, time, fft_dot, 1]
    outputs = tf.keras.layers.Conv2D(8, [5,5], padding="same",
                                     name='conv2_1')(mixed_mag_batch)  # [batch, time, fft_dot, 8]
    outputs = tf.keras.layers.Conv2D(16, [5,5], dilation_rate=[2,2], padding="same",
                                     name='conv2_2')(outputs)  # [batch, t, f, 16]
    outputs = tf.keras.layers.Conv2D(8, [5,5], dilation_rate=[4,4], padding="same",
                                     name='conv2_3')(outputs)  # [batch, t, f, 8]
    outputs = tf.keras.layers.Conv2D(1, [5,5], padding="same",
                                     name='conv2_4')(outputs)  # [batch, t, f, 1]
    outputs = tf.squeeze(outputs, [-1]) # [batcch, time, fft_dot]

    for i in PARAM.blstm_layers:
      forward_lstm = tf.keras.layers.LSTM(512, dropout=0.2, name='fwlstm_%d'%i)
      backward_lstm = tf.keras.layers.LSTM(512, dropout=0.2, name='bwlstm_%d'%i)
      blstm = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backward_lstm, name='blstm_%d'%i)
      outputs = blstm(outputs)

    outputs = tf.keras.layers.Dense(PARAM.fft_dot, name='out_fc')(outputs)
    return outputs

