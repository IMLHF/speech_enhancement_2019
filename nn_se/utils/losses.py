import tensorflow as tf

def batch_time_fea_real_mse(y1, y2):
  """
  y1: real, [batch, time, fft_dot]
  y2: real, [batch, time, fft_dot]
  """
  loss = tf.square(y1-y2)
  loss = tf.reduce_mean(tf.reduce_sum(loss, -1))
  return loss

def batch_time_fea_complex_mse(y1, y2):
  """
  y1: complex, [batch, time, fft_dot]
  y2: conplex, [batch, time, fft_dot]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_time_fea_real_mse(y1_real, y2_real)
  loss_imag = batch_time_fea_real_mse(y1_imag, y2_imag)
  loss = loss_real + loss_imag
  return loss

def batch_wav_L1_loss(y1, y2):
  loss = tf.reduce_mean(tf.abs(y1-y2))
  return loss

def batch_wav_L2_loss(y1, y2):
  loss = tf.reduce_mean(tf.square(y1-y2))
  return loss

def batch_wav_cos_Lp_loss(y1, y2, p):
  pass
