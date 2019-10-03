import tensorflow as tf

from .modules import Module
from ..FLAGS import PARAM
from ..utils import losses


class CNN_RNN_REAL_MASK_MODEL(Module):

  def __init__(self, clean_wav_batch, noise_wav_batch, mixed_wav_batch, behavior):
    del noise_wav_batch
    # labels
    self.clean_wav_batch = clean_wav_batch
    self.clean_spec_batch = tf.signal.stft(clean_wav_batch, PARAM.frame_length, PARAM.frame_step) # complex label
    self.clean_mag_batch = tf.math.abs(self.clean_spec_batch) # mag label

    # nn forward
    mixed_spec_batch = tf.signal.stft(mixed_wav_batch, PARAM.frame_length, PARAM.frame_step)
    mixed_mag_batch = tf.math.abs(mixed_spec_batch)
    mixed_angle_batch = tf.math.angle(mixed_spec_batch)
    mask = self.CNN_RNN_FC(mixed_mag_batch)

    # estimates
    self.est_clean_mag_batch = tf.multiply(mask, mixed_mag_batch) # mag estimated
    self.est_clean_spec_batch = tf.multiply(self.est_clean_mag_batch, tf.exp(1j*mixed_angle_batch)) # complex
    self.est_clean_wav_batch = tf.signal.inverse_stft(self.est_clean_spec_batch,PARAM.frame_length,PARAM.frame_step)

    # losses
    self.mag_mse = losses.batch_time_fea_real_mse(self.est_clean_mag_batch, self.clean_mag_batch)
    self.spec_mse = losses.batch_time_fea_complex_mse(self.est_clean_spec_batch, self.clean_spec_batch)
    self.clean_wav_L1_loss = losses.batch_wav_L1_loss(self.est_clean_wav_batch, self.clean_wav_batch)
    self.clean_wav_L2_loss = losses.batch_wav_L2_loss(self.est_clean_wav_batch, self.clean_wav_batch)

    self.loss = 0
    if PARAM.loss_name == "mag_mse":
      self.loss = self.mag_mse





