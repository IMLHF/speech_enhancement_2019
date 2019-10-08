import tensorflow as tf

from .modules import Module
from ..FLAGS import PARAM
from ..utils import losses


class CNN_RNN_REAL_MASK_MODEL(Module):
  def forward(self, mixed_wav_batch):
    outputs = self.real_networks_forward(mixed_wav_batch)
    est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch = outputs
    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch

  def get_loss(self, forward_outputs):
    est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch = forward_outputs

    # region losses
    ## frequency domain loss
    self.mag_mse = losses.batch_time_fea_real_mse(est_clean_mag_batch, self.clean_mag_batch)
    self.spec_mse = losses.batch_time_fea_complex_mse(est_clean_spec_batch, self.clean_spec_batch)

    ## time domain loss
    est_wav_len = tf.shape(est_clean_wav_batch)[-1]
    clean_wav_batch = tf.slice(self.clean_wav_batch, [0,0], [-1, est_wav_len])
    self.clean_wav_L1_loss = losses.batch_wav_L1_loss(est_clean_wav_batch, clean_wav_batch)
    self.clean_wav_L2_loss = losses.batch_wav_L2_loss(est_clean_wav_batch, clean_wav_batch)
    # engregion losses

    loss = 0
    loss_names = PARAM.loss_name.split("+")

    for name in loss_names:
      loss += {
        'mag_mse': self.mag_mse,
        'spec_mse': self.spec_mse,
        'clean_wav_L1_loss': self.clean_wav_L1_loss,
        'clean_wav_L2_loss': self.clean_wav_L2_loss,
      }[name]
    return loss
