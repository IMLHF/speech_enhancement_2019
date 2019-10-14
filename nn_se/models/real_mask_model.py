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
    self.real_net_mag_mse = losses.batch_time_fea_real_mse(est_clean_mag_batch, self.clean_mag_batch)
    self.real_net_spec_mse = losses.batch_time_fea_complex_mse(est_clean_spec_batch, self.clean_spec_batch)

    ## time domain loss
    self.real_net_wav_L1 = losses.batch_wav_L1_loss(est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.real_net_wav_L2 = losses.batch_wav_L2_loss(est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.real_net_sdrV1 = losses.batch_sdrV1_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV2 = losses.batch_sdrV2_loss(est_clean_wav_batch, self.clean_wav_batch)
    # engregion losses

    loss = 0
    loss_names = PARAM.loss_name

    for name in loss_names:
      loss += {
        'real_net_mag_mse': self.real_net_mag_mse,
        'real_net_spec_mse': self.real_net_spec_mse,
        'real_net_wav_L1': self.real_net_wav_L1,
        'real_net_wav_L2': self.real_net_wav_L2,
        'real_net_sdrV1': self.real_net_sdrV1,
        'real_net_sdrV2': self.real_net_sdrV2,
      }[name]
    return loss
