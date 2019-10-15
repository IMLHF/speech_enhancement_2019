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
    self.real_net_specTCosSimV1 = losses.batch_complexspec_timeaxis_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specFCosSimV1 = losses.batch_complexspec_frequencyaxis_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167
    self.real_net_specTFCosSimV1 = losses.batch_complexspec_TF_cos_sim_V1(est_clean_spec_batch, self.clean_spec_batch) # *0.167

    ## time domain loss
    self.real_net_wav_L1 = losses.batch_wav_L1_loss(est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.real_net_wav_L2 = losses.batch_wav_L2_loss(est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.real_net_sdrV1 = losses.batch_sdrV1_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV2 = losses.batch_sdrV2_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.real_net_sdrV3 = losses.batch_sdrV3_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.real_net_cosSimV1 = losses.batch_cosSimV1_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.167
    self.real_net_cosSimV1WT10 = self.real_net_cosSimV1*0.167 # loss weighted to 10 level
    self.real_net_cosSimV2 = losses.batch_cosSimV2_loss(est_clean_wav_batch, self.clean_wav_batch) # *0.334
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
        'real_net_sdrV3': self.real_net_sdrV3,
        'real_net_cosSimV1': self.real_net_cosSimV1,
        'real_net_cosSimV1WT10': self.real_net_cosSimV1WT10,
        'real_net_cosSimV2': self.real_net_cosSimV2,
        'real_net_specTCosSimV1': self.real_net_specTCosSimV1,
        'real_net_specFCosSimV1': self.real_net_specFCosSimV1,
        'real_net_specTFCosSimV1': self.real_net_specTFCosSimV1,
      }[name]
    return loss
