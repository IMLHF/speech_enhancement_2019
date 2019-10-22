import tensorflow as tf

from .modules import Module
from ..FLAGS import PARAM
from ..utils import losses
from .modules import RealVariables


class CNN_RNN_FC_REAL_MASK_MODEL(Module):
  def __init__(self,
               mode,
               variables: RealVariables,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    super(CNN_RNN_FC_REAL_MASK_MODEL, self).__init__(
        mode,
        variables,
        mixed_wav_batch,
        clean_wav_batch,
        noise_wav_batch)

  def forward(self, mixed_wav_batch):
    outputs = self.real_networks_forward(mixed_wav_batch)
    est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch = outputs
    return est_clean_mag_batch, est_clean_spec_batch, est_clean_wav_batch
