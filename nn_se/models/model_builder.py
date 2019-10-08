from .real_mask_model import CNN_RNN_REAL_MASK_MODEL
from ..FLAGS import PARAM

def get_model_class():
  model_class = {
      "CNN_RNN_REAL_MASK_MODEL": CNN_RNN_REAL_MASK_MODEL,
  }[PARAM.model_name]

  return model_class
