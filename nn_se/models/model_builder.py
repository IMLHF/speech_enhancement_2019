from .real_mask_model import CNN_RNN_FC_REAL_MASK_MODEL
from .complex_mask_model import CCNN_CRNN_CFC_COMPLEX_MASK_MODEL
from . import modules
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class, var = {
      "CNN_RNN_FC_REAL_MASK_MODEL": (CNN_RNN_FC_REAL_MASK_MODEL, modules.RealVariables),
      "CCNN_CRNN_CFC_COMPLEX_MASK_MODEL": (CCNN_CRNN_CFC_COMPLEX_MASK_MODEL, modules.ComplexVariables),
  }[PARAM.model_name]

  return model_class, var
