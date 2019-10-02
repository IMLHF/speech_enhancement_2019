from .dataloader.dataloader import get_batch_inputs_from_dataset
from .utils import audio
from .FLAGS import PARAM
import tensorflow as tf
import numpy as np
import os

def test_dataloader_py():
  batch=get_batch_inputs_from_dataset(PARAM.train_name)
  sess=tf.compat.v1.Session()
  sess.run(batch.initializer)
  clean, noise, mixed=sess.run([batch.clean, batch.noise, batch.mixed])
  print(np.shape(clean))
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/clean.wav"),clean[0],16000)
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/noise.wav"),noise[0],16000)
  audio.write_audio(os.path.join(PARAM.root_dir,"exp/test/mixed.wav"),mixed[0],16000)


if __name__ == "__main__":
  test_dataloader_py()
