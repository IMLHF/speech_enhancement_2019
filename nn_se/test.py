from .dataloader.dataloader import get_batch_inputs_from_dataset
from .utils import audio
from .FLAGS import PARAM
import tensorflow as tf


def test_dataloader_py():
  batch=get_batch_inputs_from_dataset(PARAM.train_name)
  sess=tf.Session()
  sess.run(batch.initializer)
  clean, noise, mixed=sess.run([batch.clean, batch.noise, batch.mixed])
  audio.write_audio(clean,"exp/test/clean.wav")
  audio.write_audio(noise,"exp/test/noise.wav")
  audio.write_audio(mixed,"exp/test/mixed.wav")


if __name__ == "__main__":
  test_dataloader_py()
