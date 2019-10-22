import tensorflow as tf
import sys

def tf_to_complex(real, imag):
  complex_v = tf.cast(real, tf.dtypes.complex64) + 1j*tf.cast(imag, tf.dtypes.complex64)
  return complex_v

def tf_complex_multiply(a, b):
  # (a_real+a_imag*i) * (b_real+b_imag*i)
  a_real = tf.math.real(a)
  a_imag = tf.math.imag(a)
  b_real = tf.math.real(b)
  b_imag = tf.math.imag(b)
  ans_real = tf.multiply(a_real, b_real) - tf.multiply(a_imag, b_imag)
  ans_imag = tf.multiply(a_real, b_imag) + tf.multiply(b_real, a_imag)
  # ans_real = tf.check_numerics(ans_real, 'ans_real is nan')
  # ans_imag = tf.check_numerics(ans_imag, 'ans_imag is nan')
  ans_real = tf.clip_by_value(ans_real, sys.float_info.min, sys.float_info.max)
  ans_imag = tf.clip_by_value(ans_imag, sys.float_info.min, sys.float_info.max)
  ans = tf_to_complex(ans_real, ans_imag)

  return ans
