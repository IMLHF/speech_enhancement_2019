from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow as tf

from ..utils import complex_value_ops as c_ops


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)


class ComplexValueLSTMCell(DropoutRNNCellMixin, Layer):
  """Cell class for the LSTM layer.
  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):

    super(ComplexValueLSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.implementation = implementation
    # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
    # and fixed after 2.7.16. Converting the state_size to wrapper around
    # NoDependency(), so that the base_layer.__setattr__ will not convert it to
    # ListWrapper. Down the stream, self.states will be a list since it is
    # generated from nest.map_structure with list, and tuple(list) will work
    # properly.
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel_real = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel_real',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.kernel_imag = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel_imag',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel_real = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel_real',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    self.recurrent_kernel_imag = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel_imag',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias_real = self.add_weight(
          shape=(self.units * 4,),
          name='bias_real',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
      self.bias_imag = self.add_weight(
          shape=(self.units * 4,),
          name='bias_imag',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    # c_tm1 : complex64
    x_i_real, x_f_real, x_c_real, x_o_real, x_i_imag, x_f_imag, x_c_imag, x_o_imag = x
    (h_tm1_i_real, h_tm1_f_real, h_tm1_c_real, h_tm1_o_real,
     h_tm1_i_imag, h_tm1_f_imag, h_tm1_c_imag, h_tm1_o_imag) = h_tm1
    i_real = self.recurrent_activation(
        x_i_real + K.dot(
            h_tm1_i_real, self.recurrent_kernel_real[:, :self.units]) - K.dot(
            h_tm1_i_imag, self.recurrent_kernel_imag[:, :self.units]))
    i_imag = self.recurrent_activation(
        x_i_imag + K.dot(
            h_tm1_i_real, self.recurrent_kernel_imag[:, :self.units]) + K.dot(
            h_tm1_i_imag, self.recurrent_kernel_real[:, :self.units]))
    f_real = self.recurrent_activation(
        x_f_real + K.dot(
            h_tm1_f_real, self.recurrent_kernel_real[:, self.units:self.units*2]) - K.dot(
            h_tm1_f_imag, self.recurrent_kernel_imag[:, self.units:self.units*2]))
    f_imag = self.recurrent_activation(
        x_f_imag + K.dot(
            h_tm1_f_real, self.recurrent_kernel_imag[:, self.units:self.units*2]) + K.dot(
            h_tm1_f_imag, self.recurrent_kernel_real[:, self.units:self.units*2]))
    i_complex = c_ops.tf_to_complex(i_real, i_imag)
    f_complex = c_ops.tf_to_complex(f_real, f_imag)
    c_act_real = self.recurrent_activation(
        x_c_real + K.dot(
            h_tm1_c_real, self.recurrent_kernel_real[:, self.units*2:self.units*3]) - K.dot(
            h_tm1_c_imag, self.recurrent_kernel_imag[:, self.units*2:self.units*3]))
    c_act_imag = self.recurrent_activation(
        x_c_imag + K.dot(
            h_tm1_c_real, self.recurrent_kernel_imag[:, self.units*2:self.units*3]) + K.dot(
            h_tm1_c_imag, self.recurrent_kernel_real[:, self.units*2:self.units*3]))
    c_act_complex = c_ops.tf_to_complex(c_act_real, c_act_imag)

    c_complex = c_ops.tf_complex_multiply(f_complex, c_tm1) + c_ops.tf_complex_multiply(i_complex, c_act_complex)
    o_real = self.recurrent_activation(
        x_o_real + K.dot(
            h_tm1_o_real, self.recurrent_kernel_real[:, self.units*3:]) - K.dot(
            h_tm1_o_imag, self.recurrent_kernel_imag[:, self.units*3:]))
    o_imag = self.recurrent_activation(
        x_o_imag + K.dot(
            h_tm1_o_real, self.recurrent_kernel_imag[:, self.units*3:]) + K.dot(
            h_tm1_o_imag, self.recurrent_kernel_real[:, self.units*3:]))
    o_complex = c_ops.tf_to_complex(o_real, o_imag)
    return c_complex, o_complex

  def _compute_carry_and_output_fused(self, z_real, z_imag, c_tm1):
    """Computes carry and output using fused kernels."""
    # c_tm1 : complex64
    z0_real, z1_real, z2_real, z3_real = z_real
    z0_imag, z1_imag, z2_imag, z3_imag = z_imag
    i_real = self.recurrent_activation(z0_real)
    i_imag = self.recurrent_activation(z0_imag)
    f_real = self.recurrent_activation(z1_real)
    f_imag = self.recurrent_activation(z1_imag)
    z2_act_real = self.activation(z2_real)
    z2_act_imag = self.activation(z2_imag)

    i_complex = c_ops.tf_to_complex(i_real, i_imag)
    f_complex = c_ops.tf_to_complex(f_real, f_imag)
    z2_act_complex = c_ops.tf_to_complex(z2_act_real, z2_act_imag)
    part1 = c_ops.tf_complex_multiply(f_complex, c_tm1)
    part2 = c_ops.tf_complex_multiply(i_complex, z2_act_complex)
    # part1_real = tf.cast(tf.math.real(part1), tf.float64)
    # part1_imag = tf.cast(tf.math.imag(part1), tf.float64)
    # part2_real = tf.cast(tf.math.real(part2), tf.float64)
    # part2_imag = tf.cast(tf.math.imag(part2), tf.float64)
    # # part1_real = tf.check_numerics(part1_real, 'part1_real is nan')
    # # part2_real = tf.check_numerics(part2_real, 'part2_real is nan')
    # # part1_imag = tf.check_numerics(part1_imag, 'part1_imag is nan')
    # # part2_imag = tf.check_numerics(part2_imag, 'part2_imag is nan')
    # c_real = part1_real + part2_real
    # c_imag = part1_imag + part2_imag
    # # c_real = tf.check_numerics(c_real, 'c_real is nan')
    # # c_imag = tf.check_numerics(c_imag, 'c_imag is nan')
    c_complex = part1 + part2
    # z3_real = tf.check_numerics(z3_real, 'z3_real is nan')
    # z3_imag = tf.check_numerics(z3_imag, 'z3_imag is nan')
    o_real = self.recurrent_activation(z3_real)
    o_imag = self.recurrent_activation(z3_imag)
    o_complex = c_ops.tf_to_complex(o_real, o_imag)
    return c_complex, o_complex

  def call(self, inputs, states, training=None):
    inputs_real = tf.math.real(inputs)
    inputs_imag = tf.math.imag(inputs)
    h_tm1 = states[0]  # previous memory state
    h_tm1_real = tf.math.real(h_tm1)
    h_tm1_imag = tf.math.imag(h_tm1)
    c_tm1 = states[1]  # previous carry state

    dp_mask_real = self.get_dropout_mask_for_cell(inputs_real, training, count=4)
    dp_mask_imag = self.get_dropout_mask_for_cell(inputs_imag, training, count=4)
    rec_dp_mask_real = self.get_recurrent_dropout_mask_for_cell(
        h_tm1_real, training, count=4)
    rec_dp_mask_imag = self.get_recurrent_dropout_mask_for_cell(
        h_tm1_imag, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i_real = inputs_real * dp_mask_real[0]
        inputs_f_real = inputs_real * dp_mask_real[1]
        inputs_c_real = inputs_real * dp_mask_real[2]
        inputs_o_real = inputs_real * dp_mask_real[3]
        inputs_i_imag = inputs_imag * dp_mask_imag[0]
        inputs_f_imag = inputs_imag * dp_mask_imag[1]
        inputs_c_imag = inputs_imag * dp_mask_imag[2]
        inputs_o_imag = inputs_imag * dp_mask_imag[3]
      else:
        inputs_i_real = inputs_real
        inputs_f_real = inputs_real
        inputs_c_real = inputs_real
        inputs_o_real = inputs_real
        inputs_i_imag = inputs_imag
        inputs_f_imag = inputs_imag
        inputs_c_imag = inputs_imag
        inputs_o_imag = inputs_imag
      k_i_real, k_f_real, k_c_real, k_o_real = array_ops.split(
          self.kernel_real, num_or_size_splits=4, axis=1)
      k_i_imag, k_f_imag, k_c_imag, k_o_imag = array_ops.split(
          self.kernel_imag, num_or_size_splits=4, axis=1)
      x_i_real = K.dot(inputs_i_real, k_i_real) - K.dot(inputs_i_imag, k_i_imag)
      x_i_imag = K.dot(inputs_i_real, k_i_imag) + K.dot(inputs_i_imag, k_i_real)
      x_f_real = K.dot(inputs_f_real, k_f_real) - K.dot(inputs_f_imag, k_f_imag)
      x_f_imag = K.dot(inputs_f_real, k_f_imag) + K.dot(inputs_f_imag, k_f_real)
      x_c_real = K.dot(inputs_c_real, k_c_real) - K.dot(inputs_c_imag, k_c_imag)
      x_c_imag = K.dot(inputs_c_real, k_c_imag) + K.dot(inputs_c_imag, k_c_real)
      x_o_real = K.dot(inputs_o_real, k_o_real) - K.dot(inputs_o_imag, k_o_imag)
      x_o_imag = K.dot(inputs_o_real, k_o_imag) + K.dot(inputs_o_imag, k_o_real)
      if self.use_bias:
        b_i_real, b_f_real, b_c_real, b_o_real = array_ops.split(
            self.bias_real, num_or_size_splits=4, axis=0)
        b_i_imag, b_f_imag, b_c_imag, b_o_imag = array_ops.split(
            self.bias_imag, num_or_size_splits=4, axis=0)
        x_i_real = K.bias_add(x_i_real, b_i_real)
        x_f_real = K.bias_add(x_f_real, b_f_real)
        x_c_real = K.bias_add(x_c_real, b_c_real)
        x_o_real = K.bias_add(x_o_real, b_o_real)
        x_i_imag = K.bias_add(x_i_imag, b_i_imag)
        x_f_imag = K.bias_add(x_f_imag, b_f_imag)
        x_c_imag = K.bias_add(x_c_imag, b_c_imag)
        x_o_imag = K.bias_add(x_o_imag, b_o_imag)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i_real = h_tm1_real * rec_dp_mask_real[0]
        h_tm1_f_real = h_tm1_real * rec_dp_mask_real[1]
        h_tm1_c_real = h_tm1_real * rec_dp_mask_real[2]
        h_tm1_o_real = h_tm1_real * rec_dp_mask_real[3]
        h_tm1_i_imag = h_tm1_imag * rec_dp_mask_imag[0]
        h_tm1_f_imag = h_tm1_imag * rec_dp_mask_imag[1]
        h_tm1_c_imag = h_tm1_imag * rec_dp_mask_imag[2]
        h_tm1_o_imag = h_tm1_imag * rec_dp_mask_imag[3]
      else:
        h_tm1_i_real = h_tm1_real
        h_tm1_f_real = h_tm1_real
        h_tm1_c_real = h_tm1_real
        h_tm1_o_real = h_tm1_real
        h_tm1_i_imag = h_tm1_imag
        h_tm1_f_imag = h_tm1_imag
        h_tm1_c_imag = h_tm1_imag
        h_tm1_o_imag = h_tm1_imag
      x = (x_i_real, x_f_real, x_c_real, x_o_real,
           x_i_imag, x_f_imag, x_c_imag, x_o_imag)
      h_tm1 = (h_tm1_i_real, h_tm1_f_real, h_tm1_c_real, h_tm1_o_real,
               h_tm1_i_imag, h_tm1_f_imag, h_tm1_c_imag, h_tm1_o_imag)
      c_complex, o_complex = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs_real *= dp_mask_real[0]
        inputs_imag *= dp_mask_imag[0]
      z_real = K.dot(inputs_real, self.kernel_real) - K.dot(inputs_imag, self.kernel_imag)
      z_imag = K.dot(inputs_real, self.kernel_imag) + K.dot(inputs_imag, self.kernel_real)
      # z_real = tf.check_numerics(z_real, "nan z_real_1")
      # z_imag = tf.check_numerics(z_imag, "nan z_imag_1")
      if 0. < self.recurrent_dropout < 1.:
        h_tm1_real *= rec_dp_mask_real[0]
        h_tm1_imag *= rec_dp_mask_imag[0]
      z_real += K.dot(h_tm1_real, self.recurrent_kernel_real) - K.dot(h_tm1_imag, self.recurrent_kernel_imag)
      z_imag += K.dot(h_tm1_real, self.recurrent_kernel_imag) + K.dot(h_tm1_imag, self.recurrent_kernel_real)
      # z_real = tf.check_numerics(z_real, "nan z_real_2")
      # z_imag = tf.check_numerics(z_imag, "nan z_imag_2")
      if self.use_bias:
        z_real = K.bias_add(z_real, self.bias_real)
        z_imag = K.bias_add(z_imag, self.bias_imag)

      z_real = array_ops.split(z_real, num_or_size_splits=4, axis=1)
      z_imag = array_ops.split(z_imag, num_or_size_splits=4, axis=1)
      # z_real = tf.check_numerics(z_real, "nan z_real_3")
      # z_imag = tf.check_numerics(z_imag, "nan z_imag_3")
      z_real = (z_real[0], z_real[1], z_real[2], z_real[3])
      z_imag = (z_real[0], z_real[1], z_real[2], z_real[3])
      c_complex, o_complex = self._compute_carry_and_output_fused(z_real, z_imag, c_tm1)

    c_real = tf.math.real(c_complex)
    c_imag = tf.math.imag(c_complex)
    # c_real = tf.check_numerics(c_real, "nan c_real_1")
    # c_imag = tf.check_numerics(c_imag, "nan c_imag_1")
    c_act_real = self.activation(c_real)
    c_act_imag = self.activation(c_imag)
    # c_act_real = tf.check_numerics(c_act_real, "nan c_act_real_1")
    # c_act_imag = tf.check_numerics(c_act_imag, "nan c_act_imag_1")
    c_act_complex = c_ops.tf_to_complex(c_act_real, c_act_imag)
    h_complex = c_ops.tf_complex_multiply(o_complex, c_act_complex)
    return h_complex, [h_complex, c_complex]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(ComplexValueLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


class ComplexValueConv2d:
  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs
               ):
    name_real = None
    name_imag = None
    if 'name' in kwargs:
      name_real = kwargs['name']+"_real"
      name_imag = kwargs['name']+"_imag"
      del kwargs['name']
    self.w_real = tf.keras.layers.Conv2D(filters,
                                         kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format,
                                         dilation_rate=dilation_rate,
                                         activation=activation,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint,
                                         bias_constraint=bias_constraint,
                                         name=name_real,
                                         **kwargs)
    self.w_imag = tf.keras.layers.Conv2D(filters,
                                         kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format,
                                         dilation_rate=dilation_rate,
                                         activation=activation,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint,
                                         bias_constraint=bias_constraint,
                                         name=name_imag,
                                         **kwargs)

  def __call__(self, complex_inputs):
    real = tf.math.real(complex_inputs)
    imag = tf.math.imag(complex_inputs)

    ans_real = self.w_real(real) - self.w_imag(imag)
    ans_imag = self.w_real(imag) + self.w_imag(real)
    ans = c_ops.tf_to_complex(ans_real, ans_imag)
    return ans


class ComplexValueDense:
  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    name_real = None
    name_imag = None
    if 'name' in kwargs:
      name_real = kwargs['name']+"_real"
      name_imag = kwargs['name']+"_imag"
      del kwargs['name']

    self.dense_real = tf.keras.layers.Dense(units,
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            name=name_real,
                                            **kwargs)
    self.dense_imag = tf.keras.layers.Dense(units,
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            name=name_imag,
                                            **kwargs)

  def __call__(self, complex_inputs):
    real = tf.math.real(complex_inputs)
    imag = tf.math.imag(complex_inputs)

    ans_real = self.dense_real(real) - self.dense_imag(imag)
    ans_imag = self.dense_real(imag) + self.dense_imag(real)
    ans = c_ops.tf_to_complex(ans_real, ans_imag)
    return ans
