from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers




def limit_mem():
    tf.compat.v1.keras.backend.get_session().close()
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=cfg))


def tensorflow_variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class SparseToDense(Dense):

    def call(self, inputs):
        output = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Dense3D(Dense):

    def __init__(self,
                 units,
                 heads,
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

        super(Dense3D, self).__init__(
            units * heads,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.heads = int(heads)

    def call(self, inputs):
        shape = inputs.get_shape().as_list()
        output_shape = list(map(lambda x: -1 if x is None else x, shape[:-1])) + [self.units // self.heads, self.heads]
        return tf.reshape(super(Dense3D, self).call(inputs), np.asarray(output_shape))

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if isinstance(input_shape[-1], tensor_shape.Dimension):
            dim_value = input_shape[-1].value
        else:
            dim_value = input_shape[-1]
        if dim_value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units // self.heads).concatenate(self.heads)

    def get_config(self):
        base_config = super(Dense3D, self).get_config().copy()
        base_config['units'] = self.units // self.heads
        base_config['heads'] = self.heads
        return base_config


class DenseSplitted(Dense):

    def __init__(self,
                 units,
                 heads,
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

        super(DenseSplitted, self).__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.heads = int(heads)

    def build(self, input_shape):

        input_shape = tensor_shape.TensorShape(input_shape)
        if isinstance(input_shape[-1], tensor_shape.Dimension):
            last_dim = input_shape[-1].value
        else:
            last_dim = input_shape[-1]

        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        # units = n items
        # last_dim = neurons in last layer
        # heads = neurons of this layer

        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.units, last_dim, self.heads],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, self.heads],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)

        shape = list(map(lambda x: -1 if x is None else x,
                         inputs.get_shape().as_list()[:1] + self.kernel.get_shape().as_list()))

        # Old way
        #outputs = tf.squeeze(tf.matmul(tf.expand_dims(inputs, axis=-2),
        #                               tf.tile(tf.expand_dims(self.kernel, axis=0),
        #                                       (tf.shape(input=inputs)[0], 1, 1, 1))),
        #                     axis=-2)

        outputs = tf.reduce_sum(
            tf.multiply(
                tf.expand_dims(inputs, axis=-1),
                self.kernel
            ), axis=-2)

        if self.use_bias:
            outputs = tf.add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if isinstance(input_shape[-1], tensor_shape.Dimension):
            dim_val = input_shape[-1].value
        else:
            dim_val = input_shape[-1]
        if dim_val is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.heads)

    def get_config(self):
        base_config = super(DenseSplitted, self).get_config().copy()
        base_config['heads'] = self.heads
        return base_config


class MultiHeadAttention(Layer):

    def __init__(self, q_dim, v_dim, heads,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MultiHeadAttention, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.q_dim = int(q_dim)
        self.v_dim = int(v_dim)
        self.z_dim = None
        self.heads = int(heads)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):

        input_shape = tensor_shape.TensorShape(input_shape)
        if isinstance(input_shape[-1], tensor_shape.Dimension):
            last_dim = input_shape[-1].value
        else:
            last_dim = input_shape[-1]

        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        self.z_dim = last_dim

        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.wq = [self.add_weight('wq',
                                   shape=[last_dim, self.q_dim],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   dtype=self.dtype,
                                   trainable=True) for _ in range(self.heads)]
        self.wk = [self.add_weight('wk',
                                   shape=[last_dim, self.q_dim],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   dtype=self.dtype,
                                   trainable=True) for _ in range(self.heads)]
        self.wv = [self.add_weight('wv',
                                   shape=[last_dim, self.v_dim],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   dtype=self.dtype,
                                   trainable=True) for _ in range(self.heads)]
        self.wz = self.add_weight('wv',
                                  shape=[self.heads * self.v_dim, self.z_dim],
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  dtype=self.dtype,
                                  trainable=True)
        self.built = True

    def call(self, query, key):
        query = ops.convert_to_tensor(query)
        key = ops.convert_to_tensor(key)

        qs = [tf.matmul(query, w) for w in self.wq]
        ks = [tf.matmul(key, w) for w in self.wk]
        vs = [tf.matmul(key, w) for w in self.wv]

        attentions = [tf.keras.backend.batch_dot(qs[i], ks[i], axes=(1, 2)) for i in range(self.heads)]
        attentions = [tf.keras.backend.batch_dot(tf.nn.softmax(attentions[i] / tf.math.sqrt(float(self.q_dim))), vs[i],
                                                 axes=(1, 1))
                      for i in range(self.heads)]

        outputs = tf.matmul(tf.concat(attentions, axis=-1), self.wz)
        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable
        print(outputs.shape)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if isinstance(input_shape[-1], tensor_shape.Dimension):
            dim_val = input_shape[-1].value
        else:
            dim_val = input_shape[-1]
        if dim_val is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.z_dim)


class MultiHeadSelfAttention(Layer):

    def __init__(self, q_dim, v_dim, heads,
                 add_normalization=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(
            q_dim, v_dim, heads,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs)
        self.add_normalization = add_normalization

    def call(self, inputs):
        outputs = super(MultiHeadSelfAttention, self).call(inputs, inputs)

        if self.add_normalization:
            outputs = tf.keras.layers.LayerNormalization()(tf.math.add(outputs, inputs))

        return outputs

