
import tensorflow as tf
import keras
from keras.layers import Lambda, Add
from keras.layers import Layer
from keras import backend as K
from keras.utils import get_custom_objects


class FactFilter(Layer):
    """get probability of generate from source layer.
    """
            
    def __init__(self, 
                 args,
                 dropout: float = 0.0,
                 activation='relu', 
                 kernel_initializer='glorot_normal', 
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None,
                 **kwargs): 
        """Initialize the layer.
        :param activation: Activations for linear mappings.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        """
        self.dropout = dropout
        self.args = args
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        super(FactFilter, self).__init__(**kwargs)

    def get_config(self):
        config = { 
            'args': self.args,
            'dropout': self.dropout,
            'activation': keras.activations.serialize(self.activation), 
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer), 
            'bias_initializer': keras.initializers.serialize(self.bias_initializer), 
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer), 
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer), 
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint), 
            'bias_constraint': keras.constraints.serialize(self.bias_constraint), 
        } 
        base_config = super(FactFilter, self).get_config() 
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(FactFilter, self).build(input_shape)

    def apply_dropout_if_needed(self, _input, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(_input, self.dropout)

            return K.in_train_phase(dropped_softmax, _input,
                                    training=training)
        return _input

    def gumbel_sampling(self, logits, tau):
        ep = 1e-20
        U = K.random_uniform(K.shape(logits), 0, 0.001)
        # add Gumbel noise
        y = logits - K.log(-K.log(U + ep) + ep)
        y = K.softmax(y / tau)

        y_hard = K.cast(K.equal(y, K.max(y, axis=-1, keepdims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
        #y = y_hard
        return y

    def compute_output_shape(self, input_shape):
        _shape = input_shape[3]
        #_shape = input_shape[4]
        _shape1 = (input_shape[0][0], input_shape[0][1], input_shape[0][2], 1)
        _shape2 = (input_shape[0][0], input_shape[0][1], 1, 1)
        #return [_shape, _shape1, _shape1, _shape1, _shape2, _shape2, _shape2]
        return [_shape, _shape1, _shape1, _shape1]

    def call(self, inputs):
        tar_ca, fact_ca, simulate_ca, fact_enc_output, inp_is_train, tar_mask, fact_mask = inputs
        fact_mask_shape = K.shape(fact_mask)
        t_shape = K.shape(tar_ca)
        f_shape = K.shape(fact_ca)
        s_shape = K.shape(simulate_ca)
        hidden_dim = t_shape[-1]

        # bs, tar_len, dim
        tar_vec = K.reshape(tar_ca, (t_shape[0], t_shape[-2], t_shape[-1]))
        tar_mask = K.permute_dimensions(tar_mask, [0, 2, 1])
        tar_vec = tar_vec * tar_mask
        tar_vec = K.expand_dims(tar_vec, axis=1)
        tar_vec = K.repeat_elements(tar_vec, self.args.fact_number, axis=1)
        tar_vec = K.reshape(tar_vec, (-1, t_shape[-2], t_shape[-1]))

        fact_vec = K.reshape(fact_ca, (-1, f_shape[-2], f_shape[-1]))
        #fact_enc_output = K.reshape(fact_enc_output, (-1, f_shape[-2], f_shape[-1]))
        fact_mask = K.reshape(fact_mask, (-1, fact_mask_shape[-1]))
        fact_mask = K.expand_dims(fact_mask, axis=-1)
        fact_vec = fact_vec * fact_mask

        simulate_vec = K.repeat_elements(simulate_ca, self.args.fact_number, axis=1)
        simulate_vec = K.reshape(simulate_vec, (-1, s_shape[-2], s_shape[-1]))

        fact_vec_trans = K.permute_dimensions(fact_vec, [0, 2, 1])

        # bs, tar_len, fact_len
        # word-level
        tar_fact_simis = K.batch_dot(tar_vec, fact_vec_trans)
        simulate_fact_simis = K.batch_dot(simulate_vec, fact_vec_trans)

        tar_fact_simis = K.sigmoid(tar_fact_simis)
        simulate_fact_simis = K.sigmoid(simulate_fact_simis)

        tar_fact_simis = K.mean(tar_fact_simis, axis=1) 
        tar_fact_simis = K.cast(tar_fact_simis, dtype=K.floatx())
        tar_fact_simis = K.reshape(tar_fact_simis, (f_shape[0], f_shape[1], f_shape[2], 1))
        simulate_fact_simis = K.mean(simulate_fact_simis, axis=1) 
        simulate_fact_simis = K.cast(simulate_fact_simis, dtype=K.floatx())
        simulate_fact_simis = K.reshape(simulate_fact_simis, (f_shape[0], f_shape[1], f_shape[2], 1))

        is_train = K.cast(K.mean(inp_is_train), dtype='int32')
        filter_simis = K.switch(K.equal(is_train, 1), tar_fact_simis, simulate_fact_simis)

        fact_mask = K.reshape(fact_mask, (fact_mask_shape[0], fact_mask_shape[1], fact_mask_shape[2], 1))
        filter_simis = filter_simis * fact_mask

        filter_fact = filter_simis * fact_enc_output
        final_simis = filter_simis

        return [filter_fact, tar_fact_simis, simulate_fact_simis, final_simis]

get_custom_objects().update({
    'FactFilter': FactFilter,
})

