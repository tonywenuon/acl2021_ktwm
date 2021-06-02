
import tensorflow as tf
import keras
from keras.layers import Lambda, Add
from keras.layers import Layer
from keras import backend as K
from keras.utils import get_custom_objects


class SimulateVector(Layer):
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
        self.args = args
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        super(SimulateVector, self).__init__(**kwargs)

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
        base_config = super(SimulateVector, self).get_config() 
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if type(input_shape) == 'list':
            enc = input_shape[0]
        else:
            enc = input_shape

        enc = input_shape[0]
        #enc = input_shape
        hidden_dim = enc[-1]
        # source, facts and self_attention Tensor
        self.W = [self.add_weight(
            name='Wsimu%s'%i,
            shape=(self.args.src_seq_length, hidden_dim, hidden_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        ) for i in range(self.args.tar_seq_length)] 
        self.b = [self.add_weight(
            name='bsimu%s'%i,
            shape=(hidden_dim,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        ) for i in range(self.args.tar_seq_length)]
        super(SimulateVector, self).build(input_shape)

    def apply_dropout_if_needed(self, _input, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(_input, self.dropout)

            return K.in_train_phase(dropped_softmax, _input,
                                    training=training)
        return _input

    def compute_output_shape(self, input_shape):
        # if have two inputs
        return input_shape[0]
        #return input_shape

    def call(self, inputs):
        enc_src, mask = inputs
        mask = K.expand_dims(mask, axis=-1)
        enc_src = enc_src * mask

        _shape = K.shape(enc_src)
        enc_src = K.reshape(enc_src, (_shape[0], _shape[-2], _shape[-1]))
        enc_src = K.permute_dimensions(enc_src, [1, 0, 2])
        hidden_dim = _shape[-1]

        results = []
        for i in range(self.args.tar_seq_length):
            _dot = K.dot(K.reshape(enc_src[0], (-1, hidden_dim)), self.W[i][0])
            for j in range(1, self.args.src_seq_length):
                _dot += K.dot(K.reshape(enc_src[j], (-1, hidden_dim)), self.W[i][j])
            _dot = K.bias_add(_dot, self.b[i])
            results.append(_dot)
        results = K.stack(results)
        results = K.permute_dimensions(results, [1, 0, 2])

        results = self.apply_dropout_if_needed(results)
        results = K.reshape(results, _shape)

        return results

get_custom_objects().update({
    'SimulateVector': SimulateVector,
})

