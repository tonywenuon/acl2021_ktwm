
import keras
from keras.layers import Lambda, Add
from keras.layers import Layer
from keras import backend as K
from keras.utils import get_custom_objects


class ProbCalcLayer(Layer):
    """get probability of generate from source layer.
    """
            
    def __init__(self, 
                 src_facts_number,
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
        :param src_facts_number: total number of src and facts.
        :param activation: Activations for linear mappings.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        """
        self.src_facts_number = src_facts_number
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        super(ProbCalcLayer, self).__init__(**kwargs)

    def get_config(self):
        config = { 
            'src_facts_number': self.src_facts_number,
            'dropout': self.dropout,
            'activation': keras.activations.serialize(self.activation), 
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer), 
            'bias_initializer': keras.initializers.serialize(self.bias_initializer), 
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer), 
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer), 
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint), 
            'bias_constraint': keras.constraints.serialize(self.bias_constraint), 
        } 
        base_config = super(ProbCalcLayer, self).get_config() 
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(ProbCalcLayer, self).build(input_shape)

    def apply_dropout_if_needed(self, _input, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(_input, self.dropout)

            return K.in_train_phase(dropped_softmax, _input,
                                    training=training)
        return _input

    def compute_output_shape(self, input_shape):
        _shape = input_shape
        #_shape = input_shape[0]
        return (_shape[0], 1, _shape[-2],_shape[-1])

    def call(self, inputs):
        sf_outputs = inputs

        sf_outputs = K.permute_dimensions(sf_outputs, [0, 2, 1, 3])

        result = K.sum(sf_outputs, axis=-2) # shape: (bs, seq_len, hidden_dim)
        result = K.expand_dims(result, axis=1)

        return result

get_custom_objects().update({
    'ProbCalcLayer': ProbCalcLayer,
})

