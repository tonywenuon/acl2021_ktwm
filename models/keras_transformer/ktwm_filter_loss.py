import tensorflow as tf
import keras
from keras.layers import Lambda, Add
from keras.layers import Layer
from keras import backend as K
from keras.utils import get_custom_objects


class Loss(Layer):
    """get probability of generate from source layer.
    """
            
    def __init__(self, 
                 pad_id = 0,
                 **kwargs): 
        self.pad_id = pad_id
        super(Loss, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'pad_id': self.pad_id
        }
        base_config = super(Loss, self).get_config() 
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (1, )

    def mse(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def wasserstein_loss(self, y_true, y_pred):
	    return K.mean(y_true * y_pred)

    def kld(self, y_true, y_pred):
        return keras.losses.kullback_leibler_divergence(y_true, y_pred)

    def call(self, inputs):
        tar_fact_simis, simulate_fact_simis, final_simis, inp_fact_mask, word_predictions, inp_tar = inputs

        alpha = 0.33
        beta = 0.33

        word_preds = K.cast(K.sparse_categorical_crossentropy(inp_tar, word_predictions), dtype=K.floatx())
        loss1 = K.mean(word_preds)  

        tar_fact_simis = K.flatten(tar_fact_simis)
        simulate_fact_simis = K.flatten(simulate_fact_simis)
        loss2 = self.mse(tar_fact_simis, simulate_fact_simis)

        fm_shape = K.shape(inp_fact_mask)
        fact_mask = K.reshape(inp_fact_mask, (-1, fm_shape[-1]))
        final_simis = K.reshape(final_simis, (-1, fm_shape[-1]))
        loss4 = K.cast(K.binary_crossentropy(fact_mask, final_simis), dtype=K.floatx())
        loss4 = K.mean(loss4)

        loss = loss1 + loss2 + loss4

        #print('loss:', loss)
        self.add_loss(loss)

        return loss

get_custom_objects().update({
    'Loss': Loss,
})

