from keras import backend as K
from keras.layers import Layer
from keras.utils import get_custom_objects

class FactPaddingMaskLayer(Layer):
    def __init__(self, src_len, **kwargs):
        self.src_len = src_len
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['src_len'] = self.src_len
        return config

    def call(self, inputs):
        k = inputs
        # give padding position 0 as mask
        mask = K.cast(k, dtype=K.floatx())
        mask = K.expand_dims(mask, axis=-2)
        mask = K.repeat_elements(mask, self.src_len, axis = -2) # shape: (batch_size, q_len, k_len)
        return mask



class PaddingMaskLayer(Layer):
    def __init__(self, src_len, pad_id, **kwargs):
        self.src_len = src_len
        self.pad_id = pad_id
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['src_len'] = self.src_len
        config['pad_id'] = self.pad_id
        return config

    def call(self, inputs):
        k = inputs
        # give padding position 0 as mask
        mask = K.cast(K.equal(k, self.pad_id), dtype='int32')
        mask = 1 - mask
        mask = K.cast(mask, dtype=K.floatx())
        mask = K.expand_dims(mask, axis=-2)
        mask = K.repeat_elements(mask, self.src_len, axis = -2) # shape: (batch_size, q_len, k_len)
        return mask

class LatentPaddingMaskLayer(Layer):
    def __init__(self, src_len, pad_id, **kwargs):
        self.src_len = src_len
        self.pad_id = pad_id
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['src_len'] = self.src_len
        config['pad_id'] = self.pad_id
        return config

    def call(self, inputs):
        k, latent_mask = inputs
        k = K.concatenate([k, latent_mask], axis=-1)
        # give padding position 0 as mask
        mask = K.cast(K.equal(k, self.pad_id), dtype='int32')
        mask = 1 - mask
        mask = K.cast(mask, dtype=K.floatx())
        mask = K.expand_dims(mask, axis=-2)
        mask = K.repeat_elements(mask, self.src_len, axis = -2) # shape: (batch_size, q_len, k_len)
        return mask


class SequenceMaskLayer(Layer):
    def call(self, inputs):
        # used in decoder, for only seeing previous terms
        seq = inputs
        last_dims = K.shape(seq)
        batch_size, q_len = last_dims[-2], last_dims[-1]

        row = K.expand_dims(K.arange(0, q_len), axis=-1)
        col = K.expand_dims(K.arange(0, q_len), axis=0)
        mask = K.expand_dims(K.cast(col <= row, K.floatx()), axis=0)

        return mask

class SelfPadMaskLayer(Layer):
    def __init__(self, pad_id, **kwargs):
        self.pad_id = pad_id
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['pad_id'] = self.pad_id
        return config

    def call(self, inputs):
        k = inputs
        # give padding position 0 as mask
        mask = K.cast(K.equal(k, self.pad_id), dtype='int32')
        mask = 1 - mask
        mask = K.cast(mask, dtype=K.floatx())
        return mask

class SingleSequenceMaskLayer(Layer):
    def __init__(self, pad_id, **kwargs):
        self.pad_id = pad_id
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['pad_id'] = self.pad_id
        return config

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 0:
            raise ValueError('Please input right shape to UsefulWordAutoPointer, received length: %s'%len(input_shape))
        return (input_shape[0], input_shape[1], 1)

    def call(self, inputs):
        k = inputs
        # give padding position 0 as mask
        mask = K.cast(K.equal(k, self.pad_id), dtype='int32')
        mask = 1 - mask
        mask = K.expand_dims(mask, axis=-1)
        mask = K.cast(mask, dtype=K.floatx())# shape: (batch_size, tar_len)
        return mask



get_custom_objects().update({
    'FactPaddingMaskLayer': FactPaddingMaskLayer,
    'PaddingMaskLayer': PaddingMaskLayer,
    'LatentPaddingMaskLayer': LatentPaddingMaskLayer,
    'SelfPadMaskLayer': SelfPadMaskLayer,
    'SequenceMaskLayer': SequenceMaskLayer,
    'SingleSequenceMaskLayer': SingleSequenceMaskLayer,
})
