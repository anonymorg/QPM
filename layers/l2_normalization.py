# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input




class L2Normalization(Layer):

    def __init__(self, axis = 1, **kwargs):
        self.axis = axis
        super(L2Normalization, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs):

        output = K.l2_normalize(inputs,axis = self.axis)
        return output

    def compute_output_shape(self, input_shape):
        
        return input_shape


def main():
    from keras.layers import Input, Dense

    encoding_dim = 50
    input_dim = 300
    a=np.random.random([5,300])
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim)(input_img) #,
    new_code=L2Normalization()(encoded)

    encoder = Model(input_img, new_code)

    b=encoder.predict(a)


if __name__ == '__main__':
    main()
