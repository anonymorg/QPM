# -*- coding: utf-8 -*-


import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input


class reshape(Layer):

    def __init__(self, shape,**kwargs):
        super(reshape, self).__init__(**kwargs)
        self.shape = shape


    def build(self, input_shape):
        super(reshape, self).build(input_shape)

    def call(self, inputs):



        output = K.reshape(inputs,self.shape)
        return output

    def compute_output_shape(self, input_shape):
        return self.shape


def main():
    from keras.layers import Input, Dense


    
    encoding_dim = 50
    input_dim = 300        
    a=np.random.random([10,300])
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim)(input_img) #, 
    new_code=reshape((-1,500))(encoded)
    
    encoder = Model(input_img, new_code)
    
    b=encoder.predict(a)


if __name__ == '__main__':
    main()
