
# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import Layer,Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,Multiply,Concatenate,Add,Subtract,Reshape,LeakyReLU
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers import *
#import keras.backend as K
from keras import backend as K
import math
import numpy as np
from tensorflow.keras import regularizers

def kronecker_product(mat1, mat2):
    batch1, n1 = mat1.get_shape().as_list()
    mat1_rsh = K.reshape(mat1, [-1, n1, 1])
    batch2, n2 = mat2.get_shape().as_list()
    mat2_rsh = K.reshape(mat2, [-1, 1, n2])
    return K.reshape(mat1_rsh * mat2_rsh, [-1, n1 * n2])

def kronecker_add(mat1, mat2):
    batch1, n1 = mat1.get_shape().as_list()
    mat1_rsh = K.reshape(mat1, [-1, n1, 1])
    batch2, n2 = mat2.get_shape().as_list()
    mat2_rsh = K.reshape(mat2, [-1, 1, n2])
    return K.reshape(mat1_rsh + mat2_rsh, [-1, n1 * n2])

def kronecker_product3D(tensors):
    tensor1 = tensors[0]
    tensor2 = tensors[1]
    batch1, n1, o1 = tensor1.get_shape().as_list()
    batch2, n2, o2 = tensor2.get_shape().as_list()
    x_list = []
    for ind1 in range(o1):
        for ind2 in range(o2):
            x_list.append(kronecker_product(tensor1[:,:,ind1], tensor2[:,:,ind2]))
    return K.reshape(Concatenate()(x_list), [-1, n1 * n2, o1 * o2])

def kronecker_add3D(tensors):
    tensor1 = tensors[0]
    tensor2 = tensors[1]
    batch1, n1, o1 = tensor1.get_shape().as_list()
    batch2, n2, o2 = tensor2.get_shape().as_list()
    x_list = []
    for ind1 in range(o1):
        for ind2 in range(o2):
            x_list.append(kronecker_add(tensor1[:,:,ind1], tensor2[:,:,ind2]))
    return K.reshape(Concatenate()(x_list), [-1, n1 * n2, o1 * o2])

class QDNN(BasicModel):

    def initialize(self): 

        self.input_amplitude = Input(shape=(4,768,),dtype='float32',name='ling-amplitude')
        self.input_phase = Input(shape=(4,768,), dtype='float32',name='ling-phase')
        self.weight = Input(shape=(1,), dtype='float32',name='weight')
        self.visual_amplitude = Input(shape=(4,2048,),dtype='float32',name='visual-amplitude')
        self.visual_phase = Input(shape=(4,2048,),dtype='float32',name='visual-phase')
        self.dense = Dense(self.opt.nb_classes, activation='softmax', kernel_regularizer= regularizers.l2(self.opt.dense_l2),name='sarcasm-idenfity' ,kernel_initializer='glorot_normal')
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding,name='dropout-embedding')
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs,name='dropout-probs')
        self.projection_task1 = ComplexMeasurement(units = self.opt.measurement_size,name='projection_task1')
        self.projection_task2 = ComplexMeasurement(units = self.opt.measurement_size,name='projection_task2')
        self.sentdense = Dense(3, activation='softmax', kernel_regularizer= regularizers.l2(self.opt.dense_l2), name='sentiment-analysis' ,kernel_initializer='glorot_normal')

    def __init__(self,opt):
        super(QDNN, self).__init__(opt)


    def build(self):
        probs1, probs2 = self.get_representation(self.input_amplitude,self.input_phase,self.weight,self.visual_amplitude,self.visual_phase)
        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            predictions = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", init_criterion = self.opt.init_mode)(probs1)
            output = GetReal()(predictions)
        else:
            sentoutput = self.sentdense(probs1)
            output = self.dense(probs2)
        model = Model([self.input_amplitude,self.input_phase,self.weight,self.visual_amplitude,self.visual_phase], [sentoutput,output])
        return model
    
    def get_representation(self,input_amplitude,input_phase,weight,visual_amplitude,visual_phase):
        
        self.weight = weight
        self.phase_encoded = input_phase
        self.amplitude_encoded = input_amplitude
        self.visamp_encode = visual_amplitude
        self.vispha_encode = visual_phase
        densesize_amp = 20 
        densesize_pha = 20 

        if math.fabs(self.opt.dropout_rate_embedding -1) < 1e-6:
            self.phase_encoded = self.dropout_embedding(self.phase_encoded)
            self.amplitude_encoded = self.dropout_embedding(self.amplitude_encoded)
            self.vispha_encode = self.dropout_embedding(self.vispha_encode)
            self.visamp_encode = self.dropout_embedding(self.visamp_encode)

        dense_phase_encoded = Dense(densesize_pha, activation='relu', name='ling-phase-encode' ,kernel_initializer='glorot_normal')(self.phase_encoded)
        dense_amplitude_encoded = Dense(densesize_amp, activation='relu', name='ling-amplitude-encode' ,kernel_initializer='glorot_normal')(self.amplitude_encoded) 
        vis_dense_phase_encoded = Dense(densesize_pha, activation='relu', name='vis-phase-encode' ,kernel_initializer='glorot_normal')(self.vispha_encode)
        vis_dense_amplitude_encoded = Dense(densesize_amp, activation='relu', name='vis-amplitude-encode' ,kernel_initializer='glorot_normal')(self.visamp_encode) 

        x = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4},name='ling-split-amplitude')(dense_amplitude_encoded)
        angle = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4},name='ling-split-phase')(dense_phase_encoded)
        vis_x = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4},name='vis-split-amplitude')(vis_dense_amplitude_encoded)
        vis_angle = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4},name='vis-split-phase')(vis_dense_phase_encoded)

        kron1 = Lambda(kronecker_product3D)([x[0], x[1]])
        deskron1 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(kron1)
        kron2 = Lambda(kronecker_product3D,name='kron2-k1-con2-k2')([deskron1, x[2]])#400
        deskron2 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(kron2)
        kron3 = Lambda(kronecker_product3D,name='kron3-k2-con3-k3')([deskron2, x[3]])#400

        conseq = []
        conseq.append(kron1)
        conseq.append(kron2)
        conseq.append(kron3)
        kronr = Concatenate(axis=1,name='ling-amp-kornr-concate')(conseq)

        ling_phase_kron1 = Lambda(kronecker_add3D)([angle[0], angle[1]])
        ling_phase_kronr = Lambda(lambda x: x*0.5)(ling_phase_kron1)#400
        deslingaddkron1 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(ling_phase_kron1)
        ling_phase_kron2 = Lambda(kronecker_add3D,name='ling-phase-kron2-k1-con2-k2')([deslingaddkron1, angle[2]])#400
        ling_phase_kron2 = Lambda(lambda x: x*0.5)(ling_phase_kron2)#400
        deslingaddkron2 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(ling_phase_kron2)
        ling_phase_kron3 = Lambda(kronecker_add3D,name='ling-phase-kron3-k2-con3-k3')([deslingaddkron2, angle[3]])#400
        ling_phase_kron3 = Lambda(lambda x: x*0.5)(ling_phase_kron3)#400

        ling_phase_conseq = []
        ling_phase_conseq.append(ling_phase_kron1)
        ling_phase_conseq.append(ling_phase_kron2)
        ling_phase_conseq.append(ling_phase_kron3)
        ling_phase_kronr = Concatenate(axis=1,name='ling-phase-kornr-concate')(ling_phase_conseq)

        vis_kron1 = Lambda(kronecker_product3D)([vis_x[0], vis_x[1]])
        vis_deskron1 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(vis_kron1)
        vis_kron2 = Lambda(kronecker_product3D,name='vis-amp-kron2-k1-con2-k2')([vis_deskron1, vis_x[2]])#400
        vis_deskron2 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(vis_kron2)
        vis_kron3 = Lambda(kronecker_product3D,name='vis-amp-kron3-k2-con3-k3')([vis_deskron2, vis_x[3]])#400

        vis_conseq = []
        vis_conseq.append(vis_kron1)
        vis_conseq.append(vis_kron2)
        vis_conseq.append(vis_kron3)
        vis_kronr = Concatenate(axis=1,name='vis-amp-kornr-concate')(vis_conseq)

        vis_phase_kron1 = Lambda(kronecker_add3D)([vis_angle[0], vis_angle[1]])
        vis_phase_kron1 = Lambda(lambda x: x*0.5)(vis_phase_kron1)#400
        desvisaddkron1 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(vis_phase_kron1)
        vis_phase_kron2 = Lambda(kronecker_add3D,name='vis-phase-kron2-k1-con2-k2')([desvisaddkron1, vis_angle[2]])#400
        vis_phase_kron2 = Lambda(lambda x: x*0.5)(vis_phase_kron2)#400
        desvisaddkron2 = Dense(densesize_amp, activation='relu' ,kernel_initializer='glorot_normal')(vis_phase_kron2)
        vis_phase_kron3 = Lambda(kronecker_add3D,name='vis-phase-kron3-k2-con3-k3')([desvisaddkron2, vis_angle[3]])#400
        vis_phase_kron3 = Lambda(lambda x: x*0.5)(vis_phase_kron3)#400

        vis_phase_conseq = []
        vis_phase_conseq.append(vis_phase_kron1)
        vis_phase_conseq.append(vis_phase_kron2)
        vis_phase_conseq.append(vis_phase_kron3)
        vis_phase_kronr = Concatenate(axis=1,name='vis-phase-kornr-concate')(vis_phase_conseq)

        cosa = Lambda(lambda x:  K.cos(x),name='ling-cos-imaga')(ling_phase_kronr)
        sina = Lambda(lambda x:  K.sin(x),name='ling-sin-imaga')(ling_phase_kronr)
        finalreal = Multiply(name='cosaMulkronr')([cosa,kronr])
        finalimag = Multiply(name='sinaMulkronr')([sina,kronr])

        vis_cosa = Lambda(lambda x:  K.cos(x),name='vis-cos-imaga')(vis_phase_kronr)
        vis_sina = Lambda(lambda x:  K.sin(x),name='vis-sin-imaga')(vis_phase_kronr)
        vis_finalreal = Multiply(name='vis_cosaMulkronr')([vis_cosa,vis_kronr])
        vis_finalimag = Multiply(name='vis_sinaMulkronr')([vis_sina,vis_kronr])

        if self.opt.network_type.lower() == 'complex_mixture':
            [sentence_embedding_real1, sentence_embedding_imag1]= ComplexMixture()([finalreal, finalimag, self.weight])

        elif self.opt.network_type.lower() == 'complex_superposition':
            [sentence_embedding_real2, sentence_embedding_imag2]= ComplexSuperposition()([finalreal, finalimag, self.weight])

        else:
            print('Wrong input network type -- The default mixture network is constructed.')

            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([finalreal, finalimag, self.weight])
            [vis_sentence_embedding_real, vis_sentence_embedding_imag]= ComplexMixture()([vis_finalreal, vis_finalimag, self.weight])

            real_part = Lambda(lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7))([sentence_embedding_real,vis_sentence_embedding_real])
            imag_part = Lambda(lambda x: 0.5*x[0]+0.5*x[1]+K.sqrt(x[0]*x[1]+1e-7))([sentence_embedding_imag,vis_sentence_embedding_imag])

        if self.opt.network_type == "ablation" and self.opt.ablation == 1:
            sentence_embedding_real = Flatten()(sentence_embedding_real)
            sentence_embedding_imag = Flatten()(sentence_embedding_imag)
            probs = [sentence_embedding_real, sentence_embedding_imag]
        else:
            
            probs_task1 =  self.projection_task1([real_part, imag_part])
            probs_task2 =  self.projection_task2([real_part, imag_part])
            
            probs_task1 = Dense(512, activation='relu',kernel_initializer='glorot_normal')(probs_task1)
            probs_task1= self.dropout_probs(probs_task1)
            probs_task1 = Dense(128, activation='relu',kernel_initializer='glorot_normal')(probs_task1)
            probs_task1= self.dropout_probs(probs_task1)

            probs_task2 = Dense(512, activation='relu',kernel_initializer='glorot_normal')(probs_task2)
            probs_task2= self.dropout_probs(probs_task2)
            probs_task2 = Dense(128, activation='relu',kernel_initializer='glorot_normal')(probs_task2)
            probs_task2= self.dropout_probs(probs_task2)
        return probs_task1,probs_task2




