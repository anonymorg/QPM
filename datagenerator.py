import keras
import math
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten


class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=True):
        if 1 == dataset:
            self.lingamppath = 'mustard_data/lingdata/'
            self.lingphapath = 'dataset2/phase/phase'
            self.imgamppath = 'mustard_data/visdata/'
            self.imgphapath = 'mustard_data/phase/phase'
            self.sarlabelpath = 'mustard_data/label/label.txt'
            self.sentlabelpath = 'mustard_data/label/sentlabel.txt'
            self.datalen = 600
            self.indexes = np.arange(600)
            self.splpoint = 599
        else:
            self.lingamp = 'mustard_data/lingdata/'

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.floor(self.datalen / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        lingamp, lingpha, visamp, vispha, sarlabel, sentlabel = self.gen_ling_amp_pha_sarl_sentl(batch_indexs)
        weight = self.getmustardweight()
        x_batch = {'ling-amplitude':lingamp ,'ling-phase':lingpha, 'visual-amplitude':visamp, 'visual-phase':vispha, 'weight':weight}
        y_batch = {'sarcasm-idenfity':sarlabel, 'sentiment-analysis':sentlabel}

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def getmustardweight(self):
        weight = [1]
        allWeights = []
        for i in range(self.batch_size):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights

    def gen_ling_amp_pha_sarl_sentl(self, batch_datas):
        tmp_amp = []
        tmp_phase = []
        tmp_vis_amp = []
        tmp_vis_phase = []
        for i in batch_datas:
            utt_con = np.loadtxt(\
                        self.lingamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_amp.append(utt_con[-4:])
            
            phase = np.loadtxt(\
                        self.lingphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_phase.append(phase[-4:])

            vis_utt_con = np.loadtxt(\
                        self.imgamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_vis_amp.append(vis_utt_con[-4:])
                
            vis_phase = np.loadtxt(\
                        self.imgphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_vis_phase.append(vis_phase[-4:])
        
        label = np.loadtxt(\
                        self.sarlabelpath,\
                        dtype=int, \
                        delimiter=",")
        label = label.astype(np.int)
        label = [label[k] for k in batch_datas]

        sentlabel = np.loadtxt(\
                        self.sentlabelpath,\
                        dtype=int, \
                        delimiter=",")
        sentlabel = sentlabel.astype(np.int)
        sentlabel = [sentlabel[k] for k in batch_datas]
        tmp_amp = np.array(tmp_amp)
        tmp_phase = np.array(tmp_phase)
        tmp_vis_amp = np.array(tmp_vis_amp)
        tmp_vis_phase = np.array(tmp_vis_phase)
        label = np.array(label)
        sentlabel = np.array(sentlabel)

        return tmp_amp, tmp_phase, tmp_vis_amp, tmp_vis_phase, label, sentlabel

class EvalDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, shuffle=True):
        if 1 == dataset:
            self.lingamppath = 'mustard_data/lingdata/'
            self.lingphapath = 'dataset2/phase/phase'
            self.imgamppath = 'mustard_data/visdata/'
            self.imgphapath = 'mustard_data/phase/phase'
            self.sarlabelpath = 'mustard_data/label/label.txt'
            self.sentlabelpath = 'mustard_data/label/sentlabel.txt'
            self.datalen = 600
            self.evallen = 50
            self.indexes = np.arange(600,650)
            self.splpoint = 599
        else:
            self.lingamp = 'mustard_data/lingdata/'
        self.shuffle = shuffle

    def __len__(self):
        return self.evallen

    def __getitem__(self, index):
        batch_indexs = self.indexes
        lingamp, lingpha, visamp, vispha, sarlabel, sentlabel = self.gen_ling_amp_pha_sarl_sentl(batch_indexs)
        weight = self.getmustardweight()
        x_batch = {'ling-amplitude':lingamp ,'ling-phase':lingpha, 'visual-amplitude':visamp, 'visual-phase':vispha, 'weight':weight}
        y_batch = {'sarcasm-idenfity':sarlabel, 'sentiment-analysis':sentlabel}

        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def getmustardweight(self):
        weight =[1]
        allWeights = []
        for i in range(self.evallen):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights

    def gen_ling_amp_pha_sarl_sentl(self, batch_datas):
        tmp_amp = []
        tmp_phase = []
        tmp_vis_amp = []
        tmp_vis_phase = []
        for i in batch_datas:
            utt_con = np.loadtxt(\
                        self.lingamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_amp.append(utt_con[-4:])
            
            phase = np.loadtxt(\
                        self.lingphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_phase.append(phase[-4:])

            vis_utt_con = np.loadtxt(\
                        self.imgamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_vis_amp.append(vis_utt_con[-4:])
            
            vis_phase = np.loadtxt(\
                        self.imgphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_vis_phase.append(vis_phase[-4:])
        
        label = np.loadtxt(\
                        self.sarlabelpath,\
                        dtype=int, \
                        delimiter=",")
        label = label.astype(np.int)
        label = [label[k] for k in batch_datas]

        sentlabel = np.loadtxt(\
                        self.sentlabelpath,\
                        dtype=int, \
                        delimiter=",")
        sentlabel = sentlabel.astype(np.int)
        sentlabel = [sentlabel[k] for k in batch_datas]
        tmp_amp = np.array(tmp_amp)
        tmp_phase = np.array(tmp_phase)
        tmp_vis_amp = np.array(tmp_vis_amp)
        tmp_vis_phase = np.array(tmp_vis_phase)
        label = np.array(label)
        sentlabel = np.array(sentlabel)
        return tmp_amp, tmp_phase, tmp_vis_amp, tmp_vis_phase, label, sentlabel

def getValidData(dataset):
    if 1 == dataset:
        lingamppath = 'mustard_data/lingdata/'
        lingphapath = 'dataset2/phase/phase'
        imgamppath = 'mustard_data/visdata/'
        imgphapath = 'mustard_data/phase/phase'
        sarlabelpath = 'mustard_data/label/label.txt'
        sentlabelpath = 'mustard_data/label/sentlabel.txt'
        datalen = 600
        evallen = 50
        valStart = 650
        vallen = 90
        indexes = np.arange(600,650)
    else:
        lingamp = 'mustard_data/lingdata/'

    def gen_ling_amp_pha_sarl_sentl(batch_datas):
        tmp_amp = []
        tmp_phase = []
        tmp_vis_amp = []
        tmp_vis_phase = []
        for i in batch_datas:
            utt_con = np.loadtxt(\
                        lingamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_amp.append(utt_con[-4:])
            
            phase = np.loadtxt(\
                        lingphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_phase.append(phase[-4:])

            vis_utt_con = np.loadtxt(\
                        imgamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_vis_amp.append(vis_utt_con[-4:])
            
            vis_phase = np.loadtxt(\
                        imgphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_vis_phase.append(vis_phase[-4:])
        
        label = np.loadtxt(\
                        sarlabelpath,\
                        dtype=int, \
                        delimiter=",")
        label = label.astype(np.int)
        label = [label[k] for k in batch_datas]

        sentlabel = np.loadtxt(\
                        sentlabelpath,\
                        dtype=int, \
                        delimiter=",")
        sentlabel = sentlabel.astype(np.int)
        sentlabel = [sentlabel[k] for k in batch_datas]

        tmp_amp = np.array(tmp_amp)
        tmp_phase = np.array(tmp_phase)
        tmp_vis_amp = np.array(tmp_vis_amp)
        tmp_vis_phase = np.array(tmp_vis_phase)
        label = np.array(label)
        sentlabel = np.array(sentlabel)

        return tmp_amp, tmp_phase, tmp_vis_amp, tmp_vis_phase, label, sentlabel

    def getmustardweight():
        weight = [1]
        allWeights = []
        for i in range(vallen):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights

    batch_indexs = indexes
    lingamp, lingpha, visamp, vispha, sarlabel, sentlabel = gen_ling_amp_pha_sarl_sentl(indexes)
    weight = getmustardweight()
    return lingamp, lingpha, visamp, vispha, weight, sarlabel, sentlabel


def getEvalData(dataset):
    if 1 == dataset:
        lingamppath = 'mustard_data/lingdata/'
        lingphapath = 'dataset2/phase/phase'
        imgamppath = 'mustard_data/visdata/'
        imgphapath = 'mustard_data/phase/phase'
        sarlabelpath = 'mustard_data/label/label.txt'
        sentlabelpath = 'mustard_data/label/sentlabel.txt'
        datalen = 600
        evallen = 50
        valStart = 600
        vallen = 40
        indexes = np.arange(650,690)
    else:
        lingamp = 'mustard_data/lingdata/'

    def gen_ling_amp_pha_sarl_sentl(batch_datas):
        tmp_amp = []
        tmp_phase = []
        tmp_vis_amp = []
        tmp_vis_phase = []
        for i in batch_datas:
            utt_con = np.loadtxt(\
                        lingamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_amp.append(utt_con[-4:])
            
            phase = np.loadtxt(\
                        lingphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_phase.append(phase[-4:])

            vis_utt_con = np.loadtxt(\
                        imgamppath+str(i+1), \
                        dtype=float, \
                        delimiter=",")
            tmp_vis_amp.append(vis_utt_con[-4:])
            
            vis_phase = np.loadtxt(\
                        imgphapath+str(i)+'.txt',\
                        dtype=float, \
                        delimiter=",")
            tmp_vis_phase.append(vis_phase[-4:])
        
        label = np.loadtxt(\
                        sarlabelpath,\
                        dtype=int, \
                        delimiter=",")
        label = label.astype(np.int)
        label = [label[k] for k in batch_datas]

        sentlabel = np.loadtxt(\
                        sentlabelpath,\
                        dtype=int, \
                        delimiter=",")
        sentlabel = sentlabel.astype(np.int)
        sentlabel = [sentlabel[k] for k in batch_datas]
        tmp_amp = np.array(tmp_amp)
        tmp_phase = np.array(tmp_phase)
        tmp_vis_amp = np.array(tmp_vis_amp)
        tmp_vis_phase = np.array(tmp_vis_phase)
        label = np.array(label)
        sentlabel = np.array(sentlabel)
        return tmp_amp, tmp_phase, tmp_vis_amp, tmp_vis_phase, label, sentlabel

    def getmustardweight():
        weight = [1]
        allWeights = []
        for i in range(vallen):
            allWeights.append(weight)
        allWeights = np.array(allWeights)
        return allWeights

    batch_indexs = indexes
    lingamp, lingpha, visamp, vispha, sarlabel, sentlabel = gen_ling_amp_pha_sarl_sentl(indexes)
    weight = getmustardweight()
    return lingamp, lingpha, visamp, vispha, weight, sarlabel, sentlabel



if __name__ == '__main__':
    training_generator = TrainDataGenerator(dataset=1, batch_size=16, shuffle=True)
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape =(4,768)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax' ,name='sarcasm-idenfity' ,kernel_initializer='glorot_normal'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(training_generator, epochs=20, max_queue_size=10, workers=1)
