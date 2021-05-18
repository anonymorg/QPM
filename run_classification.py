# -*- coding: utf-8 -*-
import tensorflow as tf
from params import Params
from models import representation as models
from dataset import classification as dataset
from tools import units
from tools.save import save_experiment
from loadmydata import *
import itertools
import argparse
import tensorflow.keras.backend as K
import numpy as np 
from keras.utils import plot_model
from datagenerator import TrainDataGenerator , getValidData, getEvalData
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,balanced_accuracy_score, confusion_matrix
import logging
import time
import keras

gpu_count = len(units.get_available_gpus())
dir_path,global_logger = units.getLogger()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def EvalModel(dataset, model):
    result = []
    eval_data = getEvalData(dataset=1)
    sent_val_predict, val_predict = model.predict([eval_data[0], eval_data[1], eval_data[4], eval_data[2], eval_data[3]])
    val_predict = 1-np.argmax(val_predict, -1)
    sent_val_predict = np.argmax(sent_val_predict, -1)
    val_targ = eval_data[5]
    sent_val_targ = eval_data[6]
    if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
        val_targ = np.argmax(val_targ, -1)
    if len(sent_val_targ.shape) == 2 and sent_val_targ.shape[1] != 1:
        sent_val_targ = np.argmax(sent_val_targ, -1)
    tn, fp, fn, tp = confusion_matrix(val_targ, val_predict).ravel()
    _specificity = tn / (tn+fp)
    sent_val_bacc = balanced_accuracy_score(sent_val_targ, sent_val_predict)
    sent_val_f1 = f1_score(sent_val_targ, sent_val_predict, average='micro')
    sent_val_recall = recall_score(sent_val_targ, sent_val_predict, average='micro')
    sent_val_precision = precision_score(sent_val_targ, sent_val_predict, average='micro')
    sent_val_acc = accuracy_score(sent_val_targ, sent_val_predict) 
    result.append(sent_val_bacc)
    result.append(sent_val_acc)
    result.append(sent_val_f1)
    result.append(sent_val_precision)
    result.append(sent_val_recall)
    _val_bacc = balanced_accuracy_score(val_targ, val_predict)
    _val_f1 = f1_score(val_targ, val_predict, average='micro')
    _val_recall = recall_score(val_targ, val_predict, average='micro')
    _val_precision = precision_score(val_targ, val_predict, average='micro')
    _val_acc = accuracy_score(val_targ, val_predict)
    result.append(_val_bacc)
    result.append(_val_acc)
    result.append(_val_f1)
    result.append(_val_precision)
    result.append(_val_recall)
    result.append(_specificity)
    print('eval result:')
    print("- val_bacc: %f - val_acc: %f  — val_f1: %f — val_precision: %f — val_recall: %f — specificity: %f" % (_val_bacc, _val_acc, _val_f1, _val_precision, _val_recall, _specificity))
    print("- sent_val_bacc: %f - sent_val_acc: %f  — sent_val_f1: %f — sent_val_precision: %f — sent_val_recall: %f" % (sent_val_bacc, sent_val_acc, sent_val_f1, sent_val_precision, sent_val_recall))

    return result

class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        sent_val_predict, val_predict = self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[4], self.validation_data[2], self.validation_data[3]])
        val_predict = 1-np.argmax(val_predict, -1)
        sent_val_predict = np.argmax(sent_val_predict, -1)
        val_targ = self.validation_data[5]
        sent_val_targ = self.validation_data[6]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)
        if len(sent_val_targ.shape) == 2 and sent_val_targ.shape[1] != 1:
            sent_val_targ = np.argmax(sent_val_targ, -1)

        tn, fp, fn, tp = confusion_matrix(val_targ, val_predict).ravel()
        _specificity = tn / (tn+fp)
        sent_val_bacc = balanced_accuracy_score(sent_val_targ, sent_val_predict)
        sent_val_f1 = f1_score(sent_val_targ, sent_val_predict, average='micro')
        sent_val_recall = recall_score(sent_val_targ, sent_val_predict, average='micro')
        sent_val_precision = precision_score(sent_val_targ, sent_val_predict, average='micro')
        sent_val_acc = accuracy_score(sent_val_targ, sent_val_predict) 
        logs['sent_val_acc'] = sent_val_acc
        logs['sent_val_f1'] = sent_val_f1
        logs['sent_val_recall'] = sent_val_recall
        logs['sent_val_precision'] = sent_val_precision

        _val_bacc = balanced_accuracy_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        _val_acc = accuracy_score(val_targ, val_predict) 
        logs['val_acc'] = _val_acc
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        logs['specificity'] = _specificity
        print("- val_bacc: %f - val_acc: %f  — val_f1: %f — val_precision: %f — val_recall: %f — specificity: %f" % (_val_bacc, _val_acc, _val_f1, _val_precision, _val_recall, _specificity))
        print("- sent_val_bacc: %f - sent_val_acc: %f  — sent_val_f1: %f — sent_val_precision: %f — sent_val_recall: %f" % (sent_val_bacc, sent_val_acc, sent_val_f1, sent_val_precision, sent_val_recall))

        logger.info("- val_bacc: %f - val_acc: %f  — val_f1: %f — val_precision: %f — val_recall: %f — specificity: %f" % (_val_bacc, _val_acc, _val_f1, _val_precision, _val_recall, _specificity))
        logger.info("- sent_val_bacc: %f - sent_val_acc: %f  — sent_val_f1: %f — sent_val_precision: %f — sent_val_recall: %f" % (sent_val_bacc, sent_val_acc, sent_val_f1, sent_val_precision, sent_val_recall))

        return


def run(params,reader,logger):
    params = dataset.process_embedding(reader,params)
    qdnn = models.setup(params)
    model = qdnn.getModel()

    print(model.summary())

    model.compile(loss = {'sarcasm-idenfity' : params.loss,
                        'sentiment-analysis' : params.loss},
                        loss_weights={'sarcasm-idenfity':0.65, 'sentiment-analysis': 0.35},
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr), 
                    metrics=['accuracy'])

    training_generator = TrainDataGenerator(dataset=1, batch_size=params.batch_size, shuffle=True)
    val_x, val_phase, val_vis_x, val_vis_phase, val_w, val_y, val_senty = getValidData(dataset=1)
    history = model.fit_generator(training_generator,epochs= params.epochs,\
        callbacks=[Metrics(valid_data=(val_x, val_phase, val_vis_x, val_vis_phase, val_w, val_y, val_senty))])
    evalresult = EvalModel(dataset=1,model=model)
    
    return history, evalresult


grid_parameters ={
        "dataset_name":["SST_2"],
        "wordvec_path":["glove/glove.6B.50d.txt"],
        "loss": ["categorical_crossentropy"],
        "optimizer":["adam"],
        "batch_size":[48],
        "activation":["sigmoid"],
        "amplitude_l2":[0.0000005],
        "phase_l2":[0.0000005],
        "dense_l2":[0.0000005],
        "measurement_size" :[1000],
        "lr" : [0.001],
        "epochs" : [60],
        "dropout_rate_embedding" : [0.5],
        "dropout_rate_probs" : [0.65],
        "ablation" : [1],
    }
if __name__=="__main__":
    logging.basicConfig(filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("explog511.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    parameters= parameters[::-1]
    logger.info("parameters: %s",parameters)
    params = Params()
    config_file = 'config/qdnn.ini'
    params.parse_config(config_file)    
    for parameter in parameters:
        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
        if old_dataset != params.dataset_name:
            print("switch {} to {}".format(old_dataset,params.dataset_name))
            reader=dataset.setup(params)
            params.reader = reader
        history,evalresult = run(params,reader,logger)
        logger.info("history: %s",str(history))
        logger.info("- eval_bacc: %f - eval_acc: %f  — eval_f1: %f — eval_precision: %f — eval_recall: %f — specificity: %f" % (evalresult[5], evalresult[6], evalresult[7], evalresult[8], evalresult[9], evalresult[10]))
        logger.info("- sent_eval_bacc: %f - sent_eval_acc: %f  — sent_eval_f1: %f — sent_eval_precision: %f — sent_eval_recall: %f" % (evalresult[0], evalresult[1], evalresult[2], evalresult[3],evalresult[4]))
        K.clear_session()


