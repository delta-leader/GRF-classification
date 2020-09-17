import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import pandas as pd
import numpy as np

from DataFetcher import DataFetcher, filter_out_val_set, filter_out_subjects
from GRFScaler import GRFScaler
from ModelTester import ModelTester, create_heatmap, normalize_per_component
from GRFImageConverter import GRFImageConverter
from ImageFilter import ImageFilter

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Conv2D, Conv1D
from tensorflow.keras.applications import ResNet50, Xception
#from tensorflow.keras.optimizers import Adam

RAND_SEED = 1
from tensorflow import random as tfrand
from numpy.random import seed
tfrand.set_seed(RAND_SEED)
seed(RAND_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.keras import backend


def resetRand():
    """Resets the random number generators to the inital seed."""
    backend.clear_session()
    tfrand.set_seed(RAND_SEED)
    seed(RAND_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def get_Model(input_shape, num_filters, kernel_size, neurons=20, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a model to classify Calcaneus vs Ankle"""
    
    model = Sequential()   
    model.add(Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    model.add(Dropout(rate=0.1))
    model.add(Dense(2, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_Model(train, test, class_dict):
    """Test different settings for the Calcaneus-Model"""

    num_filters = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [2, 3, 5, 7, 9, 11]
    num_neurons = 30
    filepath = "models/output/Calcaneus/Conv1D"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_Model(kernel_size=kernel, num_filters=num_filter, neurons=num_neurons, input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="Conv_K{}_F{}.dat".format(kernel, num_filter), model_name="Conv1D - Calcaneus", plot_name="Conv_K{}_F{}.png".format(kernel, num_filter))
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_Conv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="Conv1D - Calcaneus")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


def get_HatamiModel(input_shape, kernel_sizes=[(3,3), (3,3)], num_filters=[32, 32], neurons=128, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with a 2D convolutional layer before the hidden dense layer."""
    
    model = Sequential()   
    model.add(Conv2D(filters=num_filters[0], kernel_size=kernel_sizes[0], strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=num_filters[1], kernel_size=kernel_sizes[1], strides=1, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_HatamiModel(train, test, class_dict):
    """Test different settings for the Hatami-Model"""

    num_neurons = 128
    num_filters = [[32, 32]]
    kernel_sizes = [[(3,3), (3,3)]]
    filepath = "models/output/Calcaneus/Hatami/MTF/Affected/BOOST"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_HatamiModel(kernel_sizes=kernel, num_filters=num_filter, neurons=num_neurons, input_shape=(train["affected"]["mtf"].shape[1], train["affected"]["mtf"].shape[2],  train["affected"]["mtf"].shape[3])) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_image_model(model,  useNonAffected=False, images=["mtf"], epochs=100, data_dict=train, test_dict=test, logfile="MTF.dat", model_name="MTF - Hatami", plot_name="MTF.png", boost=True)
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    #create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_2DConv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + 2DConv")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))





if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    converter = GRFImageConverter()
    converter.enableGpu()
    


    #train_all = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, scaler=scaler, concat=False, val_setp=0.0, include_info=True)
    train_calc = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN", stepsize=1, averageTrials=False, scaler=scaler, concat=False, val_setp=0.2, include_info=True)
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True)
    
    #print(train_calc)
    train_calc, dict_calc = filter_out_subjects(train_calc, train["info_val"], fetcher.get_class_dict(), ["A", "C"])
    #print(train_calc)
    print(train_calc["label"].shape)
    print(train_calc["label_val"].shape)
    info = train_calc["info"]
    info_val = train_calc["info_val"]
    del train_calc['info_val']
    del train_calc["info"]

    
    conv_args = {
        "num_bins": 32,
         "range": (-1, 1),
         "dims": 3,
         "delay": 4,
         "metric": "euclidean"
    }

    #imgFilter = ImageFilter("median", (7,7))
    mtf_calc = converter.convert(train_calc, conversions=["mtf"], conv_args=conv_args)
    #gaf_test = converter.convert(test, conversions=["gaf", "rcp"], conv_args=conv_args)

    #rotate_and_stack(gaf_train, ["gasf"], [3])
    #rotate_and_stack(gaf_test, ["gasf"], [3])

    mtf_calc["label"] = train_calc["label"]
    mtf_calc["label_val"] = train_calc["label_val"]
    mtf_calc["info"] = info
    mtf_calc["info_val"] = info_val
    #gaf_test["label"] = test["label"]

    test_HatamiModel(mtf_calc, None, dict_calc)
