import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import pandas as pd
import numpy as np

from DataFetcher import DataFetcher, filter_out_val_set
from GRFScaler import GRFScaler
from ModelTester import ModelTester, create_heatmap, normalize_per_component
from GRFImageConverter import GRFImageConverter
from ImageFilter import ImageFilter

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Conv2D, Input
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
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_HatamiModel(train, test, class_dict):
    """Test different settings for the Hatami-Model"""

    num_neurons = 128
    num_filters = [[32, 32]]
    kernel_sizes = [[(3,3), (3,3)]]
    filepath = "models/output/Image/Hatami/GADF+MTF"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_HatamiModel(kernel_sizes=kernel, num_filters=num_filter, neurons=num_neurons, input_shape=(train["affected"]["mtf"].shape[1], train["affected"]["mtf"].shape[2],  train["affected"]["mtf"].shape[3]*4)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_image_model(model, images=["gasf","mtf"], epochs=100, data_dict=train, test_dict=test, logfile="GADFMTF.dat", model_name="GAdF + MTF - Hatami", plot_name="GADFMTF.png")
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    #create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_2DConv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + 2DConv")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


def test_ResNet(train, test, class_dict):
    """Test different settings for the Hatami-Model"""

    #num_neurons = 128
    #num_filters = [[32, 128]]
    #kernel_sizes = [[(3,3), (3,3)]]
    filepath = "models/output/Image/Xception/GASF"
    optimizer = "adam"

    resetRand()
    input_tensor = Input(shape=(train["affected"]["gasf"].shape[1], train["affected"]["gasf"].shape[2],  train["affected"]["gasf"].shape[3]*2))
    #model = ResNet50(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling=None, classes=5)
    model = Xception(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling=None, classes=5)
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
    acc, _, val_acc, _ = tester.test_image_model(model, images=["gasf"], epochs=100, data_dict=train, test_dict=test, logfile="GASF.dat", model_name="GASF - ResNet50", plot_name="GASF.png")

    #create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_2DConv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + 2DConv")
    print("Accuracy {}, ValAccuracy: ,{}".format(acc, val_acc))


def rotate_and_stack(data, images, rot=[1,2,3]):
    keys = ["affected", "non_affected"]
    if "affected_val" in data.keys():
        keys += ["affected_val", "non_affected_val"]
    print(keys)
    for key in keys:
        for image in images:
            for i in rot:
                new = np.copy(data[key][image])
                np.rot90(new, i, axes=[1,2])
                data[key][image]= np.concatenate([data[key][image], new], axis=-1)
        print(data[key][image].shape)


if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    converter = GRFImageConverter()
    converter.enableGpu()
    


    #train_all = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, scaler=scaler, concat=False, val_setp=0.0, include_info=True)
    train, test = fetcher.fetch_data(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)
    conv_args = {
        "num_bins": 20,
         "range": (-1, 1),
         "dims": 3,
         "delay": 4,
         "metric": "euclidean"
    }

    #imgFilter = ImageFilter("median", (7,7))
    gaf_train = converter.convert(train, conversions=["gaf", "mtf"], conv_args=conv_args)
    gaf_test = converter.convert(test, conversions=["gaf", "mtf"], conv_args=conv_args)

    #rotate_and_stack(gaf_train, ["gasf"], [3])
    #rotate_and_stack(gaf_test, ["gasf"], [3])

    gaf_train["label"] = train["label"]
    gaf_train["label_val"] = train["label_val"]
    gaf_test["label"] = test["label"]


    test_HatamiModel(gaf_train, gaf_test, fetcher.get_class_dict())
    #test_ResNet(gaf_train, gaf_test, fetcher.get_class_dict())