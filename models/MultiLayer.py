import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from DataFetcher import DataFetcher, filter_out_val_set
from GRFScaler import GRFScaler
from ModelTester import ModelTester
from ModelTester import create_heatmap

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Activation, Add, BatchNormalization, MaxPooling1D, Flatten, Dropout
#from tensorflow.keras.optimizers import Adam

RAND_SEED = 1
from tensorflow import random as tfrand
tfrand.set_seed(RAND_SEED)
from numpy.random import seed
seed(RAND_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.keras import backend


def resetRand():
    """Resets the random number generators to the inital seed."""
    backend.clear_session()
    tfrand.set_seed(RAND_SEED)
    seed(RAND_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'



def get_MultiLayerModel(input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with multiple hidden dense layer."""
    
    model = Sequential()   
    model.add(Input(shape=(input_shape,)))  
    model.add(Dense(20, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dense(20, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dense(20, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_MultiLayerModel(train, test, class_dict):
    """Test different settings for the MultiLayer-MLP"""

    filepath = "models/output/MultiLayer/Dense/Experiments"
    optimizer = "adam"

    resetRand()
    model = get_MultiLayerModel(train["affected"].shape[1]*2) 
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    tester.save_model_plot(model, "MultiLayer.png")
    acc, _, val_acc, _ = tester.test_model(model, data_dict=train, epochs=100, test_dict=test, logfile="MultiLayer.dat", model_name="MultiLayer", plot_name="MultiLayer.png")

    print("Accuracy - Val-Accuracy, {}, {}".format(acc, val_acc))


def get_MultiLayerConvModel(input_shape, dilation_rate=1, kernel_sizes=[3, 3, 3], num_filters=[32, 32, 16], activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with multiple 1D-convolutional layers."""

    model = Sequential()
    model.add(Conv1D(filters=num_filters[0], kernel_size=(kernel_sizes[0]), strides=1, dilation_rate=dilation_rate, padding="same", activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))
    model.add(Dropout(rate=0.2))

    model.add(Conv1D(filters=num_filters[1], kernel_size=(kernel_sizes[1]), strides=1, padding="same", activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(Conv1D(filters=num_filters[2], kernel_size=(kernel_sizes[2]), strides=1, padding="same", activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(Flatten())    
    model.add(Dense(25, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(rate=0.2))
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))

    return model


def test_MultiLayerConvModel(train, test, class_dict):
    """Test different settings for the MultiLayerConv-MLP"""

    filepath = "models/output/MultiLayer/Conv/dilated/5/DropOut"
    optimizer = "adam"

    resetRand()
    model = get_MultiLayerConvModel(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), kernel_sizes=[9,5,3], num_filters=[32, 32, 16], dilation_rate=5)
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    tester.save_model_plot(model, "MultiLayerConv.png")
    acc, _, val_acc, _ = tester.test_model(model, data_dict=train, epochs=100, test_dict=test, logfile="MultiLayerConv.dat", model_name="MultiLayerConv", plot_name="MultiLayerConv.png")

    print("Accuracy - Val-Accuracy, {}, {}".format(acc, val_acc))




if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))

    train, test = fetcher.fetch_data(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)
    #train_all = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN", stepsize=1, averageTrials=False, scaler=scaler, concat=False, val_setp=0.0, include_info=True)
    #train_new = filter_out_val_set(train_all, train)
    test_MultiLayerConvModel(train, test, fetcher.get_class_dict())
