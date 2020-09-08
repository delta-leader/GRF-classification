import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
from collections import namedtuple

from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from ModelTester import ModelTester, create_heatmap, resetRand

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, SeparableConv1D, concatenate


def wandb_init():
    """Initalizes the W&B init() call and returns the created configuration file.

    ----------
    Returns:
    config: wandb.config
        The configuration file to be used.
    """

    wandb.init(project="diplomarbeit", config=create_config())
    return wandb.config


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "1DCNN Sweep 1Layer",
        "method": "grid",
        "description": "Find the optimal number of layers/neurons",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "layers": {
                "value": 1
            },
            "filters0": {
                "values": [10, 20, 30, 40, 50]
                #"min": 40,
                #"max": 190
            },
            #"filters1": {
            #    "values": [10, 20, 30, 40, 50]
            #    #"min": 40,
            #    #"max": 190
            #},
            #"batch_normalization": {
            #    "distribution": "categorical",
            #    "values": [True, False]
            #},
            #"dropout": {
            #    "distribution": "uniform",
            #    "min": 0.1,
            #    "max": 0.5
            #}
            "kernel0": {
                "values": [(3), (5), (7), (9), (11), (13), (15)]
                #"min": 40,
                #"max": 190
            },
            #"kernel1": {
            #    "values": [(3), (5), (7), (9), (11), (13), (15)]
            #    #"min": 40,
            #    #"max": 190
            #},
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the 1DCNN."""

    config = {
        "layers": 2,
        "filters0": 32,
        "filters1": 32,
        "filters2": 32,
        "kernel0": (8),
        "kernel1": (5),
        "kernel2": (3),
        "stride0": 1,
        "stride1": 1,
        "stride2": 1,
        "dilation0": 1,
        "dilation1": 1,
        "dilation2": 1,
        "batch_normalization": False,
        "max_pooling": None,
        "pooling_stride": None,
        "mlp_neurons": 50,
        "cnn_dropout": None,
        "mlp_dropout": None,
        "separable": False,
        "skipConnections": False,
        "padding": "same",
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "batch_size": 32,
        "epochs": 100
    }

    return config

def create_1DCNN(input_shape, config):
    """Creates a 1-dimensional CNN according to the specifications in 'config'

    Parameters:
    input_shape : tupel of int
        Specifies the size of the Input (aka the first) layer.

    config: Either wandb.config or namedtuple
        Contains the specifications for the model (i.e. number of layers, number of neurons, rate of dropout, etc.).

    ----------
    model : tensorflow.keras.model
        The model created according to the specifications.
    """
    
    #model = Sequential()
    #model.add(Input(shape=input_shape))
    print(input_shape)
    input_layer = Input(shape=input_shape)

    # adds BatchNormalization, MaxPooling & Dropout if specified
    def finish_layer(conv, config):
        if config.batch_normalization:
            conv = BatchNormalization()(conv)
            #model.add(BatchNormalization())
        if config.max_pooling is not None:
            conv = MaxPooling1D(pool_size=config.max_pooling, strides=config.pooling_stride, padding=config.padding)(conv)
        if config.cnn_dropout is not None:
            conv = Dropout(rate=config.cnn_dropout)(conv)
            #model.add(Dropout(rate=config.cnn_dropout))
        return conv

    def add_conv_layer(conv, config, layer):
        if config.separable:
            conv = SeparableConv1D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), strides=getattr(config, "stride{}".format(layer)), dilation_rate=getattr(config, "dilation{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding)(conv)
            #model.add(SeparableConv1D(filters=getattr(config, "filter{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), strides=getattr(config, "stride{}".format(layer)), dilation_rate=getattr(config, "dilation{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer))
        else:
            conv = Conv1D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), strides=getattr(config, "stride{}".format(layer)), dilation_rate=getattr(config, "dilation{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding)(conv)
        return conv

    # add convolutional layers
    conv = input_layer
    skip_modules = []
    for layer in range(config.layers):
        conv = add_conv_layer(conv, config, layer)
        conv = finish_layer(conv, config)
        if config.skipConnections:
            skip_modules.append(conv)
        #add_conv_layer(model, config, layer)
        #finish_layer(model, config)
    
    # dense layers
    
    if config.skipConnections:
        conv = concatenate([Flatten()(x) for x in ([input_layer]+skip_modules)])
    else:
        conv = Flatten()(conv)

    conv = Dense(config.mlp_neurons, activation=config.activation, kernel_regularizer=config.regularizer)(conv)
    #model.add(Flatten())
    #model.add(Dense(config.mlp_neurons, activation=config.activation, kernel_regularizer=config.regularizer))
    if config.mlp_dropout is not None:
        conv = Dropout(rate=config.mlp_dropout)(conv)
        #model.add(Dropout(rate=config.mlp_dropout))

    out = Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer)(conv)
    #model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))

    model = Model(input_layer, out)
    
    return model


def validate_1DCNN(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the 1DCNN.
    Two modes are available:
    'sweep' == True -> performs a sweep of hyperparameters according to the specified sweep-configuration.
    'sweep' == False -> Performs a single training and evaluation (on test and validation set) according to the configured settings. Includes creationg of plots and confusion matrices.

    Parameters:
    train : dict
        Containing the GRF-data for training and validation.
    
    test : dict, default=None
        Containing the GRF-data for the test-set.
    
    class_dict: dict, default=None
        Dictionary that maps the numbered labels to names. Used to create the confusion matrix.

    seep : bool, default=False
        If true performs a hyperparameter sweep using W&B according to the specified sweep-configuration (using only the validation-set)
        Otherwise a local training and evalution run is performed, providing the results for both validation- and test-set.
    """
      
    if sweep:
        sweep_config = create_sweep_config()
        tester = ModelTester(class_dict=class_dict) 

        def train_MLP():
            config = wandb_init()
            resetRand()
            model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=train_MLP)
    
    else:
        filepath = "models/output/MLP/WandB/CNN"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, optimizer=config.optimizer, class_dict=class_dict)
        resetRand()
        model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "MLP_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, logfile="MLP_1L_N100.dat", model_name="MLP - 1 Hidden Layer", plot_name="MLP_1L_N100.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))


def get_ConvSkipModel(kernel_size, num_filters, neurons, input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with a convolutional layer using skip connections to feed the input directly into the hidden dense layer.
    Basically the hidden dense layer uses both the output from the convolutional layer and the unmodified data as input."""
    
    # branch 1
    input_layer = Input(shape=input_shape)
    conv = Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer)(input_layer)
    conv = Flatten()(conv)

    # branch 2
    flattened_input = Flatten()(input_layer)
               
    # join branches
    concatenated = concatenate([conv, flattened_input])
    concatenated = Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer)(concatenated)
    #concatenated = Dropout(rate=0.2)(concatenated)
    out = Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer) (concatenated)
    model = Model(input_layer, out)
    
    return model

def test_ConvSkipModel(train, test, class_dict, train_norm=None, test_norm=None):
    """Test different settings for the ConvSkipModel-MLP"""

    #num_neurons = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    num_neurons = 50
    num_filters = [8, 16, 32, 64, 128, 256]# 512, 1024]
    kernel_sizes = [2, 3, 5, 7, 9, 11, 13, 15, 21, 33]
    filepath = "models/output/MLP/Conv/Skip/test"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_ConvSkipModel(kernel, num_filter, num_neurons, (train["affected"].shape[1], train["affected"].shape[2]*2)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="MLP_ConvSkip_K{}_F{}.dat".format(kernel, num_filter), model_name="MLP + Conv (with skip-connections)", plot_name="MLP_ConvSkip_K{}_F{}.png".format(kernel, num_filter))
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_ConvSkip_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + Conv (with skip-connections)")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


def get_Conv2DModel(kernel_size, num_filters, neurons, input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with a 2D convolutional layer before the hidden dense layer."""
    
    model = Sequential()   
    model.add(Conv2D(filters=num_filters, kernel_size=kernel_size, strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    #model.add(SeparableConv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer))
    #model.add(Dropout(rate=0.1))
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_Conv2DModel(train, test, class_dict):
    """Test different settings for the 2D-ConvLayer-MLP"""

    #num_neurons = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    num_neurons = 50
    num_filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    kernel_sizes = [(3,1), (5,1), (7,1), (9,1), (11,1), (13,1)]
    filepath = "models/output/MLP/Conv/2D/Tx1xS/NonAveraged"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_Conv2DModel(kernel, num_filter, num_neurons, (train["affected"].shape[1], 1,  train["affected"].shape[2]*2)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_model(model, shape="2D_Tx1xS", data_dict=train, test_dict=test, logfile="MLP_2DConv_K{}_F{}.dat".format(kernel, num_filter), model_name="MLP + 2DConv", plot_name="MLP_2DConv_K{}_F{}.png".format(kernel, num_filter))
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_2DConv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + 2DConv")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)

    validate_1DCNN(train, sweep=True, class_dict=fetcher.get_class_dict())

   
