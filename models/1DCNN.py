import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
from collections import namedtuple

from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from ModelTester import ModelTester, create_heatmap, resetRand, wandb_init

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, Flatten, Dropout, BatchNormalization, SeparableConv1D, concatenate


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "1DCNN Sweep 1Layer (dilated)",
        "method": "bayes",
        "description": "Find the optimal number of filters, kernel-sizes, etc.",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "filters0": {
                "distribution": "int_uniform",
                "min": 20,
                "max": 250
            },
            #"filters1": {
            #    "values": [10, 20, 30, 40, 50]
            #    #"min": 40,
            #    #"max": 190
            #},
            "batch_normalization": {
                "distribution": "categorical",
                "values": [True, False]
            },
            "separable": {
                "distribution": "categorical",
                "values": [True, False]
            },
            "skipConnections": {
                "distribution": "categorical",
                "values": [True, False]
            },
            "dropout_cnn": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5
            },
            "dropout_mlp": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5
            },
            "kernel0": {
                "distribution": "int_uniform",
                "min": 2,
                "max": 20
            },
            "pool_type": {
                "distribution": "categorical",
                "values": ["max", "avg", None]
            },
            "pool_size":{
                "distribution": "int_uniform",
                "min": 2,
                "max": 4
            },
            "stride0":{
                "distribution": "int_uniform",
                "min": 1,
                "max": 5
            },
            "neurons":{
                "distribution": "int_uniform",
                "min": 20,
                "max": 200
            },
            #"kernel1": {
            #    "values": [(3), (5), (7), (9), (11), (13), (15)]
            #    #"min": 40,
            #    #"max": 190
            #},
            #"learning_rate":{
            #    "distribution": "uniform",
            #    "min": 0.0001,
            #    "max": 0.01
            #},
            #"beta_1":{
            #    "distribution": "uniform",
            #    "min": 0.5,
            #    "max": 0.99
            #},
            #"beta_2":{
            #    "distribution": "uniform",
            #    "min": 0.6,
            #    "max": 0.999
            #},
            #"amsgrad":{
            #    "distribution": "categorical",
            #    "values": [True, False]
            #},
            #"epochs":{
            #    "distribution": "int_uniform",
            #    "min": 20,
            #    "max": 200
            #},
            #"batch_size":{
            #    "distribution": "int_uniform",
            #    "min": 8,
            #    "max": 512
            #},
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the 1DCNN."""

    config = {
        "layers": 2,
        "filters0": 131,
        "filters1": 31,
        "filters2": 32,
        "kernel0": 18,
        "kernel1": 6,
        "kernel2": 3,
        "stride0": 1,
        "stride1": 1,
        "stride2": 1,
        "dilation0": 12,
        "dilation1": 14,
        "dilation2": 1,
        "batch_normalization": False,
        "pool_type": "max",
        "pool_size": 3,
        "pool_stride": None,
        "neurons": 46,
        "dropout_cnn": 0.14204535896572307,
        "dropout_mlp": 0.43700599895787645,
        "separable": False,
        "skipConnections": False,
        "padding": "same",
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-07,
        "amsgrad": False,
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
    
    input_layer = Input(shape=input_shape)

    # adds BatchNormalization, Pooling & Dropout if specified
    def finish_layer(conv, config):
        if config.batch_normalization:
            conv = BatchNormalization()(conv)
        if config.pool_type == "max":
            conv = MaxPooling1D(pool_size=config.pool_size, strides=config.pool_stride)(conv)
        if config.pool_type == "avg":
            conv = AveragePooling1D(pool_size=config.pool_size, strides=config.pool_stride)(conv)
        if config.dropout_cnn is not None:
            conv = Dropout(rate=config.dropout_cnn)(conv)
        return conv

    def add_conv_layer(conv, config, layer):
        if config.separable:
            conv = SeparableConv1D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), strides=getattr(config, "stride{}".format(layer)), dilation_rate=getattr(config, "dilation{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding)(conv)
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
    
    # add skipConnections if specified
    if config.skipConnections:
        conv = concatenate([Flatten()(x) for x in ([input_layer]+skip_modules)])
    else:
        conv = Flatten()(conv)

    # add dense layers
    if config.neurons is not None:
        conv = Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer)(conv)
        if config.dropout_mlp is not None:
            conv = Dropout(rate=config.dropout_mlp)(conv)

    out = Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer)(conv)

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

    sweep : bool, default=False
        If true performs a hyperparameter sweep using W&B according to the specified sweep-configuration (using only the validation-set)
        Otherwise a local training and evalution run is performed, providing the results for both validation- and test-set.
    """
      
    if sweep:
        sweep_config = create_sweep_config()
        tester = ModelTester(class_dict=class_dict) 

        def trainNN():
            config = wandb_init(create_config())
            resetRand()
            model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN, count=1000)
    
    else:
        filepath = "./output/1DCNN"
        #filepath = "models/output/MLP/WandB/CNN"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "1DCNN_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape="1D", logfile="1DCNN_1L.dat", model_name="1DCNN - 1 Layer", plot_name="1DCNN_1L.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    #scaler = GRFScaler(scalertype="standard")
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)
    #val = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True)
    #train = set_valSet(train, val, parse=None)
    
    validate_1DCNN(train, sweep=False, class_dict=fetcher.get_class_dict())

   
