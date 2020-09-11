import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
from collections import namedtuple

from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from ModelTester import ModelTester, create_heatmap, resetRand, wandb_init

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, SeparableConv1D, concatenate


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "1DCNN Sweep 1Layer",
        "method": "grid",
        "description": "Find the optimal number of filters, kernel-sizes, etc.",
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
        "layers": 1,
        "filters0": 32,
        "filters1": 32,
        "filters2": 32,
        "kernel0": 8,
        "kernel1": 5,
        "kernel2": 3,
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
        "dropout_cnn": None,
        "dropout_mlp": None,
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

    # adds BatchNormalization, MaxPooling & Dropout if specified
    def finish_layer(conv, config):
        if config.batch_normalization:
            conv = BatchNormalization()(conv)
        if config.max_pooling is not None:
            conv = MaxPooling1D(pool_size=config.max_pooling, strides=config.pooling_stride, padding=config.padding)(conv)
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
    conv = Dense(config.mlp_neurons, activation=config.activation, kernel_regularizer=config.regularizer)(conv)
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

        def train_MLP():
            config = wandb_init(create_config())
            resetRand()
            model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=train_MLP)
    
    else:
        filepath = "./output/1DCNN"
        #filepath = "models/output/MLP/WandB/CNN"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, optimizer=config.optimizer, class_dict=class_dict)
        resetRand()
        model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "1DCNN_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, logfile="1DCNN_1L.dat", model_name="1DCNN - 1 Layer", plot_name="1DCNN_1L.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)

    validate_1DCNN(train, sweep=True, class_dict=fetcher.get_class_dict())

   
