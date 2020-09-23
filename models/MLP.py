import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
from collections import namedtuple

from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from ModelTester import ModelTester, resetRand, wandb_init

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "MLP Sweep 1Layer - Hyperparameters",
        "method": "bayes",
        "description": "Find the optimal hyperparameters for 1Layer",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            #"layers": {
            #    "values": [1, 2]
            #},
            #"neurons0": {
            #    "distribution": "int_uniform",
            #    "min": 20,
            #    "max": 300
            #},
            #"neurons2": {
            #    "distribution": "int_uniform",
            #    "min": 20,
            #    "max": 300
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
            "learning_rate":{
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            "beta_1":{
                "distribution": "uniform",
                "min": 0.5,
                "max": 0.99
            },
            "beta_2":{
                "distribution": "uniform",
                "min": 0.6,
                "max": 0.999
            },
            "amsgrad":{
                "distribution": "categorical",
                "values": [True, False]
            },
            "epochs":{
                "distribution": "int_uniform",
                "min": 20,
                "max": 300
            },
            "batch_size":{
                "distribution": "int_uniform",
                "min": 8,
                "max": 512
            },
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the MLP."""

    config = {
        "layers": 2,
        "neurons0": 50,
        "neurons1": 80,
        "neurons2": 50,
        "batch_normalization": False,
        "dropout": None,
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "learning_rate": 0.0018516894394818243, #0.001
        "beta_1": 0.934566655519663, #0.9
        "beta_2": 0.7533619902033079, #0.999
        "epsilon": 1e-07,
        "amsgrad": False,
        "batch_size": 92,
        "epochs": 220
    }

    return config

def create_MLP(input_shape, config):
    """Creates a MLP according to the specifications in 'config'

    Parameters:
    input_shape : tupel of int
        Specifies the size of the Input (aka the first) layer.

    config: Either wandb.config or namedtuple
        Contains the specifications for the model (i.e. number of layers, number of neurons, rate of dropout, etc.).

    ----------
    model : tensorflow.keras.model
        The model created according to the specifications.
    """
    
    model = Sequential()
    model.add(Input(shape=input_shape))

    # adds BatchNormalization & Dropout if specified
    def finish_layer(model, config):
        if config.batch_normalization:
            model.add(BatchNormalization())
        if config.dropout is not None:
            model.add(Dropout(rate=config.dropout))
    
    # add layers
    for layer in range(config.layers):
        model.add(Dense(getattr(config, "neurons{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer))
        finish_layer(model, config)

    model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))
    
    return model


def validate_MLP(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the MLP.
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
            model = create_MLP(input_shape=(train["affected"].shape[1]*2,), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN, count=500)
    
    else:
        filepath = "./output/MLP"
        #filepath = "models/output/MLP/WandB/Test"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_MLP(input_shape=(train["affected"].shape[1]*2,), config=config)
        tester.save_model_plot(model, "MLP_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape="1D", logfile="MLP_1L_N90.dat", model_name="MLP - 1 Hidden Layer", plot_name="MLP_1L_N90.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    #scaler = GRFScaler(scalertype="standard")
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=True, val_setp=0.2, include_info=False)

    validate_MLP(train, sweep=True, class_dict=fetcher.get_class_dict())

   
