import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
from collections import namedtuple

from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from ModelTester import ModelTester, resetRand, wandb_init

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Conv2D, AveragePooling2D, LSTM


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "Alharthi2D - Hyperparameters",
        "method": "bayes",
        "description": "Find the optimal hyperparameters",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "batch_normalization": {
                "distribution": "categorical",
                "values": [True, False]
            },
            "avg_pooling": {
                "distribution": "categorical",
                "values": [True, False]
            },
            "regularizer": {
                "distribution": "categorical",
                "values": [None, "l2"]
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.4,
                "max": 0.6
            },
            #"dropout_mlp": {
            #    "distribution": "uniform",
            #    "min": 0.1,
            #    "max": 0.3
            #},
            "learning_rate":{
                "distribution": "uniform",
                "min": 0.0001,
                "max": 0.01
            },
            "beta_1":{
                "distribution": "uniform",
                "min": 0.6,
                "max": 0.99
            },
            "beta_2":{
                "distribution": "uniform",
                "min": 0.7,
                "max": 0.999
            },
            "epsilon":{
                "distribution": "uniform",
                "min": 1e-08,
                "max": 1e-06
            },
            "amsgrad":{
                "distribution": "categorical",
                "values": [True, False]
            },
            "epochs":{
                "distribution": "int_uniform",
                "min": 20,
                "max": 200
            },
            "batch_size":{
                "distribution": "int_uniform",
                "min": 16,
                "max": 512
            },
        }
    }

    return sweep_config


def create_config_1D():
    """Creates the configuration file with the settings for the 1-dimensional variant of the network described in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019).
    """

    config = {
        "layers": 4,
        "filters0": 12,
        "filters1": 24,
        "filters2": 48,
        "filters3": 96,
        "kernel0": 2,
        "kernel1": 2,
        "kernel2": 2,
        "kernel3": 2,
        "neurons": 50,
        "pool_size": 2,
        "padding": "same",
        "batch_normalization": False,
        "dropout_cnn": 0.5,
        "dropout_mlp": 0.2,
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": "l2",
        "optimizer": "adam",
        "learning_rate": 0.002,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": False,
        "batch_size": 180,
        "epochs": 100
    }

    return config


def create_1D(input_shape, config):
    """Creates a 1-dimensional CNN according to the specifications in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019)
    and the settings obtained from 'config'

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

    # adds BatchNormalization if specified
    def finish_layer(model, config):
        if config.batch_normalization:
            model.add(BatchNormalization())

    # add 1DCNN-layers
    for layer in range(config.layers):
        model.add(Conv1D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding))
        finish_layer(model, config)
        model.add(MaxPooling1D(pool_size=config.pool_size))

    model.add(Flatten())
    model.add(Dropout(rate=config.dropout_cnn))
    model.add(Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer))
    model.add(Dropout(rate=config.dropout_mlp))

   
    model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))
    
    return model


def create_config_2D():
    """Creates the configuration file with the settings for the 2-dimensional variant of the network described in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019).
    """

    config = {
        "layers": 2,
        "filters0": 12,
        "filters1": 24,
        "filters2": 48,
        "filters3": 96,
        "kernel0": (2,1),
        "kernel1": (1,2),
        "kernel2": 2,
        "kernel3": 2,
        "avg_pooling" : True,
        "pool_size0": (2,2),
        "pool_size1": (2,2),
        "pool_size2": (2,2),
        "neurons": 100,
        "padding": "same",
        "batch_normalization": False,
        "dropout": 0.5,
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": "l2",
        "optimizer": "adam",
        "learning_rate": 0.002,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": False,
        "batch_size": 180,
        "epochs": 100
    }

    return config


def create_2D(input_shape, config):
    """Creates a 2-dimensional CNN according to the specifications in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019)
    and the settings obtained from 'config'

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

    # adds BatchNormalization if specified
    def finish_layer(model, config, layer):
        if config.batch_normalization:
            model.add(BatchNormalization())
        if config.avg_pooling:
            model.add(AveragePooling2D(pool_size=getattr(config, "pool_size{}".format(layer))))


    # add 2DCNN-layers
    for layer in range(config.layers):
        model.add(Conv2D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding))
        finish_layer(model, config, layer)

    model.add(Flatten())
    model.add(Dropout(rate=config.dropout))
    model.add(Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer))
    model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))
   
    return model


def create_config_LSTM():
    """Creates the configuration file with the settings for the LSTM network described in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019).
    """

    config = {
        "layers": 2,
        "units0": 100,
        "units1": 40,
        "neurons": 20,
        "dropout_lstm": 0.2,
        "dropout_mlp": 0.5,
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "learning_rate": 0.002,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": False,
        "batch_size": 200,
        "epochs": 100
    }

    return config


def create_LSTM(input_shape, config):
    """Creates a LSTM network according to the specifications in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019)
    and the settings obtained from 'config'

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

    # add lstm-layers
    for layer in range(config.layers):
        if layer < (config.layers - 1):
            model.add(LSTM(units=getattr(config, "units{}".format(layer)), kernel_regularizer=config.regularizer, return_sequences=True))
        else:
            model.add(LSTM(units=getattr(config, "units{}".format(layer)), kernel_regularizer=config.regularizer, dropout=config.dropout_lstm, recurrent_dropout=config.dropout_lstm, return_sequences=False))

    model.add(BatchNormalization())
    model.add(Dropout(rate=config.dropout_mlp))
    model.add(Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer))  
    model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))
    
    return model


def validate_model(train, model="1D", test=None, class_dict=None, sweep=False):
    """Trains and tests the the networks specified in
    "Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing" (Alharthi et al. 2019).
    
    Available models are:
    '1D': 1-dimensional CNN as specified in the paper.
    '2D': 2-dimensional CNN as specified in the paper.
    'LSTM': LSTM-network as specified in the paper.

    Training can be conducted in two modes:
    'sweep' == True -> performs a sweep of hyperparameters according to the specified sweep-configuration.
    'sweep' == False -> Performs a single training and evaluation (on test and validation set) according to the configured settings. Includes creationg of plots and confusion matrices.


    Parameters:
    train : dict
        Containing the GRF-data for training and validation.

    model : string, default="1D"
        Specifies the model to be used. Must be one of '1D', '2D' or 'LSTM'.
    
    test : dict, default=None
        Containing the GRF-data for the test-set.
    
    class_dict: dict, default=None
        Dictionary that maps the numbered labels to names. Used to create the confusion matrix.

    sweep : bool, default=False
        If true performs a hyperparameter sweep using W&B according to the specified sweep-configuration (using only the validation-set)
        Otherwise a local training and evalution run is performed, providing the results for both validation- and test-set.

    ----------
    Raises:
    ValueError : If 'model' is not one of '1D', '2D' or 'LSTM'.
    """

    create_model = None
    model_config = None
    shape = "1D"
    input_shape = (train["affected"].shape[1], train["affected"].shape[2]*2)
    if model == "1D":
        create_model = create_1D
        model_config = create_config_1D()
    if model == "2D":
        create_model = create_2D
        model_config = create_config_2D()
        shape = "2D_SST"
        input_shape = (2, train["affected"].shape[2], train["affected"].shape[1])
    if model == "LSTM":
        create_model = create_LSTM
        model_config = create_config_LSTM()
    
    if create_model is None:
        raise ValueError("'{}' does not specify a valid model. Supported values are '1D', '2D' or 'LSTM'.".format(model))
      
    if sweep:
        sweep_config = create_sweep_config()
        tester = ModelTester(class_dict=class_dict) 

        def trainNN():
            config = wandb_init(model_config)
            resetRand()
            model = create_model(input_shape=input_shape, config=config)
            tester.perform_sweep(model, config, train, shape=shape, useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/Alharthi/" + model
        #filepath = "models/output/MLP/WandB/Alharthi"
        config = model_config
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_model(input_shape=input_shape, config=config)
        tester.save_model_plot(model, "Alharthi_2D.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape=shape, logfile="Alharthi_2D.dat", model_name="Alharthi - 2D", plot_name="Alharthi_2D.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)

    validate_model(train, model="2D", test=None, class_dict=fetcher.get_class_dict(), sweep=True)