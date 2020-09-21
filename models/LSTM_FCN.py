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
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Dense, Activation, LSTM, concatenate, Reshape, multiply
from tensorflow.keras.backend import permute_dimensions


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "LSTM-FCN Sweep",
        "method": "bayes",
        "description": "Find the optimal hyperparameters.",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "units": {
                "distribution": "uniform",
                "min": 8,
                "max": 128
            },
            "squeeze_and_excite":{
                "distribution": "categorical",
                "values": [True, False]
            },
            "dim_shuffle":{
                "distribution": "categorical",
                "values": [True, False]
            },
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
    """Creates the configuration file with the settings for the LSTM-FCNs network described in
    "LSTM Fully Convolutional Networks for Time Series Classification" (Karim et al., 2018) and
    "Multivariate LSTM-FCNs for time series classification" (Karim et al. 2019).
    """

    config = {
        "layers": 3,
        "filters0": 128,
        "filters1": 256,
        "filters2": 128,
        "kernel0": 8,
        "kernel1": 5,
        "kernel2": 3,
        "padding": "valid",
        "activation": "relu",
        "squeeze_and_excite": True,
        "ratio": 16,
        "dim_shuffle": True,
        "units": 128,
        "dropout_lstm": 0.8,
        "activation_lstm": "tanh",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": False,
        "batch_size": 128,
        "epochs": 100 #2000, 250
    }

    return config

def create_LSTMFCN(input_shape, config):
    """Create a LSTM_FCN network according to the specifications in
    "LSTM Fully Convolutional Networks for Time Series Classification" (Karim et al., 2018) or
    "Multivariate LSTM-FCNs for time series classification" (Karim et al. 2019)
     depending on the settings obtained from 'config'

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

    def add_squeeze_and_excite_block(conv, layer):
        filters = getattr(config, "filters{}".format(layer))
        se_shape = (1, filters)
        se = GlobalAveragePooling1D()(conv)
        se = Reshape(se_shape)(se)
        se = Dense(filters//config.ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        return multiply([conv, se])

    # add cnn layers
    conv = input_layer
    for layer in range(config.layers):
        conv = Conv1D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), kernel_regularizer=config.regularizer, padding=config.padding)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(config.activation)(conv)

        if config.squeeze_and_excite and layer < (config.layers -1): 
            conv = add_squeeze_and_excite_block(conv, layer)

    conv = GlobalAveragePooling1D()(conv)

    # add lstm layer
    if config.dim_shuffle:
        lstm = permute_dimensions(input_layer, (0, 2, 1))
    else:
        lstm = input_layer
    lstm = LSTM(units=config.units, activation=config.activation_lstm, kernel_regularizer=config.regularizer, dropout=config.dropout_lstm, return_sequences=False)(lstm)

    # concatenate
    out = concatenate([conv, lstm])
    out = Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer)(out)

    model = Model(input_layer, out)

    return model


def validate_LSTMFCN(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the LSTM-FCN networks as defined in 
    "LSTM Fully Convolutional Networks for Time Series Classification" (Karim et al., 2018) and
    "Multivariate LSTM-FCNs for time series classification" (Karim et al. 2019).
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
            model = create_LSTMFCN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/LSTMFCN"
        #filepath = "models/output/MLP/WandB/LSTMFCN"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_LSTMFCN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "LSTMFCN_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape="1D", logfile="LSTMFCN.dat", model_name="LSTM-FCN", plot_name="LSTMFCN.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)

    validate_LSTMFCN(train, test=None, class_dict=fetcher.get_class_dict(), sweep=False)