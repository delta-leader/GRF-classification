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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, AveragePooling1D, BatchNormalization, LSTM, SeparableConv1D, Flatten, concatenate


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "LSTM Sweep 1Layer(DropOut)",
        "method": "grid",
        "description": "Find the optimal number of units/layers, etc.",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "units0": {
                "distribution": "int_uniform",
                "min": 20,
                "max": 100
            },
            #"units1": {
            #    "distribution": "int_uniform",
            #    "min": 20,
            #    "max": 100
            #},
            "neurons": {
                "distribution": "int_uniform",
                "min": 20,
                "max": 200
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
            "dropout_lstm": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5
            },
            #"dropout_recurrent": {
            #    "distribution": "uniform",
            #    "min": 0.0,
            #    "max": 0.5
            #},
            "dropout_mlp": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.5
            },
            #"kernel0": {
            #    "values": [(3), (5), (7), (9), (11), (13), (15)]
            #    #"min": 40,
            #    #"max": 190
            #},
            #"kernel1": {
            #    "values": [(3), (5), (7), (9), (11), (13), (15)]
            #    #"min": 40,
            #    #"max": 190
            #},
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the LSTM."""

    config = {
        "layers_lstm": 1,
        "units0": 90,
        "units1": 60,
        "units2": 30,
        "dropout_lstm": 0.0,
        "dropout_recurrent": 0.0,
        "activation_lstm": "tanh",
        "mode_cnn": None,
        "layers_cnn": 0,
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
        "pool_type": None,
        "pool_size": 2,
        "pool_stride": None,
        "neurons": None,
        "dropout_cnn": None,
        "dropout_mlp": None,
        "separable": False,
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

def create_LSTM(input_shape, config):
    """Creates a LSTM (or LSTM + CNN) network according to the specifications in 'config'

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
    
    def add_lstm_layer(lstm, config, layer, return_seq):
        lstm = LSTM(units=getattr(config, "units{}".format(layer)), activation=config.activation_lstm, kernel_regularizer=config.regularizer, dropout=config.dropout_lstm, recurrent_dropout=config.dropout_recurrent, return_sequences=return_seq)(lstm)
        return lstm

    lstm = input_layer

    # add cnn if specified
    if config.mode_cnn is not None:
        conv = input_layer       
        for layer in range(config.layers_cnn):
            conv = add_conv_layer(conv, config, layer)
            conv = finish_layer(conv, config)

        if config.mode_cnn == "serial":
            lstm = conv

    # add lstm layers
    for layer in range(config.layers_lstm):
        return_seq = True
        if layer == (config.layers_lstm - 1):
            return_seq = False
        lstm = add_lstm_layer(lstm, config, layer, return_seq)


    # apply lstm and cnn in parallel if specified
    if config.mode_cnn == "parallel":
        conv = Flatten()(conv)
        out = concatenate([conv, lstm])
    else:
        out = lstm

    # add dense layers
    if config.neurons is not None:
        out = Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer)(out)
        if config.dropout_mlp is not None:
            out = Dropout(rate=config.dropout_mlp)(out)

    out = Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer)(out)

    model = Model(input_layer, out)
    
    return model


def validate_LSTM(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the LSTM network.
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
            model = create_LSTM(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/LSTM"
        #filepath = "models/output/MLP/WandB/LSTM"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_LSTM(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "LSTM_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape="1D", logfile="LSTM_1L.dat", model_name="LSTM - 1 Layer", plot_name="LSTM_1L.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False)

    validate_LSTM(train, sweep=True, class_dict=fetcher.get_class_dict())