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
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Activation, Add, BatchNormalization


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "ResNet Sweep",
        "method": "grid",
        "description": "Find the optimal hyperparameters.",
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
    """Creates the configuration file with the settings for ResNet as described in
    "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline" (Wang and Oates, 2016).
    """

    config = {
        "blocks": 3,
        "layers": 3,
        "filters0": 64,
        "filters1": 128,
        "filters2": 128,
        "kernel0": 8,
        "kernel1": 5,
        "kernel2": 3,
        "padding": "same",
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": False,
        "batch_size": 32,
        "epochs": 100
    }

    return config

def create_ResNet(input_shape, config):
    """Creates ResNet according to the specifications in
    "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline" (Wang and Oates, 2016)
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
    
    input_layer = Input(shape=input_shape)

    def residual_block(tensor, config, block):
        out = tensor
        expanded_in = Conv1D(filters=getattr(config, "filters{}".format(block)), kernel_size=1, kernel_regularizer=config.regularizer, padding=config.padding)(tensor)
        expanded_in = BatchNormalization()(expanded_in)
        for layer in range(config.layers):
            out = Conv1D(filters=getattr(config, "filters{}".format(block)), kernel_size=getattr(config, "kernel{}".format(layer)), kernel_regularizer=config.regularizer, padding=config.padding)(out)
            out = BatchNormalization()(out)
            if layer == (config.layers - 1):
                out = Add()([expanded_in, out])
            out = Activation(config.activation)(out)
        return out

    out = input_layer
    for block in range(config.blocks):
        out = residual_block(out, config, block)

    out = GlobalAveragePooling1D()(out)
    out = Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer)(out)

    model = Model(input_layer, out)

    return model


def validate_ResNet(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the ResNet as defined in 
    "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline" (Wang and Oates, 2016).
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
            model = create_ResNet(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/ResNet"
        #filepath = "models/output/MLP/WandB/ResNet"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_ResNet(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "ResNet_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape="1D", logfile="ResNet.dat", model_name="ResNet", plot_name="ResNet.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2,include_info=False)

    validate_ResNet(train, test=None, class_dict=fetcher.get_class_dict(), sweep=False)
