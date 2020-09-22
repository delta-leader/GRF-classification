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
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Activation, Add, BatchNormalization, MaxPooling1D, concatenate


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "InceptionTime Sweep",
        "method": "bayes",
        "description": "Find the optimal hyperparameters.",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
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
                "max": 200
            },
            "batch_size":{
                "distribution": "int_uniform",
                "min": 8,
                "max": 512
            },
            "bottleneck_size":{
                "distribution": "int_uniform",
                "min": 1,
                "max": 10
            },
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the InceptionTime network as described in
    "InceptionTime: Finding AlexNet for Time Series Classification" (Fawaz et al., 2019).
    """

    config = {
        "blocks": 2,
        "layers": 3,
        "bottleneck_size": 5, 
        "nb_filters": 32,
        "kernel_sizes": [10, 20, 40],
        "stride": 1,
        "pool_size": 3,
        "padding": "same",
        "activation": "linear",
        "activation_out": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "use_bias": False,
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

def create_InceptionTime(input_shape, config):
    """Creates the InceptionTime network according to the specifications in
    "InceptionTime: Finding AlexNet for Time Series Classification" (Fawaz et al., 2019)
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

    def inception_block(tensor, config):
        out = tensor
        for layer in range(config.layers):
            input_inception = Conv1D(filters=config.bottleneck_size, kernel_size=1, padding=config.padding, activation=config.activation, use_bias=config.use_bias)(out)

            conv_list = []
            for kernel in range(len(config.kernel_sizes)):
                conv_list.append(Conv1D(filters=config.nb_filters, kernel_size=config.kernel_sizes[kernel], strides=config.stride, padding=config.padding, activation=config.activation, use_bias=config.use_bias)(input_inception))

            pool = MaxPooling1D(pool_size=config.pool_size, strides=config.stride, padding=config.padding)(out)
            pool = Conv1D(filters=config.nb_filters, kernel_size=1, padding=config.padding, activation=config.activation, use_bias=config.use_bias)(pool)
            conv_list.append(pool)
        
            out = concatenate(conv_list, axis=2)
            out = BatchNormalization()(out)
            out = Activation(activation=config.activation_out)(out)

        return out

    def add_shortcuts(residual, tensor):
        out = Conv1D(filters=int(tensor.shape[-1]), kernel_size=1, padding=config.padding, use_bias=config.use_bias)(residual)
        out = BatchNormalization()(out)
        out = Add()([out, tensor])
        out = Activation(activation=config.activation_out)(out)
        return out


    out = input_layer
    residual = input_layer

    for block in range(config.blocks):
        out = inception_block(out, config)
        out = add_shortcuts(residual, out)
        residual = out

    out = GlobalAveragePooling1D()(out)
    out = Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer)(out)
    model = Model(input_layer, out)

    return model


def validate_InceptionTime(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the InceptionTime network as defined in 
    "InceptionTime: Finding AlexNet for Time Series Classification" (Fawaz et al., 2019).
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
            model = create_InceptionTime(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/InceptionTime"
        #filepath = "models/output/MLP/WandB/InceptionTime"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_InceptionTime(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "InceptionTime_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, shape="1D", logfile="InceptionTime.dat", model_name="InceptionTime", plot_name="InceptionTime.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2,include_info=False)

    validate_InceptionTime(train, test=None, class_dict=fetcher.get_class_dict(), sweep=True)
