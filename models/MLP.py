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
        "name": "MLP Sweep 1Layer - Dropout&BN",
        "method": "bayes",
        "description": "Find the optimal number of layers/neurons",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            #"layers": {
            #    "value": 1
            #},
            #"neurons0": {
            #    "distribution": "int_uniform",
            #    "min": 40,
            #    "max": 190
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
                "min": 0.6,
                "max": 0.99
            },
            "beta_2":{
                "distribution": "uniform",
                "min": 0.7,
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
                "min": 16,
                "max": 512
            },
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the MLP."""

    config = {
        "layers": 1,
        "neurons0": 190,
        "neurons1": 50,
        "neurons2": 50,
        "batch_normalization": False,
        "dropout": None,
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

        def train_MLP():
            config = wandb_init(create_config())
            resetRand()
            model = create_MLP(input_shape=(train["affected"].shape[1]*2,), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=train_MLP)
    
    else:
        filepath = "models/output/MLP/WandB/Test"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, optimizer=config.optimizer, class_dict=class_dict)
        resetRand()
        model = create_MLP(input_shape=(train["affected"].shape[1]*2,), config=config)
        tester.save_model_plot(model, "MLP_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, logfile="MLP_1L_N190.dat", model_name="MLP - 1 Hidden Layer", plot_name="MLP_1L_N190.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))

#TODO remove
#def get_ConvModel(kernel_size, num_filters, neurons, input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with a convolutional layer before the hidden dense layer."""
"""    
    model = Sequential()   
    model.add(Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    #model.add(SeparableConv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer))
    #model.add(Dropout(rate=0.2))
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_ConvModel(train, test, class_dict):
"""    """Test different settings for the 1ConvLayer-MLP"""
"""
    #num_neurons = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    num_neurons = 50
    num_filters = [8, 16, 32, 64, 128]#, 256, 512]
    kernel_sizes = [2, 3, 5, 7, 9, 11]#, 13, 15, 21, 33]
    filepath = "models/output/MLP/Conv/BOOST"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_ConvModel(kernel, num_filter, num_neurons, (train["affected"].shape[1], train["affected"].shape[2]*2)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="MLP_Conv_K{}_F{}.dat".format(kernel, num_filter), model_name="MLP + Conv", plot_name="MLP_Conv_K{}_F{}.png".format(kernel, num_filter), boost=True)
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_Conv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + Conv")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))
        

def get_DilConvModel(kernel_size, num_filters, neurons, dilation_rate, input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
"""    """Creates a sequential model with a dilated convolutional layer before the hidden dense layer."""
"""    
    model = Sequential()   
    model.add(Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, dilation_rate=dilation_rate, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_DilConvModel(train, test, class_dict):
"""    """Test different settings for the dilated Conv-MLP"""
"""
    #num_neurons = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    num_neurons = 50
    #num_filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    num_filters = 256
    kernel_sizes = [2, 3, 4, 5, 6]
    dilation_rates = [2, 3, 5, 7, 10, 13, 15]
    filepath = "models/output/MLP/Conv/Dilated"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for dilation in dilation_rates:
            resetRand()
            model = get_DilConvModel(kernel, num_filters, num_neurons, dilation, (train["affected"].shape[1], train["affected"].shape[2]*2)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="MLP_DilConv_F256_K{}_DR{}.dat".format(kernel, dilation), model_name="MLP + Dilated Convolution (#Filters = 256)", plot_name="MLP_DilConv_F256_K{}_DR{}.png".format(kernel, dilation))
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=dilation_rates, filename=filepath+"/Comparison_MLP_DilConv_F256_KvDR.png", yaxis_title="Kernel-Size", xaxis_title="Dilation-Rate", title="MPL + Dilated Convolution (#Filters=256)")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


def get_ConvModel2Groups(kernel_size, num_filters, neurons, input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
"""    """Creates a sequential model with a convolutional layer (separated into 2 groups) before the hidden dense layer.
    This basically separates the filters for the affected and unaffected side (i.e. a different set of filters is used for affected/non-affected."""
"""    
    # branch 1
    affected_input = Input(shape=input_shape)
    affected = Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer)(affected_input)
    #affected = SeparableConv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer)(affected_input)
    affected = Flatten()(affected)
    #affected = Dropout(rate=0.5)(affected)
        
    # branch 2
    non_affected_input = Input(shape=input_shape)
    non_affected = Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer)(non_affected_input)
    #non_affected = SeparableConv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer)(non_affected_input)
    non_affected = Flatten()(non_affected)
    #non_affected = Dropout(rate=0.5)(non_affected)
        
    # join branches
    concatenated = concatenate([affected, non_affected])
    #concatenated = Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, activation=activation, kernel_regularizer=kernel_regularizer)(concatenated)
    #concatenated = Flatten()(concatenated)
    concatenated = Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer)(concatenated)
    concatenated = Dropout(rate=0.2)(concatenated)
    out = Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer) (concatenated)
    model = Model([affected_input, non_affected_input], out)
    
    return model

def test_ConvModel2Groups(train, test, class_dict, train_norm=None, test_norm=None):
"""    """Test different settings for the 1ConvLayer-MLP"""
"""
    #num_neurons = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    num_neurons = 50
    num_filters = [8, 16, 32, 64, 128, 256]# 512, 1024]
    kernel_sizes = [2, 3, 5, 7, 9, 11, 13, 15, 21, 33]
    filepath = "models/output/MLP/Conv/Groups/2/DropOut"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_ConvModel2Groups(kernel, num_filter, num_neurons, (train["affected"].shape[1], train["affected"].shape[2])) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_model(model, groups=2, data_dict=train, test_dict=test, train_norm_dict=train_norm, test_norm_dict=test_norm, logfile="MLP_Conv2Groups_DropOut02_K{}_F{}.dat".format(kernel, num_filter), model_name="MLP + Conv (2 groups with DropOut=0.2)", plot_name="MLP_Conv2Groups_DropOut02_K{}_F{}.png".format(kernel, num_filter))
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_Conv2Groups_DropOut02_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + Conv(2 Groups with DropOut=0.2)")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


def get_ConvSkipModel(kernel_size, num_filters, neurons, input_shape, activation="relu", final_activation="softmax", kernel_regularizer=None):
"""    """Creates a sequential model with a convolutional layer using skip connections to feed the input directly into the hidden dense layer.
    Basically the hidden dense layer uses both the output from the convolutional layer and the unmodified data as input."""
"""    
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
"""    """Test different settings for the ConvSkipModel-MLP"""
"""
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
"""    """Creates a sequential model with a 2D convolutional layer before the hidden dense layer."""
"""    
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
"""    """Test different settings for the 2D-ConvLayer-MLP"""
"""
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

"""
if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=True, val_setp=0.2, include_info=False)

    validate_MLP(train, sweep=False, class_dict=fetcher.get_class_dict())

   
