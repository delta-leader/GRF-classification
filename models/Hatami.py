import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
from collections import namedtuple

from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from ModelTester import ModelTester, resetRand, wandb_init
from GRFImageConverter import GRFImageConverter, normalize_images
from ImageFilter import ImageFilter

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Conv2D, Input

def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "IMG Sweep - Hyperparameters",
        "method": "bayes",
        "description": "Find the optimal hyperparameters.",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            #"filters0": {
            #    "distribution": "int_uniform",
            #    "min": 16,
            #    "max": 64
            #},
            #"filters1": {
            #    "distribution": "int_uniform",
            #    "min": 16,
            #    "max": 64
            #},
            #"kernel0": {
            #    "distribution": "int_uniform",
            #    "min": 3,
            #    "max": 10
            #},
            #"kernel1": {
            #    "distribution": "int_uniform",
            #    "min": 3,
            #    "max": 10
            #},
            #"dropout_cnn": {
            #    "distribution": "uniform",
            #    "min": 0.1,
            #    "max": 0.5
            #},
            #"dropout_mlp": {
            #    "distribution": "uniform",
            #    "min": 0.3,
            #    "max": 0.6
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
    """Creates the configuration file with the settings for the CNN network described in
    "Classification of Time-Series Images Using Deep Convolutional Neural Networks" (Hatami et al., 2017).
    """

    config = {
        "layers": 2,
        "filters0": 32, #36, #32
        "filters1": 32,
        "kernel0": 3, #9, #3
        "kernel1": 3, #10, #3
        "padding": "valid",
        "pool_size": 2,
        "dropout_cnn": 0.25, #0.2322811230216193, #0.25
        "neurons": 128,
        "dropout_mlp": 0.5, #0.45561536933924407, #0.5
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": True,
        "batch_size": 32,
        "epochs": 150,
    }

    return config

def create_IMG(input_shape, config):
    """Creates FCN network according to the specifications in
    "Classification of Time-Series Images Using Deep Convolutional Neural Networks" (Hatami et al., 2017)
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

    # add cnn layers
    for layer in range(config.layers):
        model.add(Conv2D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), strides=1, activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding))
        model.add(MaxPooling2D(pool_size=config.pool_size))
        model.add(Dropout(config.dropout_cnn))

    model.add(Flatten())
    model.add(Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer))
    model.add(Dropout(rate=config.dropout_mlp))
    model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))
    
    return model


def validate_IMG(train, images, test=None, class_dict=None, sweep=False):
    """Trains and tests the network as defined in 
    "Classification of Time-Series Images Using Deep Convolutional Neural Networks" (Hatami et al., 2017).
    Two modes are available:
    'sweep' == True -> performs a sweep of hyperparameters according to the specified sweep-configuration.
    'sweep' == False -> Performs a single training and evaluation (on test and validation set) according to the configured settings. Includes creationg of plots and confusion matrices.

    Parameters:
    train : dict
        Containing the GRF-data for training and validation.

    images : list
        The images to consider for training and testing
    
    test : dict, default=None
        Containing the GRF-data for the test-set.
    
    class_dict: dict, default=None
        Dictionary that maps the numbered labels to names. Used to create the confusion matrix.

    sweep : bool, default=False
        If true performs a hyperparameter sweep using W&B according to the specified sweep-configuration (using only the validation-set)
        Otherwise a local training and evalution run is performed, providing the results for both validation- and test-set.
    """

    count = len(images)
    img = images[0]
      
    if sweep:
        sweep_config = create_sweep_config()
        tester = ModelTester(class_dict=class_dict)

        def trainNN():
            config = wandb_init(create_config())
            resetRand()
            model = create_IMG(input_shape=(train["affected"][img].shape[1], train["affected"][img].shape[2], train["affected"][img].shape[3]*count*2), config=config)
            tester.perform_sweep(model, config, train, shape="IMG_STACK", images=images, useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/FCN"
        #filepath = "models/output/MLP/WandB/FCN"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_IMG(input_shape=(train["affected"][img].shape[1], train["affected"][img].shape[2], train["affected"][img].shape[3]*count*2), config=config)
        tester.save_model_plot(model, "IMG_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, images=images, config=config, test=test, shape="IMG_STACK", logfile="IMG.dat", model_name="IMG", plot_name="IMG.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))
        return acc, val_acc


def get_conv_images(images):
    """Extracts the correct conversion list from a list of image.
    If either 'gadf' or 'gasf' are present in the list, they are replaced by 'gadf'.

    Parameters:
    images : list
        The list of images to be used for the classification.

    ----------
    Returns:
    img : list
        The list of transformations to compute.
    """

    img = []
    if "gadf" in images or "gasf" in images:
        img += ["gaf"]
    if "mtf" in images:
        img += ["mtf"]
    if "rcp" in images:
        img += ["rcp"]

    return img


def get_HatamiModel(input_shape, kernel_sizes=[(3,3), (3,3)], num_filters=[32, 32], neurons=128, activation="relu", final_activation="softmax", kernel_regularizer=None):
    """Creates a sequential model with a 2D convolutional layer before the hidden dense layer."""
    
    model = Sequential()   
    model.add(Conv2D(filters=num_filters[0], kernel_size=kernel_sizes[0], strides=1, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=num_filters[1], kernel_size=kernel_sizes[1], strides=1, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Dropout(rate=0.5))
    model.add(Dense(5, activation=final_activation, kernel_regularizer=kernel_regularizer))
    
    return model

def test_HatamiModel(train, test, class_dict):
    """Test different settings for the Hatami-Model"""

    num_neurons = 128
    num_filters = [[32, 32]]
    kernel_sizes = [[(3,3), (3,3)]]
    filepath = "models/output/Image/Hatami/GADF+MTF"
    optimizer = "adam"

    accAll = []
    val_accAll = []
    for kernel in kernel_sizes:
        accAcu = []
        val_accAcu =[]
        for num_filter in num_filters:
            resetRand()
            model = get_HatamiModel(kernel_sizes=kernel, num_filters=num_filter, neurons=num_neurons, input_shape=(train["affected"]["mtf"].shape[1], train["affected"]["mtf"].shape[2],  train["affected"]["mtf"].shape[3]*4)) 
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
            acc, _, val_acc, _ = tester.test_image_model(model, images=["gasf","mtf"], epochs=100, data_dict=train, test_dict=test, logfile="GADFMTF.dat", model_name="GAdF + MTF - Hatami", plot_name="GADFMTF.png")
            accAcu.append(acc)
            val_accAcu.append(val_acc)
        accAll.append(accAcu)
        val_accAll.append(val_accAcu)

    #create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_2DConv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + 2DConv")

    for kernel, values in zip(kernel_sizes, val_accAll):
        print("Kernel-Size {}, ValAccuracy: ,{}".format(kernel, ','.join(map(str, values))))


def test_ResNet(train, test, class_dict):
    """Test different settings for the Hatami-Model"""

    #num_neurons = 128
    #num_filters = [[32, 128]]
    #kernel_sizes = [[(3,3), (3,3)]]
    filepath = "models/output/Image/Xception/GASF"
    optimizer = "adam"

    resetRand()
    input_tensor = Input(shape=(train["affected"]["gasf"].shape[1], train["affected"]["gasf"].shape[2],  train["affected"]["gasf"].shape[3]*2))
    #model = ResNet50(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling=None, classes=5)
    model = Xception(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling=None, classes=5)
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    #tester.save_model_plot(model, "MLP_model_1L_N{}.png".format(neurons))
    acc, _, val_acc, _ = tester.test_image_model(model, images=["gasf"], epochs=100, data_dict=train, test_dict=test, logfile="GASF.dat", model_name="GASF - ResNet50", plot_name="GASF.png")

    #create_heatmap(val_accAll, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_MLP_2DConv_KvF.png", yaxis_title="Kernel-Size", xaxis_title="#Filters", title="MPL + 2DConv")
    print("Accuracy {}, ValAccuracy: ,{}".format(acc, val_acc))



if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(0,1))
    train0 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False, clip=True)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train1 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False, clip=True)
    conv_args = {
        "images": ["rcp"],
        "filter": None,
        "filter_size": (2,2),
        "num_bins": 25,
        "range": (-1, 1),
        "dims": 2,
        "delay": 3,
        "metric": "euclidean"
    }
    converter = GRFImageConverter()
    #if this is used with sweep, tensorflow will use the CPU
    #converter.enableGpu()
    imgFilter = ImageFilter("avg", (2,2), output_size=(98,98))
    resize = ImageFilter("resize", output_size=(98,98))
    gaf = converter.convert(train0, conversions=["gaf"], conv_args=conv_args, imgFilter=resize)
    mtf = converter.convert(train1, conversions=["mtf"], conv_args=conv_args, imgFilter=imgFilter)
    rcp = converter.convert(train1, conversions=["rcp"], conv_args=conv_args)
    rcp = normalize_images(rcp, images=["rcp"], new_range=(0,1))

    img_train={
        "label": train1["label"],
        "label_val": train1["label_val"],
        "affected":{
            "gasf": gaf["affected"]["gasf"],
            "gadf":gaf["affected"]["gadf"],
            "mtf": mtf["affected"]["mtf"],
            "rcp": rcp["affected"]["rcp"]
        },
        "affected_val":{
            "gasf": gaf["affected_val"]["gasf"],
            "gadf":gaf["affected_val"]["gadf"],
            "mtf": mtf["affected_val"]["mtf"],
            "rcp": rcp["affected_val"]["rcp"]
        },
        "non_affected":{
            "gasf": gaf["non_affected"]["gasf"],
            "gadf":gaf["non_affected"]["gadf"],
            "mtf": mtf["non_affected"]["mtf"],
            "rcp": rcp["non_affected"]["rcp"]
        },
        "non_affected_val":{
            "gasf": gaf["non_affected_val"]["gasf"],
            "gadf":gaf["non_affected_val"]["gadf"],
            "mtf": mtf["non_affected_val"]["mtf"],
            "rcp": rcp["non_affected_val"]["rcp"]
        },
    }

    accs=[]
    vals=[]
    for img in [["gasf", "mtf", "rcp"], ["gadf", "mtf", "rcp"], ["gadf", "gasf", "mtf","rcp"]]:
        acc, val=validate_IMG(img_train, images=img, class_dict=fetcher.get_class_dict(), sweep=False)
        accs.append(acc)
        vals.append(val)
    print("Accuracy:")
    for item in accs:
        print(item)
    print("Validation:")
    for item in vals:
        print(item)