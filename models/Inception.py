import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import tensorflow.keras
import wandb
import matplotlib as mpl
import numpy as np
from collections import namedtuple

from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from ModelTester import ModelTester, resetRand, wandb_init
from GRFImageConverter import GRFImageConverter
from ImageFilter import ImageFilter

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.applications import inception_v3, xception


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "Inception Sweep 2",
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
        }
    }

    return sweep_config


def create_config():
    """Creates the configuration file with the settings for the network described in
    "Scalable Classification of Univariate and Multivariate Time Series" (Karimi et al., 2018).
    """

    config = {
        "layers": 3,
        "neurons0": 800,
        "neurons1": 400,
        "neurons2": 100,
        "batch_norm": True,
        "dropout": 0.5,
        "activation": "relu",
        "final_activation": "softmax",
        "regularizer": None,
        "optimizer": "adam", #adadelta
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-08,
        "amsgrad": False,
        "batch_size": 32,
        "epochs": 100,
    }

    return config

def create_Inception(input_shape, config):
    """Creates the network according to the specifications in
    "Scalable Classification of Univariate and Multivariate Time Series" (Karimi et al., 2018)
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

    # add layers
    for layer in range(config.layers):
        model.add(units=Dense(getattr(config, "neurons{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer))
        model.add(BatchNormalization())
        model.add(Dropout(config.dropout))

    model.add(Dense(5, activation=config.final_activation, kernel_regularizer=config.regularizer))
    
    return model


def validate_Inception(train, test=None, class_dict=None, sweep=False):
    """Trains and tests the network as defined in 
    "Scalable Classification of Univariate and Multivariate Time Series" (Karimi et al., 2018).
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
            #model = create_Inception(input_shape=(train["affected"].shape[1],), config=config)
            input_tensor = Input(shape=(train["affected"].shape[1], train["affected"].shape[1], 1))
            model = InceptionV3(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling=None, classes=5)
            tester.perform_sweep(model, config, train, shape=None, useNonAffected=False)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=trainNN)
    
    else:
        filepath = "./output/FCN"
        #filepath = "models/output/MLP/WandB/Inception"
        config = create_config()
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, class_dict=class_dict)
        resetRand()
        model = create_Inception(input_shape=(train["affected"].shape[1],), config=config)
        tester.save_model_plot(model, "Inception_model.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, useNonAffected=False, config=config, test=test, shape=None, logfile="Inception.dat", model_name="Inception", plot_name="Inception.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))


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


def prepare_images(data_dict, norms, images, val_set=False, colormap=mpl.cm.jet, useNonAffected=True):
    """Arranges the images and transforms them into the format expected by the Inception network.

    Parameters:
    data_dict : dict
        Contains the image data.

    norms : dict
        Contains the name of each image and the corresponding range (i.e. minimum and maximum used for the colormap)

    images : list
        List with the names of the images to be used.
    
    val_set : bool, default=False
        If True, the validation set is converted instead of the train set.

    colormap : matplotlib.cm.colormap, default=jet
        The colormap to be used to convert the image into RGB-format

    useNonAffected : bool, default=True
        If True, data from both the affected and the unaffected side is used.

    ----------
    Returns:
    data : np.array
        The arranged and transformed data. All images are converted to RGB values based on the specified colormap and norms.
        All signals are concatenated along the colums, while different images and the non-affected leg are concatenated along the rows.
    """


    val_suffix = ""
    if val_set:
        val_suffix = "_val"

    img_data = []
    for image in images:
        if image not in data_dict["affected"+val_suffix].keys():
                raise ValueError("Image '{}' not available in data".format(image))
            
        data = np.concatenate([data_dict["affected"+val_suffix][image][:,:,:,i] for i in range(5)], axis=-1)
        if useNonAffected:
            new_data = np.concatenate([data_dict["non_affected"+val_suffix][image][:,:,:,i] for i in range(5)], axis=-1)
            data = np.concatenate([data, new_data], axis=-2)
        
        #vmin = norms[image][0]
        #vmax = norms[image][1]
        #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        #mapping = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        #data = np.apply_along_axis(lambda x: mapping.to_rgba(x, bytes=True), 0, data)
        #data = np.moveaxis(data, 1, -1)[:,:,:,:3]
        data = np.expand_dims(data, -1)
        img_data.append(data)
        
    data = np.concatenate(img_data, axis=1)
    print(data.shape)
        
    #print(data.shape)
    #mpl.pyplot.imshow(data[0])
    #mpl.pyplot.show()
    return data


def extract_features(data, useXception=False, pooling="max"):
    """Extractes the new feature set using a pretrained Inception network.

    Parameters:
    data : np.array
        Contains the images to be used as input for the network

    useXception bool, default=False
        If true, Xception will be used instead of Inception v3.

    pooling: string or None, default="max"
        The type of pooling applied to the output. If None, the output will be a 4D-Tensor instead of a 2D one.
        Valid values are 'max' and 'avg'.
    
    ----------
    Returns:
    new_data : tf.tensor
        The output from the Inception network (with the top-layer removed).
    """
    
    if useXception:
        data = xception.preprocess_input(data)
        model = Xception(include_top=False, weights="imagenet", pooling=pooling, input_shape=data.shape[1:])
    else:
        data = inception_v3.preprocess_input(data)
        model = InceptionV3(include_top=False, weights="imagenet", pooling=pooling, input_shape=data.shape[1:])
    
    new_data = model.predict(data)

    return new_data


if __name__ == "__main__":
    filepath = "../.."
    #filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False, clip=True)
    
    conv_args = {
        "images": ["gadf"],
        "filter": None,
        "filter_size": (7,7),
        "num_bins": 20,
        "range": (-1, 1),
        "dims": 3,
        "delay": 4,
        "metric": "euclidean"
    }
    converter = GRFImageConverter()
    #if this is used with sweep, tensorflow will use the CPU
    #converter.enableGpu()
    imgFilter = None
    if conv_args["filter"] is not None:
        imgFilter = ImageFilter(conv_args["filter"], conv_args["filter_size"])
    conv_train = converter.convert(train, conversions=get_conv_images(conv_args["images"]), conv_args=conv_args)
    conv_train["label"] = train["label"]
    if "label_val" in train.keys():
        conv_train["label_val"] = train["label_val"]

    norms = {
        "gadf": (-1, 1),
        "gasf": (-1, 1),
        "mtf": (0, 1)
    }
    
    img_train = prepare_images(conv_train, norms=norms, images=conv_args["images"])
    img_val = prepare_images(conv_train, norms=norms, images=conv_args["images"], val_set=True)
    #img_train = extract_features(img_train, useXception=False)
    #img_val = extract_features(img_val, useXception=False)

    data = {
        "affected": img_train,
        "label": train["label"],
        "affected_val": img_val,
        "label_val": train["label_val"]
    }
    
    validate_Inception(data, class_dict=fetcher.get_class_dict(), sweep=True)