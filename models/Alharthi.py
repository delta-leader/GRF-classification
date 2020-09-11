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
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Conv2D, LSTM


def create_sweep_config():
    """Creates the configuration file with the settings used for a sweep in W&B

    Returns:
    sweep_config : dict
        Contains the configuration for the sweep.
    """

    sweep_config = {
        "name": "Alharthi1D - Hyperparameters",
        "method": "bayes",
        "description": "Find the optimal number of layers/neurons",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "batch_normalization": {
                "distribution": "categorical",
                "values": [True, False]
            },
            "regularizer": {
                "distribution": "categorical",
                "values": [None, "l2"]
            },
            "dropout_cnn": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 0.6
            },
            "dropout_mlp": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.3
            },
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
                "min": 1e-06,
                "max": 1e-08
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
    """Creates a 1-dimensional CNN according to the specificatins in
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

    # adds BatchNormalization specified
    def finish_layer(model, config):
        if config.batch_normalization:
            model.add(BatchNormalization())

    # add layers
    for layer in range(config.layers):
        model.add(Conv1D(filters=getattr(config, "filters{}".format(layer)), kernel_size=getattr(config, "kernel{}".format(layer)), activation=config.activation, kernel_regularizer=config.regularizer, padding=config.padding))
        finish_layer(model, config)
        model.add(MaxPooling1D(pool_size=config.pool_size))

    model.add(Flatten())
    model.add(Dropout(rate=config.dropout_cnn))
    model.add(Dense(config.neurons, activation=config.activation, kernel_regularizer=config.regularizer))
    model.add(Dropout(rate=config.dropout_cnn))

   
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
    if model == "1D":
        create_model = create_1D
        model_config = create_config_1D()
    
    if create_model is None:
        raise ValueError("'{}' does not specify a valid model. Supported values are '1D', '2D' or 'LSTM'.".format(model))
      
    if sweep:
        sweep_config = create_sweep_config()
        tester = ModelTester(class_dict=class_dict) 

        def train_MLP():
            config = wandb_init(model_config)
            resetRand()
            model = create_model(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            tester.perform_sweep(model, config, train, shape="1D", useNonAffected=True)
            
        sweep_id=wandb.sweep(sweep_config, entity="delta-leader", project="diplomarbeit")
        wandb.agent(sweep_id, function=train_MLP)
    
    else:
        filepath = "models/output/MLP/WandB/Alharthi"
        config = model_config
        config = namedtuple("Config", config.keys())(*config.values())
        tester = ModelTester(filepath=filepath, optimizer=config.optimizer, class_dict=class_dict)
        resetRand()
        model = create_model(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
        tester.save_model_plot(model, "Alharthi_1D.png")
        acc, _, val_acc, _ = tester.test_model(model, train=train, config=config, test=test, logfile="Alharthi_1D.dat", model_name="Alharthi - 1D", plot_name="Alharthi_1D.png")
        print("Accuracy: {}, Val-Accuracy: {}".format(acc, val_acc))





class D1_CNN(object):
    """Creates a 1-dimensional CNN (Convolutional Neural Network) as specified by Alharthi et al. (2019) in
    'Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing'.
    The basic block of such a network consits of the following sequence of layers:
        - 1D convolutional layer (activation='relu')
        - Max-Pooling layer
    The network is then built of a number of such blocks, followed by:
        - Fully-connected (dense) layer (activation='relu')
        - Fully-connected (dense) output layer (activation='softmax')
    In its basic form the network consists of 4 block before the final Fully-connected layers.

    Parameters:
    conv_layers : int, default=4
        Specifies the number of layers (1D-conv + max-pooling) to be used in the network.

    filters : list of int, default=[12, 24, 48, 96]
        Specifies the number of filters in each layer. The first value corresponds to the first layer and so on.

    kernels : list of int, default=[2, 2, 2, 2]
        Specifies the size of the kernel for each layer. The first value corresponds to the first layer and so on.

    input_size: tupel of int of the form (time_steps x channels), default=(101, 10)
        Specifies the input shape of the network (i.e. the shape of a single sample).

    conv_activation : str, default='relu'
        Specifies the activation function to use in each layer/block.

    pool_size : int, default=2
        Specifies the size of the max-pooling window.

    dense_layers : int, default=1
        Specifies the number of Fully-connected (dense) layers.

    dense_size : list of int, default=[50]
        Specifies the number of neurons in each Fully-connected (dense) layer. The first value corresponds to the first layer and so on.

    dense_activation : str, default='relu'
        Specifies the activation function to be used in the Fully connected (dense) layers (except the output layer).
 
    output_activation : str, default='softmax'
        Specifies the activation function to be used in the final (output) layer.

    kernel_regularizer : str, default='l2'
        Specifies the rebularization function for the kernel.

    dropOut : bool, default=True
        Specifies whether or not to use dropOut before the dense layers.

    ----------
    Attributes:
    classes : int, default = 5
        The number of classes (i.e. different labels) within the data.
        This specifies the size of the final dense layer.
    """

    def __init__(self, conv_layers=4, filters=[12, 24, 48, 96], kernels=[2, 2, 2, 2], input_shape=(101, 10), conv_activation="relu", pool_size=2, dense_layers=1, dense_size=[50], dense_activation="relu", output_activation="softmax", kernel_regularizer="l2", dropOut=True):
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernels = kernels
        self.input_shape = input_shape
        self.conv_activation = conv_activation
        self.pool_size = pool_size
        self.dense_layers = dense_layers
        self.dense_size = dense_size
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.kernel_regularizer = kernel_regularizer
        self.dropOut = dropOut
        self.__verify_settings()
        self.classes = 5


    def add_layer(self, layer_type="conv", filters=128, layer_size=5):
        """Adds a new layer to the model.

        Parameters:
        layer_type : str, default='conv'
            Specifies the type of layer to add. Can be one of the following:
            - 'conv': a new convolutional layer is added.
            - 'dense': a new Fully-connected (dense) layer is added.

        filters : int, default=128
            The number of filters in the new layer.
            Ignored if layer_type='dense'.

        kernel_size : int, default=5
            Specifies either the size of the kernel (convolutional layer),
            or the number of neurons in a Fully-connected (dense layer).
        """

        _check_pos_int(filters, "number of filters")
        _check_pos_int(layer_size, "layer_size")

        if layer_type == "conv":
            self.conv_layers += 1
            self.filters.append(filters)
            self.kernels.append(layer_size)
            return

        if layer_type == "dense":
            self.dense_layers += 1
            self.dense_size.append(layer_size)
            return
        
        raise ValueError("{} does not specify a valid layer type. Please use one of 'conv'/'dense'.".format(layer_type))


    def get_model(self):
        """Generates and returns the specified modell."""

        return self.__create_model()


    def __create_model(self):
        """Creates the model according to the specified settings.
        1 block (Conv1D, Max-Pooling) is added for each convolutional layer specified.
        1 dense layer is added for each dense layer specified.
        After all these arre chained together, the final output layer is added.

        Returns:
        model : tf.keras.model
            The 1D-convolutional model.
        """

        model = Sequential()     


        for layer in range(self.conv_layers):
            # 1st layer needs to define the input shape
            if layer == 0:
                model.add(Conv1D(filters=self.filters[layer], kernel_size=(self.kernels[layer]), strides=1, padding="same", activation=self.conv_activation, kernel_regularizer=self.kernel_regularizer, input_shape=self.input_shape))
            else:
                model.add(Conv1D(filters=self.filters[layer], kernel_size=self.kernels[layer], strides=1, padding="same", activation=self.conv_activation, kernel_regularizer=self.kernel_regularizer))
            
            #model.add(BatchNormalization())
            #model.add(MaxPooling1D(pool_size=self.pool_size, strides=2, padding="same"))
            model.add(AveragePooling1D(pool_size=self.pool_size, strides=2, padding="same"))
            #model.add(BatchNormalization())
            

        model.add(Flatten())
        if self.dropOut:
            model.add(Dropout(rate=0.5))

        for layer in range(self.dense_layers):
            model.add(Dense(self.dense_size[layer], activation=self.dense_activation, kernel_regularizer=self.kernel_regularizer))

        if self.dropOut:
            model.add(Dropout(rate=0.2))
        model.add(Dense(self.classes, self.dense_activation, kernel_regularizer=self.kernel_regularizer))

        return model


    def __verify_settings(self):
        """Verifies the inital configuration of the model.
        Checks whether all settings contain valid values and raises an error if not.

        Raises:
        ValueError : If the amount of layers is invalid or does not match the spcified attributes (filters, kernels, sizes, etc.)
        TypeError : If input shape is not a tuple of (time_steps, channels).
        """

        if self.conv_layers < 1:
            raise ValueError("Can't create a model with less than 1 convolutional layer!")
        if len(self.filters) != self.conv_layers:
            raise ValueError("{} convolutional layers where specified, but only {} filter dimensions were set.".format(self.conv_layers, len(self.filters)))
        if len(self.kernels) != self.conv_layers:
            raise ValueError("{} convolutional layers where specified, but only {} kernels were specified.".format(self.conv_layers, len(self.kernels))) 

        for filter_num, kernel in zip(self.filters, self.kernels):
            _check_pos_int(filter_num, "number of filters")
            _check_pos_int(kernel, "kernel size")

        if not isinstance(self.input_shape, tuple):
            raise TypeError ("Input shape is not a tuple.")

        if len(self.input_shape) !=2:
            raise TypeError("Input shape needs to be a tuple of dimension (2) (time_steps x channels), but is of dimension ({})".format(len(self.input_shape)))

        for dim in self.input_shape:
            _check_pos_int(dim, "input_shape")

        if self.pool_size < 1:
            raise ValueError("Can't create a Max-pooling layer with a pool-size < 1.")

        if self.dense_layers < 0:
            raise ValueError("Can't create a negative amount of dense layers.")

        if len(self.dense_size) != self.dense_layers:
            raise ValueError("{} dense layers where specified, but only {} sizes were specified.".format(self.dense_layers, len(self.dense_size))) 

        for size in self.dense_size:
            _check_pos_int(size, "dense_size")





class D2_CNN(object):
    """Creates a 2-dimensional CNN (Convolutional Neural Network) similar to the one used by Alharthi et al. (2019) in
    'Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing'.
    The basic block of such a network consits of the following sequence of layers:
        - 1D convolutional layer (activation='relu')
    The network is then built of a number of such blocks, followed by:
        - Fully-connected (dense) layer (activation='relu')
        - Fully-connected (dense) output layer (activation='softmax')
    In its basic form the network consists of 2 block before the final Fully-connected layers.

    Parameters:
    conv_layers : int, default=2
        Specifies the number of convolutional layers to be used in the network.

    filters : list of int, default=[128, 128]
        Specifies the number of filters in each layer. The first value corresponds to the first layer and so on.

    kernels : list of tuple, default=[(2,1), (1,2)]
        Specifies the size of the kernel for each layer. The first value corresponds to the first layer and so on.

    input_size: tupel of int of the form (width x height x time_steps), default=(2, 5, 101)
        Specifies the input shape of the network (i.e. the shape of a single sample).

    conv_activation : str, default='relu'
        Specifies the activation function to use in each convolutional layer

    padding : str, default='valid'
        The padding used by the convolutional layer. Can be one of 'valid'/'same'

    dense_layers : int, default=1
        Specifies the number of Fully-connected (dense) layers.

    dense_size : list of int, default=[100]
        Specifies the number of neurons in each Fully-connected (dense) layer. The first value corresponds to the first layer and so on.

    dense_activation : str, default='relu'
        Specifies the activation function to be used in the Fully connected (dense) layers (except the output layer).
 
    output_activation : str, default='softmax'
        Specifies the activation function to be used in the final (output) layer.

    kernel_regularizer : str, default='l2'
        Specifies the rebularization function for the kernel.

    dropOut : bool, default=True
        Specifies whether or not to use dropOut before the dense layers.

    ----------
    Attributes:
    classes : int, default = 5
        The number of classes (i.e. different labels) within the data.
        This specifies the size of the final dense layer.
    """

    def __init__(self, conv_layers=2, filters=[128, 128], kernels=[(2,1), (1,2)], input_shape=(2, 5, 101), conv_activation="relu", padding="valid", dense_layers=1, dense_size=[100], dense_activation="relu", output_activation="softmax", kernel_regularizer="l2", dropOut=True):
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernels = kernels
        self.input_shape = input_shape
        self.conv_activation = conv_activation
        self.padding = padding
        self.dense_layers = dense_layers
        self.dense_size = dense_size
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.kernel_regularizer = kernel_regularizer
        self.dropOut = dropOut
        self.__verify_settings()
        self.classes = 5


    def add_layer(self, layer_type="conv", filters=128, layer_size=5):
        """Adds a new layer to the model.

        Parameters:
        layer_type : str, default='conv'
            Specifies the type of layer to add. Can be one of the following:
            - 'conv': a new convolutional layer is added.
            - 'dense': a new Fully-connected (dense) layer is added.

        filters : int, default=128
            The number of filters in the new layer.
            Ignored if layer_type='dense'.

        kernel_size : int, default=5
            Specifies either the size of the kernel (convolutional layer),
            or the number of neurons in a Fully-connected (dense layer).
        """

        _check_pos_int(filters, "number of filters")
        _check_pos_int(layer_size, "layer_size")

        if layer_type == "conv":
            self.conv_layers += 1
            self.filters.append(filters)
            self.kernels.append(layer_size)
            return

        if layer_type == "dense":
            self.dense_layers += 1
            self.dense_size.append(layer_size)
            return
        
        raise ValueError("{} does not specify a valid layer type. Please use one of 'conv'/'dense'.".format(layer_type))


    def get_model(self):
        """Generates and returns the specified modell."""

        return self.__create_model()


    def __create_model(self):
        """Creates the model according to the specified settings.
        1 2d-convolution is added for each convolutional layer specified.
        1 dense layer is added for each dense layer specified.
        After all these arre chained together, the final output layer is added.

        Returns:
        model : tf.keras.model
            The 2D-convolutional model.
        """

        model = Sequential()     

        for layer in range(self.conv_layers):
            # 1st layer needs to define the input shape
            if layer == 0:
                model.add(Conv2D(filters=self.filters[layer], kernel_size=(self.kernels[layer]), strides=1, padding=self.padding, activation=self.conv_activation, kernel_regularizer=self.kernel_regularizer, input_shape=self.input_shape))
            else:
                model.add(Conv2D(filters=self.filters[layer], kernel_size=self.kernels[layer], strides=1, padding=self.padding, activation=self.conv_activation, kernel_regularizer=self.kernel_regularizer))
            model.add(BatchNormalization())

        model.add(Flatten())
        if self.dropOut:
            model.add(Dropout(rate=0.5))

        for layer in range(self.dense_layers):
            model.add(Dense(self.dense_size[layer], activation=self.dense_activation, kernel_regularizer=self.kernel_regularizer))
            model.add(BatchNormalization())

        model.add(Dense(self.classes, self.dense_activation, kernel_regularizer=self.kernel_regularizer))

        return model


    def __verify_settings(self):
        """Verifies the inital configuration of the model.
        Checks whether all settings contain valid values and raises an error if not.

        Raises:
        ValueError : If the amount of layers is invalid or does not match the spcified attributes (filters, kernels, sizes, etc.)
        TypeError : If input shape is not a tuple of (time_steps, channels).
        """

        if self.conv_layers < 1:
            raise ValueError("Can't create a model with less than 1 convolutional layer!")
        if len(self.filters) != self.conv_layers:
            raise ValueError("{} convolutional layers where specified, but only {} filter dimensions were set.".format(self.conv_layers, len(self.filters)))
        if len(self.kernels) != self.conv_layers:
            raise ValueError("{} convolutional layers where specified, but only {} kernels were specified.".format(self.conv_layers, len(self.kernels))) 

        for filter_num, kernel in zip(self.filters, self.kernels):
            _check_pos_int(filter_num, "number of filters")
            if len(kernel) != 2:
                raise ValueError("Please specify 2-dimensional kernels for 2D-convolution.")
            for x in kernel:
                _check_pos_int(x, "kernel size")

        if not isinstance(self.input_shape, tuple):
            raise TypeError ("Input shape is not a tuple.")

        if len(self.input_shape) !=3:
            raise TypeError("Input shape needs to be a tuple of dimension (3) (width x height x time_steps), but is of dimension ({})".format(len(self.input_shape)))

        for dim in self.input_shape:
            _check_pos_int(dim, "input_shape")

        if self.dense_layers < 0:
            raise ValueError("Can't create a negative amount of dense layers.")

        if len(self.dense_size) != self.dense_layers:
            raise ValueError("{} dense layers where specified, but only {} sizes were specified.".format(self.dense_layers, len(self.dense_size))) 

        for size in self.dense_size:
            _check_pos_int(size, "dense_size")





class LSTM_CNN(object):
    """Creates a LSTM (Long-Short Term Memory) Network as specified by Alharthi et al. (2019) in
    'Deep Learning for Ground Reaction Force Data Analysis: Application to Wide-Area Floor Sensing'.
    The basic block of such a network consits of the following sequence of layers:
        - LSTM (activation='tanh')
    The network is then built of a number of such blocks, followed by:
        - Fully-connected (dense) layer (activation='relu')
        - Batch-Normalization Layer
        - Fully-connected (dense) output layer (activation='softmax')
    In its basic form the network consists of 2 block before the final Fully-connected layers.

    Parameters:
    lstm_layers : int, default=2
        Specifies the number of LSTM layers to be used in the network.

    units : list of int, default=[100, 40]
        Specifies the number of memory units in each LSTM layer. The first value corresponds to the first layer and so on.

    input_size: tupel of int of the form (time_steps x channels), default=(101, 10)
        Specifies the input shape of the network (i.e. the shape of a single sample).

    lstm_activation : str, default='tanh'
        Specifies the activation function to use in each LSTM layer

    dense_layers : int, default=1
        Specifies the number of Fully-connected (dense) layers.

    dense_size : list of int, default=[20]
        Specifies the number of neurons in each Fully-connected (dense) layer. The first value corresponds to the first layer and so on.

    dense_activation : str, default='relu'
        Specifies the activation function to be used in the Fully connected (dense) layers (except the output layer).
 
    output_activation : str, default='softmax'
        Specifies the activation function to be used in the final (output) layer.

    kernel_regularizer : str, default='l2'
        Specifies the rebularization function for the kernel.

    dropOut : bool, default=True
        Specifies whether or not to use dropOut.

    ----------
    Attributes:
    classes : int, default = 5
        The number of classes (i.e. different labels) within the data.
        This specifies the size of the final dense layer.
    """

    def __init__(self, lstm_layers=2, units=[100, 40], input_shape=(101, 10), lstm_activation="relu", dense_layers=1, dense_size=[20], dense_activation="relu", output_activation="softmax", kernel_regularizer="l2", dropOut=True):
        self.lstm_layers = lstm_layers
        self.units = units
        self.input_shape = input_shape
        self.lstm_activation = lstm_activation
        self.dense_layers = dense_layers
        self.dense_size = dense_size
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.kernel_regularizer = kernel_regularizer
        self.dropOut = dropOut
        self.__verify_settings()
        self.classes = 5


    def add_layer(self, layer_type="lstm", units=50):
        """Adds a new layer to the model.

        Parameters:
        layer_type : str, default='lstm'
            Specifies the type of layer to add. Can be one of the following:
            - 'lstm': a new LSTM layer is added.
            - 'dense': a new Fully-connected (dense) layer is added.

        units : int, default=50
            The number of units (LSTM) or neurons (Dense) for the new layer.
        """

        _check_pos_int(units, "units")

        if layer_type == "lstm":
            self.lstm_layers += 1
            self.units.append(units)
            return

        if layer_type == "dense":
            self.dense_layers += 1
            self.dense_size.append(units)
            return
        
        raise ValueError("{} does not specify a valid layer type. Please use one of 'conv'/'dense'.".format(layer_type))


    def get_model(self):
        """Generates and returns the specified modell."""

        return self.__create_model()


    def __create_model(self):
        """Creates the model according to the specified settings.
        1 2d-convolution is added for each convolutional layer specified.
        1 dense layer is added for each dense layer specified.
        After all these arre chained together, the final output layer is added.

        Returns:
        model : tf.keras.model
            The 2D-convolutional model.
        """

        model = Sequential()     

        dropOut = 0.0
        if self.dropOut:
            dropOut = 0.2

        for layer in range(self.lstm_layers):
            # 1st layer needs to define the input shape
            if layer == 0:
                model.add(LSTM(units=self.units[layer], activation=self.lstm_activation, kernel_regularizer=self.kernel_regularizer, dropout=dropOut, recurrent_dropout=dropOut, input_shape=self.input_shape, return_sequences=True))
            else:
                # last layer does not need return sequence
                if layer == self.lstm_layers -1:
                    model.add(LSTM(units=self.units[layer], activation=self.lstm_activation, kernel_regularizer=self.kernel_regularizer, dropout=dropOut, recurrent_dropout=dropOut))
                else:
                    model.add(LSTM(units=self.units[layer], activation=self.lstm_activation, kernel_regularizer=self.kernel_regularizer, dropout=dropOut, recurrent_dropout=dropOut, return_sequences=True))

        model.add(BatchNormalization())
        #model.add(Flatten())
        if self.dropOut:
            model.add(Dropout(rate=0.5))

        for layer in range(self.dense_layers):
            model.add(Dense(self.dense_size[layer], activation=self.dense_activation, kernel_regularizer=self.kernel_regularizer))
            model.add(BatchNormalization())

        model.add(Dense(self.classes, self.dense_activation, kernel_regularizer=self.kernel_regularizer))

        return model


    def __verify_settings(self):
        """Verifies the inital configuration of the model.
        Checks whether all settings contain valid values and raises an error if not.

        Raises:
        ValueError : If the amount of layers is invalid or does not match the spcified attributes (layers, units, etc.)
        TypeError : If input shape is not a tuple of (time_steps, channels).
        """

        if self.lstm_layers < 1:
            raise ValueError("Can't create a model with less than 1 LSTM layer!")
        if len(self.units) != self.lstm_layers:
            raise ValueError("{} LSTM layers where specified, but only {} unit dimensions were set.".format(self.lstm_layers, len(self.units)))

        for unit in self.units:
            _check_pos_int(unit, "number of units")

        if not isinstance(self.input_shape, tuple):
            raise TypeError ("Input shape is not a tuple.")

        if len(self.input_shape) !=2:
            raise TypeError("Input shape needs to be a tuple of dimension (2) (time_steps x channels), but is of dimension ({})".format(len(self.input_shape)))

        for dim in self.input_shape:
            _check_pos_int(dim, "input_shape")

        if self.dense_layers < 0:
            raise ValueError("Can't create a negative amount of dense layers.")

        if len(self.dense_size) != self.dense_layers:
            raise ValueError("{} dense layers where specified, but only {} sizes were specified.".format(self.dense_layers, len(self.dense_size))) 

        for size in self.dense_size:
            _check_pos_int(size, "dense_size")





def _check_pos_int(value, param):
    """Verifies whether or not a value is a positive integer (>0).

    Parameters: 
    value : int
        The valueto be checked.

    param : string
        The name of the parameter (for printing in the error-message).

    ----------
    Raises:
    TypeError : If the passed value is not an integer.
        The error message specifies the name of the parameter passed in 'params'.
    """

    if not isinstance(value, int):
        raise TypeError("The {} needs to be an integer.".format(param))

    if value < 1:
        raise ValueError("The {} needs to be a positive integer (>0).".format(param))




def test_1ConvLayer(train, test, optimizer, class_dict):
    """Test different settings for Alharthi's 1D model with 1 convolutional layer"""

    #kernel_sizes = [2, 3, 5, 7, 9, 11, 13, 15]
    kernel_sizes = [3]
    num_filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    #num_filters = [16]
    dense_sizes = [25, 50, 75, 100, 125, 150]
    #dense_size = [50]
    filepath = "models/output/Alharthi/1D/1Layer/Dense/FixedKernel"
    optimizer_config = optimizer.get_config()

    acc = []
    for dense in dense_sizes:
        kernel_acc = []
        for num_filter in num_filters:
            resetRand()
            optimizer = optimizer.from_config(optimizer_config)
            model = D1_CNN(conv_layers=1, filters=[num_filter], kernels=kernel_sizes, dense_size=[dense]).get_model()
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            _, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="L1_DenseK3_D{}_F_{}.dat".format(dense, num_filter), model_name="Alharthi 1DCNN model - 1 Convolutional Layer (kernel_size=3)", plot_name="L1_DenseK3_D{}_F{}.png".format(dense, num_filter))
            kernel_acc.append(val_acc)
        acc.append(kernel_acc)

    create_heatmap(acc, yaxis=dense_sizes, xaxis=num_filters, filename=filepath+"/Comparison_L1_DenseK3_DvF.png")


def test_2ConvLayer(train, test, optimizer, class_dict):
    """Test different settings for Alharthi's 1D model with 2 convolutional layer."""

    kernel_sizes = [2, 3, 5, 7, 9, 11, 13, 15]
    num_filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    filepath = "models/output/Alharthi/1D/2Layer/AvgPooling"
    optimizer_config = optimizer.get_config()

    acc = []
    for kernel in kernel_sizes:
        kernel_acc = []
        for num_filter in num_filters:
            resetRand()
            optimizer = optimizer.from_config(optimizer_config)
            model = D1_CNN(conv_layers=2, filters=[16, num_filter], kernels=[2, kernel]).get_model()
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            _, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="L2_AvgPooling_K{}_F{}.dat".format(kernel, num_filter), model_name="Alharthi 1DCNN model - 2 Convolutional Layer (1. Layer: kernel_size=5, #filters=16, with AveragePooling instead of MaxPooling)", plot_name="L2_AvgPooling_K{}_F{}.png".format(kernel, num_filter))
            kernel_acc.append(val_acc)
        acc.append(kernel_acc)

    create_heatmap(acc, yaxis=kernel_sizes, xaxis=num_filters, filename=filepath+"/Comparison_L2_AvgPooling_KvF.png")


def test_original(train, test, optimizer, class_dict):
    """Tests the original network of the paper with different settings."""

    filepath = "models/output/Alharthi/1D/4Layer/NoReg&BatchNorm"

    batch_sizes = [32, 64, 128, 256, 584]
    model = D1_CNN(kernel_regularizer=None).get_model()
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    acc = []
    optimizer_config = optimizer.get_config()
    tester.save_model_plot(model, "NoReg&BatchNormAfterPool_model.png")
    for batch in batch_sizes:
        resetRand()
        model = D1_CNN(kernel_regularizer=None).get_model()
        optimizer = optimizer.from_config(optimizer_config)
        _, _, val_acc, _ = tester.test_model(model, data_dict=train, test_dict=test, batch_size=batch, optimizer=optimizer, logfile="NoReg&BatchNormAfterPool_B{}.dat".format(batch), model_name="Alharthi 1DCNN model - NoReg&BatchNormAfterPool_B{}".format(batch), plot_name="NoReg&BatchNormAfterPool_B{}.png".format(batch))
        acc.append(val_acc)

    print(acc)

    # No dropout
    # model = D1_CNN(dropOut=False).get_model()
    # tester.test_model(model, data_dict=train, test_dict=test, logfile="Original_NoDroputOut.dat", model_name="Alharthi 1DCNN model - No DropOut", plot_name="Original_NoDroputOut.png")

    # No regularization
    # model = D1_CNN(kernel_regularizer=None).get_model()
    # tester.test_model(model, data_dict=train, test_dict=test, logfile="Original_Noregularization.dat", model_name="Alharthi 1DCNN model - No Regularization", plot_name="Original_NoRegularization.png")


def test_2D(train, test, optimizer, class_dict):
    """Tests the 2D convolutional network of the paper with different settings."""

    filepath = "models/output/Alharthi/2D"

    model = D2_CNN(kernel_regularizer=None, dropOut=False).get_model()
    #tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    tester = ModelTester(filepath=filepath, optimizer="adam", class_dict=class_dict)
    #tester.save_model_plot(model, "default_model.png")
    accuracy, _, _, _ = tester.test_model(model, data_dict=train, shape="2D", test_dict=test, logfile="Default_norm.dat", model_name="Alharthi 2DCNN model - BatchNormalization", plot_name="Default_norm.png")


def test_LSTM(train, test, optimizer, class_dict):
    """Tests the LSTM network of the paper with different settings."""

    filepath = "models/output/Alharthi/LSTM"

    model = LSTM_CNN().get_model()
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    tester.save_model_plot(model, "LSTM_model.png")
    accuracy, _, _ , _ = tester.test_model(model, data_dict=train, shape="1D", test_dict=test, logfile="LSTM_original.dat", model_name="Alharthi LSTM model - Original", plot_name="LSTM_original.png")


if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train, test = fetcher.fetch_data(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2)
    #optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    #test_1ConvLayer(train, test, optimizer, fetcher.get_class_dict())
    #test_2ConvLayer(train, test, optimizer, fetcher.get_class_dict())
    #test_original(train, test, optimizer, fetcher.get_class_dict())
    #test_2D(train, test, optimizer, fetcher.get_class_dict())
    #test_LSTM(train, test, optimizer, fetcher.get_class_dict())

    validate_model(train, model="1D", test=None, class_dict=fetcher.get_class_dict(), sweep=True)