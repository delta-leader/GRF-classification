import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from ModelTester import ModelTester
from ModelTester import create_heatmap

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, AveragePooling1D
from tensorflow.keras.optimizers import Adam


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

    kernels : list of int, default=[7, 5, 3, 2]
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

    def __init__(self, conv_layers=4, filters=[12, 24, 48, 96], kernels=[7, 5, 3, 2], input_shape=(101, 10), conv_activation="relu", pool_size=2, dense_layers=1, dense_size=[50], dense_activation="relu", output_activation="softmax", kernel_regularizer="l2", dropOut=True):
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
            The 1D FCN model.
        """

        model = Sequential()     

        for layer in range(self.conv_layers):
            # 1st layer needs to define the input shape
            if layer == 0:
                model.add(Conv1D(filters=self.filters[layer], kernel_size=(self.kernels[layer]), strides=1, padding="same", activation=self.conv_activation, kernel_regularizer=self.kernel_regularizer, input_shape=self.input_shape))
            else:
                model.add(Conv1D(filters=self.filters[layer], kernel_size=self.kernels[layer], strides=1, padding="same", activation=self.conv_activation, kernel_regularizer=self.kernel_regularizer))


            model.add(MaxPooling1D(pool_size=self.pool_size))
            #model.add(AveragePooling1D(pool_size=self.pool_size))

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

    kernel_sizes = [2, 3, 5, 7, 9, 11, 13, 15]
    #num_filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    num_filters = [16]
    dense_size = [25, 50, 75, 100, 125, 150]
    filepath = "models/output/Alharthi/1D/1Layer/Dense"

    acc = []
    for kernel in kernel_sizes:
        kernel_acc = []
        for dense in dense_size:
            model = D1_CNN(conv_layers=1, filters=num_filters, kernels=[kernel], dense_size=[dense]).get_model()
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            accuracy, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="L1_K{}_F_16_D{}.dat".format(kernel, dense), model_name="Alharthi 1DCNN model - 1 Convolutional Layer", plot_name="L1_K{}_F16_D{}.png".format(kernel, dense))
            kernel_acc.append(accuracy)
        acc.append(kernel_acc)

    create_heatmap(acc, kernels=kernel_sizes, filters=dense_size, filename=filepath+"/Comparison_summary.png")


def test_2ConvLayer(train, test, optimizer, class_dict):
    """Test different settings for Alharthi's 1D model with 2 convolutional layer."""

    kernel_sizes = [2, 3, 5, 7, 9, 11, 13, 15]
    num_filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    filepath = "models/output/Alharthi/1D/2Layer"

    acc = []
    for kernel in kernel_sizes:
        kernel_acc = []
        for num_filter in num_filters:
            model = D1_CNN(conv_layers=2, filters=[128, num_filter], kernels=[2, kernel]).get_model()
            tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
            accuracy, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="L2_K{}_F{}.dat".format(kernel, num_filter), model_name="Alharthi 1DCNN model - 2 Convolutional Layer", plot_name="L2_K{}_F{}.png".format(kernel, num_filter))
            kernel_acc.append(accuracy)
        acc.append(kernel_acc)

    create_heatmap(acc, kernels=kernel_sizes, filters=num_filters, filename=filepath+"/Comparison_summary.png")


def test_original(train, test, optimizer, class_dict):
    """Tests the original network of the paper with different settings."""

    filepath = "models/output/Alharthi/1D/4Layer/AvgPooling"

    batch_sizes =[64, 128, 256, 584]
    model = D1_CNN().get_model()
    tester = ModelTester(filepath=filepath, optimizer=optimizer, class_dict=class_dict)
    acc = []
    #tester.save_model_plot(model, "Original_model.png")
    for batch in batch_sizes:
        accuracy, _ = tester.test_model(model, data_dict=train, test_dict=test, logfile="Original_B{}.dat".format(batch), model_name="Alharthi 1DCNN model - Original_B{}".format(batch), plot_name="Original_B{}.png".format(batch))
        acc.append(accuracy)

    print(acc)

    # No dropout
    # model = D1_CNN(dropOut=False).get_model()
    # tester.test_model(model, data_dict=train, test_dict=test, logfile="Original_NoDroputOut.dat", model_name="Alharthi 1DCNN model - No DropOut", plot_name="Original_NoDroputOut.png")

    # No regularization
    # model = D1_CNN(kernel_regularizer=None).get_model()
    # tester.test_model(model, data_dict=train, test_dict=test, logfile="Original_Noregularization.dat", model_name="Alharthi 1DCNN model - No Regularization", plot_name="Original_NoRegularization.png")



if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    train, test = fetcher.fetch_data(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2)
    optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    #test_1ConvLayer(train, test, optimizer, fetcher.get_class_dict())
    #test_2ConvLayer(train, test, optimizer, fetcher.get_class_dict())
    test_original(train, test, optimizer, fetcher.get_class_dict())