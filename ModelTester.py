import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback

from tensorflow import random as tfrand
from numpy.random import seed
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


def resetRand(seed=1):
    """Resets the random number generators to the passed seed.
    Additionally sets the tensorflow environmental variable 'TF_DETERMINISTIC_OPS' to 1 for repeatable training.
    
    ----------
    Parameters:
    seed : int, default=1
        The seed to be used for the random number generators."""
    backend.clear_session()
    tfrand.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def wandb_init(config):
    """Initalizes the W&B init() call and returns the created configuration file.

    Parameters:
    config: dict
        Dictionary containing the desired configurations.

    ----------
    Returns:
    config: wandb.config
        The configuration file to be used.
    """

    wandb.init(project="diplomarbeit", config=config)
    return wandb.config


class ModelTester(object):
    """A testing framework for different CNN-models.
    Compiles and tests a model with different settings.
    Provides functionality to print and save the results into a file.
    Optinally plots for accuracy and loss can be created and saved.
    Additionally provides the functinality to create a schematic plot of the model (displaying the layers).

    Parameters :
    optimizer : tf.keras.optimizers.Optimizer, default="adam"
        The default optimizer to be used with this instance of ModelTester.

    loss : string, default="categorical_crossentropy"
        The default loss-function to be used with this instance of ModelTester.

    metrics : list of string, default=["accuracy"]
        The default metrics to be recorded by this instance of ModelTester.

    epochs : int, default=100
        The default number of epochs to run a single model.

    batch_size : int, default=32
        The default number of samples processed within a single batch.
  
    class_dict : dict, default={"HC":0, "H":1, "K":2, "A":3, "C":4}
            The dictionary used for encoding the class-label to integer values (and the other way round).

    filepath : string, default="output\"
        The filepath to the location where the output should be stored (e.g. plots and logfiles).
    """

    def __init__(self, loss="categorical_crossentropy", metrics=["accuracy"], class_dict={"HC":0, "H":1, "K":2, "A":3, "C":4}, filepath="output/"):
        self.loss = loss
        self.metrics = metrics
        self.class_dict = class_dict
        if filepath[-1] != "/":
            filepath += "/"
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath[:-1])


    def save_model_plot(self, model, filename="model_details.png"):
        """Creates a schematic plot of the model showing the different layers.

        Parameters:
        model : tensorflow.keras.model
            The model to create the plot for.
        """

        filename = self.filepath + filename
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)


    def predict_model(self, filepath, data_dict, val_set=False, boost=False, shape="1D", images=None, useNonAffected=True):
    

        if not isinstance(data_dict, dict):
            raise TypeError("Data was not provided as a dictionary.")

        # verify that all necessary datasets are there
        _check_keys(data_dict, useNonAffected, val_set)

        val_suffix = ""
        if val_set:
            val_suffix = "_val"

        data = _extract_data_from_dict(data_dict, val_set=val_set, useNonAffected=useNonAffected, shape=shape, images=images)
        model = load_model(filepath)

        predictions = model.predict(data)
        if boost:
            predicted_labels, labels = _majority_voting(data_dict, predictions, val_set=val_set)          
        else:
            predicted_labels = np.argmax(predictions, axis=1)
            labels = data_dict["label"+val_suffix]

        mask = data_dict["info"]['AFFECTED_SIDE'] == 2
        corr =predicted_labels[mask] == labels[mask]
        print(np.unique(corr, return_counts=True))
        #mask_pred = (predicted_labels != 1)
        #mask_orig = (labels == 1)
        #combined_mask = np.stack([mask_pred, mask_orig], axis=-1).all(axis=1)
        #print(np.unique(data_dict["label"], return_counts=True))
        #print(data_dict["info_val"]["CLASS_LABEL_DETAILED"][combined_mask].value_counts())
        

        conf_mat = _create_confusion_matrix(predicted_labels, labels)
        print("Confusion Matrix (Rows=Prediction, Columns=Real):")
        reverse_dict = dict((v,k) for k,v in self.class_dict.items())
        print("Order: {}".format(reverse_dict))
        print(conf_mat)
        count = np.sum(conf_mat)
        correct_count = np.sum(np.diagonal(conf_mat))
        accuracy = correct_count/count
        print("Accuracy: {}".format(accuracy))
        print("Correct predictions: {}".format(correct_count))
        print("Wrong predictions: {}".format(count-correct_count))


    def perform_sweep(self, model, config, train, shape="1D", images=None, useNonAffected=True, loss=None, metrics=None):
        """Performs a single sweep across the parameters set defined in the configuration file of W&B

        Parameters:
        model : tensorflow.keras.model
            The model to compile and train.

        config: wandb.config
            The configuration file of W&B. Contains the current settings to use.

        train : dict
            Contains the GRF-data. Must at least contain the keys 'affected', 'affected_val', 'label' and 'label_val'.
            Can optinally contain the data for the non_affected side under 'non_affected' and 'non_affected_val'

        shape : string, default='1D'
            Specifies how to extract the GRF-data from the dictionary.
            Options are:
            - '1D' : Input for 1D convolution or MLP. The data is concatenated along the last dimension (i.e. the input shape is identical to the output shape except that the last dimension will be twice as long if data from the non_affected side is used).
            - '2D_TS1' : Input for 2D convolution (with a single channel), time_steps x signals x 1
            - '2D_T1S' : Input for 2D convolution (with height=1, i.e. actually 1-dimensional), time_steps x 1 x signals
            - '2D_SST' : 2 x 5 x time_steps (only possible when useNonAffected is True, arranges the data in two rows, first affected, than non_affected)
            - '2D_TLS' : time_steps x 2 x 5 (only possible when useNonAffected is True, arranges the channels in two rows, first affected, than non_affected)
            - '2D_TSL': time_steps x 5 x 2 (only possible when useNonAffected is True, arranges the channels in two columns, first affected, than non_affected)
            - 'IMG_STACK' : Data format in which all transformed images are stacked along the last axis, resulting in width x height x singals * number of images
            - None : No transformation necessary, data is ready-to-use.

        useNonAffected : bool, default=True
            If False, only the data from the affected side is used for the sweep.

        loss : string, default=None
            The loss-function to be used when compiling this model.
            Will default to the loss function of the instance if None.

        metrics : list, default=None
            The metrics to be used when compiling this model.
            Will default to the metrics defined in the instance if None.

        ----------
        Raises:
        TypeError : If 'train' is not a dicitionary.
        ValueError : If one of the necessary keys is not available in the dictionaries.
        ValueError : If shape is '2D_SST' and the non-affected data is not used.
        """

        
        if not isinstance(train, dict):
            raise TypeError("Training-data was not provided as a dictionary.")
       
        # verify that all necessary datasets are there
        _check_keys(train, useNonAffected, True)

        train_data = _extract_data_from_dict(train, val_set=False, useNonAffected=useNonAffected, shape=shape, images=images)
        val_data = _extract_data_from_dict(train, val_set=True, useNonAffected=useNonAffected, shape=shape, images=images)

        # get default values
        loss, metrics = self.__check_for_default(loss, metrics)

        model.compile(optimizer=_configure_optimizer(config), loss=loss, metrics=metrics)
        model.fit(train_data, to_categorical(train["label"]), validation_data=(val_data, to_categorical(train["label_val"])), batch_size=config.batch_size, epochs=config.epochs, verbose=2, callbacks=[WandbCallback(monitor='val_accuracy')])


    def test_model(self, model, train, config, shape="1D", images=None, useNonAffected=True, test=None, loss=None, metrics=None, logfile="log.dat", model_name="Test-Model", plot_name="model.png", create_plot=True, show_plot=False, store_model=None, boost=False):
        #TODO
        """Compiles and fits the model accurding to the specified settings.
        A summary of the model and it's output are saved to the specified logfile.
        Calculates the maximum loss and accuracy achieved by the model.
        Optinally loss and accuracy can be saved/displayed as a plot.

        A confusion matrix is created for the validation set (and optionally for the test set) and saved to the logfile.

        Parameters:
        model : tensorflow.keras.model
            The model to compile and test.

        data_dict : dict
            Contains the GRF-data. Must at least contain the keys 'affected', 'affected_val', 'label' and 'label_val'.
            Can optinally contain the data for the non_affected side under 'non_affected' and 'non_affected_val'

        shape : string, default='1D'
            Specifies how to extract the GRF-data from the dictionary.
            Options are:
            - '1D' : Input for 1D convolution or MLP. The data is concatenated along the last dimension (i.e. the input shape is identical to the output shape except that the last dimension will be twice as long if data from the non_affected side is used).
            - '2D_TS1' : Input for 2D convolution (with a single channel), time_steps x signals x 1
            - '2D_T1S' : Input for 2D convolution (with height=1, i.e. actually 1-dimensional), time_steps x 1 x signals
            - '2D_SST' : 2 x 5 x time_steps (only possible when useNonAffected is True, arranges the data in two rows, first affected, than non_affected)
            - '2D_TLS' : time_steps x 2 x 5 (only possible when useNonAffected is True, arranges the channels in two rows, first affected, than non_affected)
            - '2D_TSL': time_steps x 5 x 2 (only possible when useNonAffected is True, arranges the channels in two columns, first affected, than non_affected)
            - 'IMG_STACK' : Data format in which all transformed images are stacked along the last axis, resulting in width x height x singals * number of images
            - None : No transformation necessary, data is ready-to-use.

        useNonAffected : bool, default=True
            If False, only the data from the affected side is used for training and testing.

        test_dict : dict, default=None
            Dictionary containing the test-set for the GRF-data.
            Must contain at least the keys 'affected' and 'label'.
            If useNonAffected is True must also contain the key 'non_affected'.

        loss : string, default=None
            The loss-function to be used. If None, the default loss-function of the instance is used.

        metrics : list of string, None
            The metrics to be recorded. If None, the default metrics of the instance are used.

        batch_size : int, default=Non
            The number of samples processed within a single batch. If None, the default batch_sie of the instance is used.

        logfile : string, default="log.dat"
            The name of the logfile to be saved.
            The logfile is saved at the location specified in the 'filepath' of the instance.

        model_name : string, default="Test-Model"
            The nome of the model written at the beginning of the logfile.

        plot_name : string, default="model.png"
            The name of the plotfile to be saved
            The plot is saved at the location specified in the 'filepath' of the instance.
        
        create_plot : bool, default=True
            If True, plots for accuracy and loss are created and saved under the name specified in 'plot_name'.

        show_plot : bool, default=False
            If True the plots are immediately displayed on creation (blocking).

        boost : bool, default=False
            If True, majority voting will be applied to produce a final classification for all trials within the same session.

        ----------
        Returns:

        ----------
        Raises:
        TypeError : If 'data_dict' or 'test_dict' are no dictionaries.
        ValueError : If one of the necessary keys is not available in one of the dictionaries.
        ValueError : If shape is '2D_SST' and the non-affected data is not used.
        ValueError : If 'boost' is true and 'info_val' is not in the train-set (or 'info' is not in the test-set).
        """

        if not isinstance(train, dict):
            raise TypeError("Training-data was not provided as a dictionary.")
               
        # extract test data
        if test is not None:
            if not isinstance(test, dict):
                raise TypeError("Test-data was not provided as a dictionary.")
            # verify that all necessary datasets are there
            _check_keys(test, useNonAffected, False)

            test_data = _extract_data_from_dict(test, val_set=False, useNonAffected=useNonAffected, shape=shape, images=images)

        # verify that all necessary datasets are there
        _check_keys(train, useNonAffected, True)

        # extract train data
        train_data = _extract_data_from_dict(train, val_set=False, useNonAffected=useNonAffected, shape=shape, images=images)
        val_data = _extract_data_from_dict(train, val_set=True, useNonAffected=useNonAffected, shape=shape, images=images)

        # get default values
        loss, metrics = self.__check_for_default(loss, metrics)

        # Log settings
        model.compile(optimizer=_configure_optimizer(config), loss=loss, metrics=metrics)
        logfile = open(self.filepath + logfile, "a")
        logfile.write(model_name + ":\n")
        logfile.write("Optimizer: {}, Loss: {}, Metrics: {}\n".format(config.optimizer, loss, metrics))
        logfile.write("Training for {} epochs.\n\n".format(config.epochs))

        # use the Logger to print to file & stdout
        terminal = sys.stdout
        sys.stdout = Logger(logfile)
        print(model.summary())
        print("\n\n")

        callbacks = []
        if store_model is not None:
            callbacks.append(ModelCheckpoint(filepath=store_model, save_weights_only=False, monitor="val_accuracy", mode="max", save_best_only=True, save_freq="epoch"))
        # setting shuffle=False would disable shuffling before selecting batches -> yields slightly better results?
        train_history = model.fit(train_data, to_categorical(train["label"]), validation_data=(val_data, to_categorical(train["label_val"])), batch_size=config.batch_size, epochs=config.epochs, verbose=2, callbacks=callbacks)
        

        # calculate maximum accuracy
        accuracy = max(train_history.history["accuracy"])
        val_accuracy = max(train_history.history["val_accuracy"])
        print("\nMaximum Accuracy for Train-Set: {} after {} epochs.".format(accuracy, train_history.history["accuracy"].index(accuracy)))
        print("Maximum Accuracy for Validation-Set: {} after {} epochs.".format(val_accuracy, train_history.history["val_accuracy"].index(val_accuracy)))

        # calculate minimum loss
        loss = min(train_history.history["loss"])
        val_loss = min(train_history.history["val_loss"])
        print("Minimum Loss for Train-Set: {} after {} epochs.".format(loss, train_history.history["loss"].index(loss)))
        print("Minimumm Loss for Validation-Set: {} after {} epochs.\n".format(val_loss, train_history.history["val_loss"].index(val_loss)))
        sys.stdout = terminal
        

        # Confusion Matrix for validation-set
        predictions = model.predict(val_data, batch_size=config.batch_size)
        if boost:
            logfile.write("\nConfusion Matrix for VALIDATION data (with boosting):\n")
            predicted_labels, labels = _majority_voting(train, predictions, val_set=True)
        else:
            predicted_labels = np.argmax(predictions, axis=1)
            labels = train["label_val"]
            logfile.write("\nConfusion Matrix for VALIDATION data:\n")      

        _log_confusion_matrix(logfile, _create_confusion_matrix(predicted_labels, labels), self.class_dict)
        

        # Plot accuracy and loss
        if create_plot or show_plot:
            self.__plot(train_history.history, plot_name, create_plot, show_plot, config.epochs)

        # Confusion Matrix for test-set
        if test is not None:
            predictions = model.predict(test_data, batch_size=config.batch_size)
            if boost:
                logfile.write("\nConfusion Matrix for TEST data (with boosting):\n")
                predicted_labels, labels = _majority_voting(test, predictions, val_set=False)
            else:
                predicted_labels = np.argmax(predictions, axis=1)
                labels = test["label"]
                logfile.write("\nConfusion Matrix for TEST data:\n")
                
            _log_confusion_matrix(logfile, _create_confusion_matrix(predicted_labels, labels), self.class_dict)

        logfile.close()

        return accuracy, loss, val_accuracy, val_loss

    
    def __check_for_default(self, loss, metrics):
        """Check whether any of the passed parameters is None and used the default values from the instance in such a case.

        Parameters:
        loss : string
            The loss-function to be used. If None, the default loss-function of the instance is used.

        metrics : list of string
            The metrics to be recorded. If None, the default metrics of the instance are used.
        ----------
        Returns:
        loss : string
            Either the passed value or the default from the instance (if None).
        metrics : list string
            Either the passed value or the default from the instance (if None).
        """

        if loss is None:
            loss = self.loss
        if metrics is None:
            metrics = self.metrics

        return loss, metrics


    def __plot(self, history, plot_name, save, show, xlimit):
        """Creates the corresponding plots to a training history.
        Saves or displays (or both) the plots depending on the settings.

        Parameters:
        history : tf.keras.callbacks.History
            Contains the training history of the model.

        plot_name : string
            The filename of the plot.

        save : bool
            If True the plot is saved to the file specified in 'plot_name'.

        show : bool
            If True the plot is immediately displayed.

        xlimit : int
            Range limit for the x-axis.
        """

        filename, ending = plot_name.split(".")

        # plot accuracy
        plt.plot(history["accuracy"])
        plt.plot(history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim(0, xlimit)
        plt.legend(["train", "validation"], loc="upper left")

        if save:
            plt.savefig(self.filepath + filename + "_accuracy." + ending)

        if show:
            plt.show()

        # plot loss
        plt.figure()
        plt.plot(history["loss"])
        plt.plot(history["val_loss"])
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0, xlimit)
        plt.legend(["train", "validation"], loc="upper right")

        if save:
            plt.savefig(self.filepath + filename + "_loss." + ending)

        if show:
            plt.show()

        plt.close("all")





def _create_confusion_matrix(labels, expected_labels):
    """Creates the confusion matrix between original labels and predicted labels.

    Parameters:
    labels : ndarray
        Contains the predicted class labels.

    expected_labels : ndarray
        Contains the original class labels.

    ----------
    Returns:
    conf_mat : nd.array
        Containing the confusion matrix

    ----------
    Raises:
    ValueError : If 'labels' and 'expected_labels' have a different shape.
    """

    if labels.shape != expected_labels.shape:
        raise ValueError("Something went wrong during the label prediction. Length of the final labels does not match.")

    _, counts = np.unique(expected_labels, return_counts=True)
    num_labels = counts.shape[0]
    conf_mat = np.zeros((num_labels, num_labels), dtype=int)
    for i in range(labels.shape[0]):
        conf_mat[labels[i], expected_labels[i]] += 1

    return conf_mat


def _log_confusion_matrix(logfile, conf_mat, class_dict):
    """Writes the confusion matrix to the logfile.

    Parameter:
    logfile : file
        The logfile to write to.
        
    conf_mat : ndarray of shape(5,5)
        Contains the Confusion matrix.

    class_dict : dict
        Contains the conversion from class labels to integers.
        Must contains entries from 0-4 (5 class labels).
    """

    logfile.write("Confusion Matrix (Rows=Prediction, Columns=Real):\n")
    logfile.write("Order: ")
    reverse_dict = dict((v,k) for k,v in class_dict.items())
    for i in range(len(reverse_dict)):
        logfile.write("   "+ reverse_dict[i])
    logfile.write("\n")
    logfile.write(np.array_repr(conf_mat))
    logfile.write("\n")


def _check_keys(data_dict, useNonAffected, val_set):
    """Verifies whether the necessary keys exist within the dictionary or not.
    Checks the following keys: ["affected", "label"] and additionaly ["non_affected"] if "useNonAffected" is True.
    If 'val_set' is True the following keys are verified as well: ["affected_val", "label_val"] and additionaly ["non_affected_val"] if "useNonAffected" is True.


    Parameters:
    data_dict : dict
        The dictionary in which to look for the keys.

    useNonAffected : bool,
        If True, the existence of the keys "non_affected"/"non_affected_val" are verified.

    val_set : bool,
        If True the keys for the validation set are checked in addition to the normal ones.

    ----------
    Raises:
        ValueError : If one of the provided keys does not exist within the dictionary.
    """

    keys = ["affected", "label"]
    if useNonAffected:
        keys += ["non_affected"]

    val_keys = []
    if val_set:
        for key in keys:
            val_keys += [key+"_val"]
    
    keys += val_keys

    for key in keys:
        if key not in data_dict.keys():
            raise ValueError("Key '{}' not available in the provided data-dictionary.".format(key))


def _extract_data_from_dict(data_dict, val_set, useNonAffected, shape, images):
    """Extracts the data needed for training/validation from the given dictionary, according to the settings.
    If 'usNonAffected' is true the data for the non_affected side is added according to the argument specified in 'shape'
    Possible modes are:
    'shape' == '1D' : Data format is not changed (i.e. either time-steps or time-steps x signals for the non concatenated case). Non_affected data is appended along the last axis.
    'shape' == '2D_TS1' : Data format is not changed (i.e. time-steps x signals), but a third dimension is added for the channels to meet the format keras expects (i.e. time-steps x signals x 1).
    'shape' == '2D_T1S' : Data format is changed to contain a new dimension in the middle (i.e. time-steps x 1 x channels).
    'shape' == '2D_SST' : Data format is changed to that the first two dimensions correspond to the signals an the last is the time (i.e. 2 x 5 x time-steps). Only valid if data for the non-affected side is used.
    'shape' == '2D_TLS' : Data format is changed to that the first dimension is time, the second is the leg and the last is the signals (i.e. time-steps x 2 x 5). Only valid if data for the non-affected side is used.
    'shape' == '2D_TSL' : Data format is changed to that the first dimension is time, the second is the signals and the last is the leg (i.e. time-steps x 5 x 2). Only valid if data for the non-affected side is used.
    'shape' == 'IMG_STACK' : Data format in which all transformed images are stacked along the last axis, resulting in width x height x singals * number of images
    'shape' == None : Data is already in the desired format, no modification is necessary.

    Parameters:
    data_dict : dictionary
        Contains the GRF measurements. Must have at least the keys 'afftected' and 'label'.
    
    val_set : bool
        Whether to extract the validation-set (True) or the train-set (False).

    useNonAffected : bool
        If True, the data for the affected and non_affected side are combined, otherwise just the affected side is used.

    shape : string
        Specifies the purpose of the output shape.
        Possible values are ('1D' - used for 1DCNN, MLP, LSTM, '2D_TS1' - used for 2DCNN, '2D_T1S' - experimental use only, '2D_SST' - used for Alharthi 2D, '2D_TLS' & '2D_TSL' used for comparison in 2DCNN, 'IMG_STACK' used for Image Transformations (Hatami))
    
    images : list
        Specifies the list of images to use in case of shape being equal to 'IMG_STACK'.

    ----------
    Returns:
    data : ndarray
        Numpy array containing the extracted data according to the parameters specified.

    ----------
    Raises:
    ValueError : If shape is not one of '1D', '2D_TS1', '2D_T1S' or '2D_SST' or None
    ValueError : If shape is '2D_SST', '2D_TLS' or '2D_TSL' and useNonAffected is False (can not arrange the channels in 2 dimensions in such a case).
    Value Error : If shape is 'IMG_STACK' and 'images' is None or empty.
    ValueError : If shape is 'IMG_STACK' and one of the images in not available in the dataset.
    """

    val_suffix = ""
    if val_set:
        val_suffix = "_val"

    if shape is None:
        return data_dict["affected"+val_suffix]
    
    if shape not in ["1D", "2D_TS1", "2D_T1S", "2D_SST", "2D_TLS", "2D_TSL", "IMG_STACK"]:
        raise ValueError("Shape '{}' is not a valid format, please select one of '1D', '2D_TS1', '2D_T1S', '2D_SST' '2D_TLS', '2DTSL' or 'IMG_STACK'.".format(shape))

    # Image data
    if shape == "IMG_STACK":
        if not isinstance(images, list) or len(images) < 1:
            raise ValueError("For using the image format (IMG_STACK), a list of valid images to be used needs to be specified.")

        first_image = True
        for image in images:
            if image not in data_dict["affected"+val_suffix].keys():
                raise ValueError("Image '{}' not available in data".format(image))
            if first_image:
                data = data_dict["affected"+val_suffix][image]
                first_image = False
            else:
                data = np.concatenate([data, data_dict["affected"+val_suffix][image]], axis=-1)
            if useNonAffected:
                data = np.concatenate([data, data_dict["non_affected"+val_suffix][image]], axis=-1)
        
        return data

    data = data_dict["affected"+val_suffix]

    # 2D data
    if shape in ["2D_SST", "2D_TLS", "2D_TSL"]:
        if not useNonAffected:
            raise ValueError("Shape '2DSST' can only be applied if the data from the non-affected side is used.")
        
        if shape == "2D_SST":
            data = np.stack([data, data_dict["non_affected"+val_suffix]], axis=1)
            data = np.swapaxes(data, -2, -1)
        if shape == "2D_TLS":
            data = np.stack([data, data_dict["non_affected"+val_suffix]], axis=2)
        if shape == "2D_TSL":
            data = np.stack([data, data_dict["non_affected"+val_suffix]], axis=-1)
  
        return data

    if useNonAffected:
        data = np.concatenate([data, data_dict["non_affected"+val_suffix]], axis=-1)

    if shape == "2D_TS1":
        data = np.expand_dims(data, axis=-1)

    if shape == "2D_T1S":
        data = np.expand_dims(data, axis=-2)

    return data


def _majority_voting(data, predictions, val_set=True):
    """Applies a majority voting process to the data. The predictions for all trials recorded during a single session are aggregated to produce the final classification.
    Only trials where at least one class has a probability of >0.4 are considered for the voting process.
    Voting is conducted by taking the mode across all valid trials. If the mode is ambigous (i.e. two or more classes are predicted by the same number of trials),
    the class with the highest probabilty in total (across all valid trials) is chosen.

    Parameters:
    info : dict
        Contains the original data. Must contain either 'info' (if 'val_set' is False) or 'info_val' otherwise.

    predictions : np.array
        Contains all predictions for the data (i.e. the output of the model).

    val_set : bool, default=True
        If True, the sessions are aggregated according to the IDs provided in 'info_val'.
        If False, 'info' is used instead.
    
    ----------
    Returns:
    voted_predictions : np.array
        Contains the final classification for each session.

    labels : np.array
        Contains the corresponding lables from the original data

    ----------
    Raises:
    ValueError : If either 'info' or 'info_val' is not present (depending on the value of 'val_set')
    """
    
    val_suffix = ""
    if val_set:
        val_suffix = "_val"
                
    if "info"+val_suffix not in data.keys():
        raise ValueError("Key 'info_val' is not available, boosting can not be used.")
    info = data["info"+val_suffix]
    sessions = info['SESSION_ID'].unique()
    voted_predictions = []
    labels = []
    for session in sessions:
        session_indices = np.where(info["SESSION_ID"]==session)[0]
        session_pred = np.take(predictions, session_indices, axis=0)
        mask = (session_pred > 0.4)
        mask = np.any(mask, axis=1)
        if not mask.any():
            mask = (session_pred > 0.3)
            mask = np.any(mask, axis=1)
        predicted_labels = np.argmax(session_pred[mask,:], axis=1)
        counts = np.bincount(predicted_labels)
        mode = np.where(counts==np.max(counts))[0]
        if mode.shape[0] > 1:
            #print("DRAW")
            probabilities = np.sum(session_pred[mask], axis=0)
            voted_predictions.append(np.argmax(probabilities))

        else:
            voted_predictions.append(mode[0])

        session_labels = np.take(data["label"+val_suffix], session_indices, axis=0)
        # TESTING only
        # assert np.all(session_labels == session_labels[0])
        labels.append(session_labels[0])

    return np.array(voted_predictions), np.array(labels)


def _configure_optimizer(config):
    """Sets up and configures a keras.optimizer according to the specifications in 'config'

    Parameters:
    config : wandb.config or namedtuple
        Contains the configuration for the optimizer.

    ----------
    Returns:
    optimzer : keras.optimizer
        The configured optimizer.

    ----------
    Raises:
    ValueError : If optimizer specified is not supported by this function.
    """

    optimizer = None

    if config.optimizer == "adam":
        optimizer = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon, amsgrad=config.amsgrad)
    
    if optimizer is None:
        raise ValueError("Optimizer '{}' is not supported, please specify a different one.")
    else:
        return optimizer





class Logger(object):
    """Helper object in order to support printing to a file and stdout at the same time.

    Parameters:
    logfile : string
        The filename for the logfile to write to.
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.logfile = logfile

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        """Writes the specified message to the logfile and stdout.

        Parameters:
        message : string
            The message to write.
        """

        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass


def create_heatmap(data, yaxis, xaxis, filename, yaxis_title="Kernel-Size", xaxis_title="#Filters", title=None, showPlot=False):
    """Creates an accuracy heatmap for comparison of a model with different number of elements onf y- and x-axis.

    Attributes:
    data : 2-dimensional list
        Contains the resulting accuracy of multiple runs with different elements
        The first dimensions corresponds to the y-axis and the second dimension corresponds to the x-axis.

    yaxis : list
        Contains the entries along the y-axis.

    xaxis : list
        Contains the entries along the x-axis.

    yaxis_title : string, default="Kernel-Size"
        Contains the the title used for the y-axis.

    xaxis_title : string, default="#Filters"
        Contains the title used for the x-axis.

    title : string, default=None
        Contains the title used for the plot.

    filename : string
        The filename under which the plot is saved.
    
    showPlot : bool, default=False
        If True the resulting plot is desplayed immediately (blocking).
    """

    fig, ax = plt.subplots()
    y = [yaxis_title+": {}".format(i) for i in yaxis]
    x = [xaxis_title+": {}".format(i) for i in xaxis]

    
    im, _ = _heatmap(np.array(data), y, x, ax=ax, vmin=0, cmap="Wistia", cbarlabel="Accuracy")
    _annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(lambda x, _ : "{:.2f}".format(x).replace("0.", ".")), size=10)
    fig.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(filename)

    if showPlot:
        plt.show()

    plt.close()

# function taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# function taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


#TODO Remove
def normalize_per_component(data_dict, keys, feature_range=(-1, 1)):
    """Componentwise MinMax-Normalzation of the entries corresponding to 'keys' in 'data_dict'

    Parameters:
    data_dict : dict
        Contains the GRF-data.
    keys : list
        Contains the keys corresponding to data_dict (e.g. "affected", "non_affected).
    feature_range : tupel of form (min, max)
        The range of the MinMax-Normalization.
    ----------
    Returns:
    result : dict:
        Dictionary containing the same information as 'data-dict' but with componentwise normalized data (for each sample) for all entries specified by 'keys'.
    """

    result ={}
    for key in data_dict.keys():
        if key not in keys:
            result[key] = data_dict[key]

    for key in keys:
        series =  np.swapaxes(data_dict[key], -2, -1)
        min_values = np.expand_dims(series.min(axis=2), axis=-1)
        max_values = np.expand_dims(series.max(axis=2), axis=-1)
        normalized_series = (series - min_values) / (max_values - min_values)
        normalized_series = normalized_series*(feature_range[1]-feature_range[0]) + feature_range[0]
        result[key] = np.swapaxes(normalized_series, -2, -1)

    return result

        