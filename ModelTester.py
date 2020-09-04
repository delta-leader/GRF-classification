import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from scipy import stats
from tensorflow.keras.utils import plot_model
import wandb
from wandb.keras import WandbCallback

from tensorflow import random as tfrand
from numpy.random import seed
from tensorflow.keras import backend


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

    def __init__(self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"], epochs=100, batch_size=32, class_dict={"HC":0, "H":1, "K":2, "A":3, "C":4}, filepath="output/"):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
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


    def perform_sweep(self, model, config, train, shape="1D", useNonAffected=True, loss=None, metrics=None):
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
            TODO
            - '2D' : 2 x 5 x time_steps (only possible when useNonAffected is True)

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
        TODO
        ValueError : If shape is not '1D' and non-affected data is not used.
        ValueError : If 'groups' is not one of [1, 2].
        ValueError : If 'groups' > 1 and non-affected data is not used.
        """

        if not isinstance(train, dict):
            raise TypeError("Training-data was not provided as a dictionary.")

        if shape != "1D" and not useNonAffected:
            raise ValueError("Using a 2 dimensional shape requieres the usage of both legs.")
        
        # verify that all necessary datasets are there
        _check_keys(train, useNonAffected, True)

        train_data = _extract_data_from_dict(train, val_set=False, useNonAffected=useNonAffected, shape=shape)
        val_data = _extract_data_from_dict(train, val_set=True, useNonAffected=useNonAffected, shape=shape)

        # get default values
        loss, metrics = self.__check_for_default(loss, metrics)

        model.compile(optimizer=config.optimizer, loss=loss, metrics=metrics)
        model.fit(train_data, to_categorical(train["label"]), validation_data=(val_data, to_categorical(train["label_val"])), batch_size=config.batch_size, epochs=config.epochs, verbose=2, callbacks=[WandbCallback(monitor='val_accuracy')])


    def test_model(self, model, train, config, shape="1D", groups=1, train_norm_dict=None, test_norm_dict=None, useNonAffected=True, test=None, loss=None, metrics=None, logfile="log.dat", model_name="Test-Model", plot_name="model.png", create_plot=True, show_plot=False, boost=False):
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
            - '1D' : time_steps x signals (either 5 or 10 if both sides are used)
            - '2D' : 2 x 5 x time_steps (only possible when useNonAffected is True)

        useNonAffected : bool, default=True
            If False, only the data from the affected side is used for training and testing.

        test_dict : dict, default=None
            Dictionary containing the test-set for the GRF-data.
            Must contain at least the keys 'affected' and 'label'.
            If useNonAffected is True must also contain the key 'non_affected'.

        optimizer : tf.keras.optimizers.Optimizer, default=None
            The optimizer to be used. If None, the default optimizer of the instance is used.

        loss : string, default=None
            The loss-function to be used. If None, the default loss-function of the instance is used.

        metrics : list of string, None
            The metrics to be recorded. If None, the default metrics of the instance are used.

        epochs : int, default=None
            The number of epochs to train the model for. If None, the default number of epochs of the instance is used.

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

        ----------
        Returns:

        ----------
        Raises:
        TypeError : If 'data_dict' or 'test_dict' are no dictionaries.
        ValueError : If one of the necessary keys is not available in one of the dictionaries.
        ValueError : If shape is not '1D' and non-affected data is not used.
        ValueError : If 'groups' is not one of [1, 2].
        ValueError : If 'groups' > 1 and non-affected data is not used.
        """

        if not isinstance(train, dict):
            raise TypeError("Training-data was not provided as a dictionary.")
        
        """TODO
        if shape != "1D" and not useNonAffected:
            raise ValueError("Using a 2 dimensional shape requieres the usage of both legs.")
        
        if groups not in [1, 2]:
            raise ValueError("Unsupported value for 'groups' ({}) please use one of {}.".format(groups, [1, 2]))

        if groups > 1 and not useNonAffected:
            raise ValueError("Creating 2 groups (affected/non-affected) requires the usage of non-affected data.")
        """
        
        # extract test data
        if test is not None:
            if not isinstance(test, dict):
                raise TypeError("Test-data was not provided as a dictionary.")
            # verify that all necessary datasets are there
            _check_keys(test, useNonAffected, False)

            test_data = _extract_data_from_dict(test, val_set=False, useNonAffected=useNonAffected, shape=shape)
        """
        if test_dict is not None:
            if not isinstance(test_dict, dict):
                raise TypeError("Test-data was not provided as a dictionary.")
            keys = ["affected", "label"]
            if useNonAffected:
                keys += ["non_affected"]
            _check_keys(keys, test_dict)
            test_data = test_dict["affected"]
            if test_norm_dict is not None:
                test_norm_data = test_norm_dict["affected"]
            if useNonAffected:
                if groups == 2:
                    if test_norm_dict is not None:
                        test_data = np.concatenate([test_data, test_dict["non_affected"]], axis=-1)
                        test_norm_data = np.concatenate([test_norm_data, test_norm_dict["non_affected"]], axis=-1)
                        test_data = [test_data, test_norm_data]
                    else:
                        test_data = [test_dict["affected"], test_dict["non_affected"]]
                else:
                    if shape == "1D":
                        test_data = np.concatenate([test_data, test_dict["non_affected"]], axis=-1)
                    else:
                        if shape =="2D_TxS":
                            test_data = np.concatenate([test_data, test_dict["non_affected"]], axis=-1)
                            test_data = np.expand_dims(test_data, axis=-1)
                        else:
                            if shape=="2D_Tx1xS":
                                test_data = np.concatenate([test_data, test_dict["non_affected"]], axis=-1)
                                test_data = np.expand_dims(test_data, axis=-2)
                            
                            else:
                                test_data = np.stack([test_data, test_dict["non_affected"]], axis=1)
                                test_data = np.swapaxes(test_data, -2, -1)
        """

        # verify that all necessary datasets are there
        _check_keys(train, useNonAffected, True)

        # extract train data
        train_data = _extract_data_from_dict(train, val_set=False, useNonAffected=useNonAffected, shape=shape)
        val_data = _extract_data_from_dict(train, val_set=True, useNonAffected=useNonAffected, shape=shape)
        """
        train_data = data_dict["affected"]
        val_data = data_dict["affected_val"]
        if train_norm_dict is not None:
            train_norm_data = train_norm_dict["affected"]
            val_norm_data = train_norm_dict["affected_val"]
        if useNonAffected:
            if groups == 2:
                if train_norm_dict is not None:
                    train_data = np.concatenate([train_data, data_dict["non_affected"]], axis=-1)
                    val_data = np.concatenate([val_data, data_dict["non_affected_val"]], axis=-1)
                    train_norm_data = np.concatenate([train_norm_data, train_norm_dict["non_affected"]], axis=-1)
                    val_norm_data = np.concatenate([val_norm_data, train_norm_dict["non_affected_val"]], axis=-1)
                    train_data = [train_data, train_norm_data]
                    val_data = [val_data, val_norm_data]
                else:
                    train_data = [data_dict["affected"], data_dict["non_affected"]]
                    val_data = [data_dict["affected_val"], data_dict["non_affected_val"]]
            else:
                if shape == "1D":
                    train_data = np.concatenate([train_data, data_dict["non_affected"]], axis=-1)
                    val_data = np.concatenate([val_data, data_dict["non_affected_val"]], axis=-1)
                else:
                    if shape == "2D_TxS":
                            train_data = np.concatenate([train_data, data_dict["non_affected"]], axis=-1)
                            train_data = np.expand_dims(train_data, axis=-1)
                            val_data = np.concatenate([val_data, data_dict["non_affected_val"]], axis=-1)
                            val_data = np.expand_dims(val_data, axis=-1)
                    else:
                        if shape == "2D_Tx1xS":
                            train_data = np.concatenate([train_data, data_dict["non_affected"]], axis=-1)
                            train_data = np.expand_dims(train_data, axis=-2)
                            val_data = np.concatenate([val_data, data_dict["non_affected_val"]], axis=-1)
                            val_data = np.expand_dims(val_data, axis=-2)
                        else:
                            train_data = np.stack([train_data, data_dict["non_affected"]], axis=1)
                            val_data = np.stack([val_data, data_dict["non_affected_val"]], axis=1)
                            train_data = np.swapaxes(train_data, -2, -1)
                            val_data = np.swapaxes(val_data, -2, -1)
        """

        # get default values
        loss, metrics = self.__check_for_default(loss, metrics)

        # Log settings
        model.compile(optimizer=config.optimizer, loss=loss, metrics=metrics)
        logfile = open(self.filepath + logfile, "a")
        logfile.write(model_name + ":\n")
        logfile.write("Optimizer: {}, Loss: {}, Metrics: {}\n".format(config.optimizer, loss, metrics))
        logfile.write("Training for {} epochs.\n\n".format(config.epochs))

        # use the Logger to print to file & stdout
        terminal = sys.stdout
        sys.stdout = Logger(logfile)
        print(model.summary())
        print("\n\n")
        # setting shuffle=False would disable shuffling before selecting batches -> yields slightly better results?
        train_history = model.fit(train_data, to_categorical(train["label"]), validation_data=(val_data, to_categorical(train["label_val"])), batch_size=config.batch_size, epochs=config.epochs, verbose=2, callbacks=[])
        sys.stdout = terminal

        # calculate maximum accuracy
        accuracy = max(train_history.history["accuracy"])
        val_accuracy = max(train_history.history["val_accuracy"])
        logfile.write("\nMaximum Accuracy for Train-Set: {} after {} epochs.\n".format(accuracy, train_history.history["accuracy"].index(accuracy)))
        logfile.write("Maximum Accuracy for Validation-Set: {} after {} epochs.\n".format(val_accuracy, train_history.history["val_accuracy"].index(val_accuracy)))

        # calculate minimum loss
        loss = min(train_history.history["loss"])
        val_loss = min(train_history.history["val_loss"])
        logfile.write("Minimum Loss for Train-Set: {} after {} epochs.\n".format(loss, train_history.history["loss"].index(loss)))
        logfile.write("Minimumm Loss for Validation-Set: {} after {} epochs.\n".format(val_loss, train_history.history["val_loss"].index(val_loss)))

        # Plot accuracy and loss
        if create_plot or show_plot:
            self.__plot(train_history.history, plot_name, create_plot, show_plot)

        # Confusion Matrix for validation-set
        predicted_labels = np.argmax(model.predict(val_data, batch_size=config.batch_size), axis=1)
        #TODO
        if boost:
            if "info_val" not in train.keys():
                raise ValueError("Key 'info_val' is not available, boosting can not be used.")
            logfile.write("\nConfusion Matrix for VALIDATION data (with boosting):\n")
            info = train["info_val"]
            sessions = info['SESSION_ID'].unique()
            predictions = []
            labels = []
            for session in sessions:
                session_indices = np.where(info["SESSION_ID"]==session)[0]
                session_pred = np.take(predicted_labels, session_indices, axis=0)
                predictions.append(stats.mode(session_pred)[0][0])
                session_labels = np.take(train["label_val"], session_indices, axis=0)
                assert np.all(session_labels == session_labels[0])
                labels.append(session_labels[0])
            _log_confusion_matrix(logfile, _create_confusion_matrix(np.array(predictions), np.array(labels)), self.class_dict)

        else:
            logfile.write("\nConfusion Matrix for VALIDATION data:\n")      
            _log_confusion_matrix(logfile, _create_confusion_matrix(predicted_labels, train["label_val"]), self.class_dict)

        # Confusion Matrix for test-set
        if test is not None:
            predicted_labels = np.argmax(model.predict(test_data, batch_size=config.batch_size), axis=1)
            if boost:
                if "info" not in test.keys():
                    raise ValueError("Key 'info' is not availablein TEST-data, boosting can not be used.")
                logfile.write("\nConfusion Matrix for TEST data (with boosting):\n")
                info = test["info"]
                sessions = info['SESSION_ID'].unique()
                predictions = []
                labels = []
                for session in sessions:
                    session_indices = np.where(info["SESSION_ID"]==session)[0]
                    session_pred = np.take(predicted_labels, session_indices, axis=0)
                    predictions.append(stats.mode(session_pred)[0][0])
                    session_labels = np.take(test["label"], session_indices, axis=0)
                    assert np.all(session_labels == session_labels[0])
                    labels.append(session_labels[0])
                _log_confusion_matrix(logfile, _create_confusion_matrix(np.array(predictions), np.array(labels)), self.class_dict)
            
            else:
                logfile.write("\nConfusion Matrix for TEST data:\n")
                _log_confusion_matrix(logfile, _create_confusion_matrix(np.argmax(predicted_labels, axis=1), test["label"]), self.class_dict)

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
        

    def test_image_model(self, model, data_dict, images=["gasf", "gadf", "mtf", "rcp"], useNonAffected=True, test_dict=None, optimizer=None, loss=None, metrics=None, epochs=None, batch_size=None, logfile="log.dat", model_name="Test-Model", plot_name="model.png", create_plot=True, show_plot=False, boost=False):
        """Compiles and fits the model for image data according to the specified settings.
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

        images : list, default=['gasf', 'gadf', 'mtf', 'rcp']
            Specifies the images to be used for the model.
            If multiple images are specified, the data is stacked along the channel axis.

        useNonAffected : bool, default=True
            If False, only the data from the affected side is used for training and testing.

        test_dict : dict, default=None
            Dictionary containing the test-set for the GRF-data.
            Must contain at least the keys 'affected' and 'label'.
            If useNonAffected is True must also contain the key 'non_affected'.

        optimizer : tf.keras.optimizers.Optimizer, default=None
            The optimizer to be used. If None, the default optimizer of the instance is used.

        loss : string, default=None
            The loss-function to be used. If None, the default loss-function of the instance is used.

        metrics : list of string, None
            The metrics to be recorded. If None, the default metrics of the instance are used.

        epochs : int, default=None
            The number of epochs to train the model for. If None, the default number of epochs of the instance is used.

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

        ----------
        Returns:

        ----------
        Raises:
        TypeError : If 'data_dict' or 'test_dict' are no dictionaries.
        TypeError : If 'images' is not a list
        ValueError : If 'images' does not contain at least 1 entry.
        ValueError : If one of the necessary keys is not available in one of the dictionaries.
        ValueError : If the specified images are not available in the dictionary.
        """

        if not isinstance(data_dict, dict):
            raise TypeError("Data was not provided as a dictionary.")

        if not isinstance(images, list):
            raise TypeError("Image formats were not provided as a list.")

        if len(images) < 1:
            raise ValueError("At least one image-format to use must be specified.")
       
        # extract test data
        if test_dict is not None:
            if not isinstance(test_dict, dict):
                raise TypeError("Test-data was not provided as a dictionary.")
            
            _check_keys(["affected", "label"], test_dict)
            _check_keys(images, test_dict["affected"])
            if useNonAffected:
                _check_keys(["non_affected"], test_dict)
                _check_keys(images, test_dict["non_affected"])
  
            first_image = True
            for image in images:
                if first_image:
                    test_data = test_dict["affected"][image]
                    first_image = False
                else :
                    test_data = np.concatenate([test_data, test_dict["affected"][image]], axis=-1)
                if useNonAffected:
                    test_data = np.concatenate([test_data, test_dict["non_affected"][image]], axis=-1)


        _check_keys(["affected", "affected_val", "label", "label_val"], data_dict)
        _check_keys(images, data_dict["affected"])
        if useNonAffected:
            _check_keys(["non_affected", "non_affected_val"], data_dict)
            _check_keys(images, data_dict["non_affected"])
        
        first_image = True
        for image in images:
            if first_image:
                train_data = data_dict["affected"][image]
                val_data = data_dict["affected_val"][image]
                first_image = False
            else:
                train_data = np.concatenate([train_data, data_dict["affected"][image]], axis=-1)
                val_data = np.concatenate([val_data, data_dict["affected_val"][image]], axis=-1)

            if useNonAffected:
                train_data = np.concatenate([train_data, data_dict["non_affected"][image]], axis=-1)
                val_data = np.concatenate([val_data, data_dict["non_affected_val"][image]], axis=-1)


        # get default values
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss
        if metrics is None:
            metrics = self.metrics
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        # Log settings
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logfile = open(self.filepath + logfile, "a")
        logfile.write(model_name + ":\n")
        logfile.write("Optimizer: {}, Loss: {}, Metrics: {}\n".format(optimizer, loss, metrics))
        logfile.write("Training for {} epochs.\n\n".format(epochs))

        # use the Logger to print to file & stdout
        terminal = sys.stdout
        sys.stdout = Logger(logfile)
        print(model.summary())
        print("\n\n")
        # setting shuffle=False would disable shuffling before selecting batches -> yields slightly better results
        train_history = model.fit(train_data, to_categorical(data_dict["label"]), validation_data=(val_data, to_categorical(data_dict["label_val"])), batch_size=batch_size, epochs=epochs, verbose=2)
        sys.stdout = terminal

        # calculate maximum accuracy
        accuracy = max(train_history.history["accuracy"])
        val_accuracy = max(train_history.history["val_accuracy"])
        logfile.write("\nMaximum Accuracy for Train-Set: {} after {} epochs.\n".format(accuracy, train_history.history["accuracy"].index(accuracy)))
        logfile.write("Maximum Accuracy for Validation-Set: {} after {} epochs.\n".format(val_accuracy, train_history.history["val_accuracy"].index(val_accuracy)))

        # calculate minimum loss
        loss = min(train_history.history["loss"])
        val_loss = min(train_history.history["val_loss"])
        logfile.write("Minimum Loss for Train-Set: {} after {} epochs.\n".format(loss, train_history.history["loss"].index(loss)))
        logfile.write("Minimumm Loss for Validation-Set: {} after {} epochs.\n".format(val_loss, train_history.history["val_loss"].index(val_loss)))

        # Plot accuracy and loss
        if create_plot or show_plot:
            self.__plot(train_history.history, plot_name, create_plot, show_plot)

        # Confusion Matrix for validation-set
        predicted_labels = np.argmax(model.predict(val_data, batch_size=batch_size), axis=1)
        if boost:
            if "info_val" not in data_dict.keys():
                raise ValueError("Key 'info_val' is not available, boosting can not be used.")
            logfile.write("\nConfusion Matrix for VALIDATION data (with boosting):\n")
            info = data_dict["info_val"]
            sessions = info['SESSION_ID'].unique()
            predictions = []
            labels = []
            for session in sessions:
                session_indices = np.where(info["SESSION_ID"]==session)[0]
                session_pred = np.take(predicted_labels, session_indices, axis=0)
                predictions.append(stats.mode(session_pred)[0][0])
                session_labels = np.take(data_dict["label_val"], session_indices, axis=0)
                assert np.all(session_labels == session_labels[0])
                labels.append(session_labels[0])
            _log_confusion_matrix(logfile, _create_confusion_matrix(np.array(predictions), np.array(labels)), self.class_dict)

        else:
            logfile.write("\nConfusion Matrix for VALIDATION data:\n")      
            _log_confusion_matrix(logfile, _create_confusion_matrix(np.argmax(predicted_labels, axis=1), data_dict["label_val"]), self.class_dict)

        # Confusion Matrix for test-set
        if test_dict is not None:
            predicted_labels = np.argmax(model.predict(test_data, batch_size=batch_size), axis=1)
            if boost:
                if "info" not in test_dict.keys():
                    raise ValueError("Key 'info' is not availablein TEST-data, boosting can not be used.")
                logfile.write("\nConfusion Matrix for TEST data (with boosting):\n")
                info = test_dict["info"]
                sessions = info['SESSION_ID'].unique()
                predictions = []
                labels = []
                for session in sessions:
                    session_indices = np.where(info["SESSION_ID"]==session)[0]
                    session_pred = np.take(predicted_labels, session_indices, axis=0)
                    predictions.append(stats.mode(session_pred)[0][0])
                    session_labels = np.take(test_dict["label"], session_indices, axis=0)
                    assert np.all(session_labels == session_labels[0])
                    labels.append(session_labels[0])
                _log_confusion_matrix(logfile, _create_confusion_matrix(np.array(predictions), np.array(labels)), self.class_dict)
            
            else:
                logfile.write("\nConfusion Matrix for TEST data:\n")
                _log_confusion_matrix(logfile, _create_confusion_matrix(np.argmax(predicted_labels, axis=1), test_dict["label"]), self.class_dict)

        logfile.close()

        return accuracy, loss, val_accuracy, val_loss


    def __plot(self, history, plot_name, save, show):
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
        """

        filename, ending = plot_name.split(".")

        # plot accuracy
        plt.plot(history["accuracy"])
        plt.plot(history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
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
    Raises:
    ValueError : If 'labels' and 'expected_labels' have a different shape.
    """

    if labels.shape != expected_labels.shape:
        raise ValueError("Something went wrong during the label prediction. Length of the final labels does not match.")

    conf_mat = np.zeros((5, 5), dtype=int)
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


def _extract_data_from_dict(data_dict, val_set, useNonAffected, shape):
    """Extracts the data needed for training/validation from the given dictionary, according to the settings.
    If 'usNonAffected' is true the data for the non_affected side is added according to the argument specified in 'shape'
    Possible modes are:
    'shape' == '1D' : Data format is not changed (i.e. either time-steps or time-steps x signals for the non concatenated case). Non_affected data is appended along the last axis.
    TODO

    Parameters:
    data_dict : dictionary
        Contains the GRF measurements. Must have at least the keys 'afftected' and 'label'.
    
    val_set : bool
        Whether to extract the validation-set (True) or the train-set (False).

    useNonAffected : bool
        If True, the data for the affected and non_affected side are combined, otherwise just the affected side is used.

    shape : string
        Specifies the purpose of the output shape.
        Possible values are ('1D' - used for 1DCNN, MLP)
    TODO

    ----------
    Returns:
    data : ndarray
        Numpy array containing the extracted data according to the parameters specified.
    """

    val_suffix = ""
    if val_set:
        val_suffix = "_val"

    data_keys = ["affected", "label"]
        
    if useNonAffected:
        data_keys += ["non_affected"]

    data = data_dict["affected"+val_suffix]

    if useNonAffected:
        if shape == "1D":
            data = np.concatenate([data, data_dict["non_affected"+val_suffix]], axis=-1)

    return data




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

        