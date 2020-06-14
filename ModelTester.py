import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model


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


    def test_model(self, model, data_dict, useNonAffected=True, test_dict=None, optimizer=None, loss=None, metrics=None, epochs=None, batch_size=None, logfile="log.dat", model_name="Test-Model", plot_name="model.png", create_plot=True, show_plot=False):
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
        """

        if not isinstance(data_dict, dict):
            raise TypeError("Data was not provided as a dictionary.")
        
        # extract test data
        if test_dict is not None:
            if not isinstance(test_dict, dict):
                raise TypeError("Test-data was not provided as a dictionary.")
            keys = ["affected", "label"]
            if useNonAffected:
                keys += ["non_affected"]
            _check_keys(keys, test_dict)
            test_data = test_dict["affected"]
            if useNonAffected:
                test_data = np.concatenate([test_data, test_dict["non_affected"]], axis=-1)

        data_keys = ["affected", "affected_val", "label", "label_val"]
        if useNonAffected:
            data_keys += ["non_affected", "non_affected_val"]
        
        _check_keys(data_keys, data_dict)

        train_data = data_dict["affected"]
        val_data = data_dict["affected_val"]
        if useNonAffected:
            train_data = np.concatenate([train_data, data_dict["non_affected"]], axis=-1)
            val_data = np.concatenate([val_data, data_dict["non_affected_val"]], axis=-1)

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
        model.compile(optimizer, loss, metrics)
        logfile = open(self.filepath + logfile, "a")
        logfile.write(model_name + ":\n")
        logfile.write("Optimizer: {}, Loss: {}, Metrics: {}\n".format(optimizer, loss, metrics))
        logfile.write("Training for {} epochs.\n\n".format(epochs))

        # use the Logger to print to file & stdout
        terminal = sys.stdout
        sys.stdout = Logger(logfile)
        print(model.summary())
        print("\n\n")
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
        logfile.write("\nConfusion Matrix for VALIDATION data:\n")
        predicted_labels = model.predict(val_data, batch_size=batch_size)
        _log_confusion_matrix(logfile, _create_confusion_matrix(np.argmax(predicted_labels, axis=1), data_dict["label_val"]), self.class_dict)

        # Confusion Matrix for test-set
        if test_dict is not None:
            logfile.write("\nConfusion Matrix for TEST data:\n")
            predicted_labels = model.predict(test_data, batch_size=batch_size)
            _log_confusion_matrix(logfile, _create_confusion_matrix(np.argmax(predicted_labels, axis=1), test_dict["label"]), self.class_dict)

        logfile.close()

        return accuracy, loss


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
    ValueError : If the 'labels' and 'expected_labels' have a different shape.
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
    for i in range(5):
        logfile.write("   "+ reverse_dict[i])
    logfile.write("\n")
    logfile.write(np.array_repr(conf_mat))
    logfile.write("\n")


def _check_keys(keys, data_dict):
    """Verifies whether the given keys exist within the dictionary or not.

    Parameters:
    keys : list of string
        Contains the keys to verify.

    data_dict : dict
        The dictionary in which to look for the keys.

    ----------
    Raises:
        ValueError : If one of the provided keys does not exist within the dictionary.
    """

    for key in keys:
        if key not in data_dict.keys():
            raise ValueError("Key '{}' not available in the provided data-dictionary.".format(key))



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


def create_heatmap(data, kernels, filters, filename, showPlot=False):
    """Creates an accuracy heatmap for comparison of a model with different number of filters and kernel-sizes.

    Attributes:
    data : 2-dimensional list
        Contains the resulting accuracy of multiple runs with different filters and kernel-sizes.
        The first dimensions corresponds to the kernel_sizes, while the second represents the used number of filters.

    kernels : list
        Contains the kernel-sizes used.

    filters : list
        Contains the number of filters used.

    filename : string
        The filename under which the plot is saved.
    
    showPlot : bool, default=False
        If True the resulting plot is desplayed immediately (blocking).
    """

    fig, ax = plt.subplots()
    y = ["Kernel-Size: {}".format(i) for i in kernels]
    x = ["#Filters: {}".format(i) for i in filters]

    
    im, _ = _heatmap(np.array(data), y, x, ax=ax, vmin=0, cmap="Wistia", cbarlabel="Accuracy")
    _annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(lambda x, _ : "{:.2f}".format(x).replace("0.", ".")), size=10)
    fig.tight_layout()
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