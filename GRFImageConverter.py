import warnings
import numpy as np
import cupy as cp
import multiprocessing as mp
from ImageFilter import ImageFilter
from scipy.spatial.distance import pdist

#testing purposes only
from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from GRFPlotter import GRFPlotter
import pandas as pd
import time


class GRFImageConverter(object):
    """Converts the GRF data into images.

    Currently 3 types of images are supported:
    - Gramian Angular Field (GAF): both Summation (GASF) and Difference (GADF) images are created.
    - Markov Transition Field: Signals are divided into bins and probabilities of bin transitions are calculated (requieres the setting of #bins).
    - Recurrence Plots (RCP): Requires the setting of an embedding dimension and a time delay.

    Per se, all image conversions are able to work with normalized and raw GRF data under the assumption that the data fits
    certein pre-requierements and settings (depending on the selected image conversion).
    If those requierements are not met or settings are inappropriate, the output of the conversion might be junk.

    If desired, parallel processing on multi-core architecutres can be enabled.
    If available a GPU can be selected as the main unit of processing.
    Results for GPU and CPU might differ slightly due to the implementation of functions such as 'sin' or 'arccos'.

    Attributes:
    available_conv : list of string
        Exhaustive list of the availaable image conversion modes. Currently 'gaf', 'mtf' and 'rcp' are supported.

    useGpu : bool, default=False
        Specifies whether or not the GPU should be used as the main processing unit.

    parallel_threshold : int, default=100
        Specifies the amount of samples necessary to switch to parallel execution.
        If #samples >= parallel_threshold, the execution is split between all available cores of a multi-core architecture.
        Otherwise all computation is done on a single core.
    """
    
    def __init__(self, useGpu=False, parallel_threshold=100):
        self.available_conv = ["gaf", "mtf", "rcp"]
        self.useGpu = useGpu
        self.parallel_threshold = parallel_threshold


    def enableGpu(self):
        """Turns on GPU-usage for image conversion."""
        #TODO verify for all images
        self.useGpu = True


    def disableGpu(self):
        """Turns off GPU-usage for image conversions."""
        self.useGpu = False


    def set_parallel_threshold(self, parallel_threshold):
        """Sets the value of the parallel threshold.

        Parameters:
        parallel_threshold : int
            The parallel threshold to be used in order to enable multi-core execution.

        ----------
        Raises :
        TypeError : If the specified value is not an integer.
        """

        if not isinstance(parallel_threshold, int):
            raise TypeError("Parallel threshold can only be set to integer values.")

        self.parallel_threshold = parallel_threshold
    

    def convert(self, data_dict, conversions=None, conv_args=None, imgFilter=None):
        """Converts the provided data into into their corresponding converted images.
        This function is able to convert GRF-data into the following images:
        - Gramian Angular Field (GAF) images
        - Markov Transition Field (MTF) images
        - Recurrence Plot (RCP) images

        Parameters:
        data_dict: dict
            Dictionary containing the GRF-data to be converted. All datasets are converted except the one stored under 'label'.
            At least the keys 'affected' and 'non_affected' must exist for successfull conversion.

        conversions : str, list of str or None, default=None
            Only the images corresponding to the values of this parameter are created.
            Can either be as single string or a list of the following values:
            "gaf" -> creates GAF images (both GASF and GADF)
            "mtf" -> creates MTF images
            "rcp" -> creates RCP images
            If None, all images listed above are created.

        conv_args : dict
            Contains the arguments used for the image conversion.
            Depending on the desired conversions the following keys are used:
            - GAF: None, parameter is ignored. 
                   Note, however, that the computation might fail or return garbage if the signals are not within [-1, 1], 
                   since the arrcos-function is used to extract the polar coordinates.
            - MTF : 
                   1) num_bins : int
                        The number of bins used in the computation.
                   2) range : tuple of the form (min, max)
                        The range of the data.
                        #TODO assert global
            #TODO expand

        imgFilter: ImageFilter, default=None
            A filter to be applied to each converted image before the final output.
            If None, the raw images are returned.

        ----------
        Returns:
        dict : dicitonary containing the same keys as the input (except "label").
            Each entry consits of another dictionary containing at least one of the following keys (depending on the value specefied by 'conversions'):
            'gasf': ndarray of shape num_samples x width x height x channels
                    Containing the GASF conversion of the data.
            'gadf': ndarray of shape num_samples x width x height x channels
                    Containing the GADF conversion of the data.
            'mtf': ndarray of shape num_samples x width x height x channels
                    Containing the MTF conversion of the data.
            #TODO extend description
            Note: For GASF, GADF and MTF width = height = time_steps if no resize_filter is applied.

        ----------
        Raises:
        TypeError: If conversions is none of the following: None, str or list of str
        TypeError: If the imgFilter is neither None nor of type ImageFilter.
        ValueError: If one of the values specified in 'conversion' is none of the following: "gaf", "mtf" or "rcp".
        """

        if conversions is None:
            conversions = self.available_conv
        else:
            if isinstance(conversions, str):
                conversions = [conversions]
            if isinstance(conversions, list):
                for image in conversions:
                    if image not in self.available_conv:
                        raise ValueError("'{}' is not a valid conversion. Please specify any of 'gaf', 'mtf' or 'rcp'".format(image))
            else:
                raise TypeError("Conversions have to be of type str and specify any of the following: 'gaf', 'mtf' or 'rcp'")

        if imgFilter is not None:
            if not isinstance(imgFilter, ImageFilter):
                raise TypeError("Invalid value for 'imgFilter'. Please make sure that it is of type 'ImageFilter'.")

        return self.__convert_per_key(data_dict, conversions, conv_args, imgFilter)
    
  
    def __convert_per_key(self, data_dict, conversions, conv_args, imgFilter):
        """Calls the image conversion for each subset of the data corresponding to a key used in the 'data_dict' to store GRF-data.
        Additionally verifies the dimensions of each corresponding dataset.

        Parameters:
        data_dict : dict
            Dictionary containg the GRF-data.
            At least the entries 'affected' and 'non_affected' must exist within the dictionary.
            The conversion is called for all entries of the dictionary except 'label'.

        conversions :list of str
            Each value must be one of the following: "gaf", "mtf" or "rcp"
            Alias for the image conversion to call:
            "gaf" -> Creates Gramian Angular Field (GASF & GADF) images.
            "mtf" -> Creates Markov Transition Field images.
            "rcp" -> Creates Recurrence Plots

        conv_args : dict
            Containing the parameters used for some of the image conversions.

        imgFilter : ImageFilter
            A filter to be applied to the converted images.
            If None, the raw images are returned.

        ----------
        Returns:
        processed_data : dict, containing all keys with GRF-data (see above).
            Each key stores another dictionary with keys corresponding to the values specified in 'conversions'.
            The created images are stored under their corresponding key as follows:
            'gasf' -> Gramian Angular Summation Field (GASF)
            'gadf' -> Gramian Angular Difference Field (GADF)
            'mtf'  -> Markov Transition Field (MTF)
            'rcp'  -> Recurrence Plots (RCP)
             If an 'imgFilter' is passed, the filter is applied to each image within the data.

        ----------
        Raises:
        ValueError : If one of the arrays containing GRF-data is not 3-dimensional.
        """

        processed_data = {}

        # TODO remove time measurement
        start = time.time()
        for key in _get_keys(data_dict):
            if data_dict[key].ndim !=3:
                raise ValueError("Expected data to be stored in a 3-dimensinal array, but found array of dimension {}".format(data_dict[key].ndim))

            processed_data[key] = self.__call_conversion(data_dict[key], conversions, conv_args, imgFilter)

        end = time.time()
        print(end-start)

        return processed_data


    def __call_conversion(self, data, conversions, conv_args, imgFilter):
        """Calls the image conversion corresponding to the specified hardware environment.
        If 'useGpu' is True, all conversion are calculated on the GPU (using cupy)
        If calculations are done on the CPU and the number of samples is below the specified 'parallel_threshold',
        the code es executed in serial. Otherwise the computation is distributed between all available cores.

        Parameters:
        data : ndarray, ndim=3
            3-dimensional ndarray of shape num_samples x time_steps x channels.
            Contains the data to be converted. Each channel is converted independently of the others.

        conversions :list of str
            Each value must be one of the following: "gaf", "mtf" or "rcp"
            Alias for the image conversion to call:
            "gaf" -> Creates Gramian Angular Field (GASF & GADF) images.
            "mtf" -> Creates Markov Transition Field images.
            "rcp" -> Creates Recurrence Plots

        conv_args : dict
            Containing the parameters used for some of the image conversions.

        imgFilter : ImageFilter
            A filter to be applied to the converted images.
            If None, the raw images are returned.

        ----------
        Returns:
        result : dict
            Contains the result of the conversion. Dictionary containing at least one of the following keys (dpending on the values specified in 'conversions'):
            'gasf' -> Gramian Angular Summation Field (GASF)
            'gadf' -> Gramian Angular Difference Field (GADF)
            'mtf'  -> Markov Transition Field (MTF)
            'rcp'  -> Recurrence Plots (RCP)
             If an 'imgFilter' is passed, the filter is applied to each image within the data.
        """

        # swap the time_steps to be in the last dimension
        data = np.swapaxes(data, 1, 2)

        if self.useGpu:
            result = _convert_on_gpu(data, conversions, conv_args, imgFilter)
        else:
            # compute in serial if not enough data is available
            if data.shape[0] < self.parallel_threshold:
                result = _convert_on_cpu(np.apply_along_axis, data, conversions, conv_args, imgFilter)
            else:
                result = _convert_on_cpu(_apply_along_axis_parallel, data, conversions, conv_args, imgFilter)
       
        # swap the channel back to last dimension
        return _swap_channel_to_last(result)





def _get_keys(data):
    """Returns the relevant keys from the dictionary (i.e. the ones that store GRF data)

    Parameters:
    data : dict
        Dictinonary containing the following keys:
            'affected' : 3-dimensional ndarray of shape num_samples x time_steps x channels
            'non_affected' : 3-dimensional ndarray of shape num_samples x time_steps x channels
            'label' : 1-dimensional ndarray of shape num_samples x time_steps x channels
            'affected_val' (optional) : 3-dimensional ndarray of shape num_samples x time_steps x channels
            'non_affected_val' (optional) : 3-dimensional ndarray of shape num_samples x time_steps x channels

    ----------
    Returns:
    key_list : list of shape (2) or shape (4)
        Includes all keys from the input except 'label'

    ----------
    Raises:
    ValueError: If the provided dictionary contains less than 3 keys,
                or if it does not contain both of the following keys: 'affected' and 'non_affected'.

    """

    key_list = list(data.keys())

    if len(key_list) < 3:
        raise ValueError("The dictionary does not contain enough keys.")

    for key in ["affected", "non_affected"]:
        if key not in key_list:
            raise ValueError("Key: '{}' not found in the provided dictionary.".format(key))

    key_list.remove("label")

    return key_list


def _swap_channel_to_last(data_dict):
    """Swaps the channel to the last dimension for each array contained within the dictionary.

    Parameters:
    data_dict : dict
        Dictionary containing the data to be swapped. Each ndarray stored within the dictionary must have at least a dimensionality of 2.

    ----------
    Returns:
    data_dict : dict
        The same dictionary as the input, but with the channel-dimension swapped to the last position.
    """

    for key in data_dict.keys():
        data_dict[key] = np.moveaxis(data_dict[key], 1, -1)

    return data_dict


def _get_mtf_args(conv_args):
    """Extracts the conversion arguments used for the MTF conversion from the dictionary.

    Parameters:
    conv_args : dict
        Contains the conversion arguments. The following keys are use for MTF conversion:
        'num_bins' : int
            The number of bins used in the computation.
        'range' : tuple of the form (min, max)
            The range of the data.

    ----------
    Returns:
    num_bins : int
        The number of bins used in the computation.

    range_min : double
        The minimum value of the range of the data.

    range_max : double
        The maximum value of the range of the data.

    ----------
    Raises:
    TypeError : If 'conv_args' is not a dictionary.
    TypeError : It 'range' is not a tuple.
    TypeError : If 'num_bins' is not an integer.
    ValueError : If the key 'range' does not exist within the conversion arguments.
    ValueError : If 'range' is not of shape (2).
    ValueError : if the specified range does not fulfill min < max.
    """    

    if not isinstance(conv_args, dict):
        raise TypeError("Please specify the conversion arguments as a dictionary.")

    num_bins = 32
    if "num_bins" not in conv_args.keys():
        warnings.warn("Number of bins was not specified for MTF conversion. Will default to {}".format(num_bins))
    else:
        if isinstance(conv_args["num_bins"], int):
            num_bins = conv_args["num_bins"]
        else:
            raise TypeError("Number of bins needs to be specified as an integer.")
    
    if "range" not in conv_args.keys():
        raise ValueError("Please specify 'range' (tuple of  form (min, max)) as a conversion argument for MTF.")

    if isinstance(conv_args["range"], tuple):
        if len(conv_args["range"]) != 2:
            raise ValueError("The range must be specified of a tuple of shape (2) but found shape ({}).".format(len(conv_args["range"])))  
        range_min, range_max = conv_args["range"] 
        if range_min < range_max:
            return num_bins, range_min, range_max
        else :
            raise ValueError("The specified range does not fullfill min < max ({} < {}).".format(range_min, range_max))

    raise TypeError("The argument 'range' is not specified as a tuple.")


def _get_rcp_args(conv_args, signal_size, useGpu):
    """Extracts the conversion arguments used for the RCP conversion from the dictionary.

    Parameters:
    conv_args : dict
        Contains the conversion arguments. The following keys are use for MTF conversion:
        'dims' : int
            The number of embedding dimensions
        'delay' : int
            The time-delay used.
        'metric' str
            The distance metric used.

    signal_size : int
        The length of a single signal.

    useGpu : boolean
        Whether or not the conversion should be done on the GPU.

    ----------
    Returns:
    dims : int
        The number of embedding dimensions.

    range_min : int
        The time-delay.

    range_max : str
        The distance metric to be used (encoded as string).

    ----------
    Raises:
    TypeError : If 'conv_args' is not a dictionary.
    TypeError : It 'dims' is not an integer.
    TypeError : If 'delay' is not an integer.
    TypeError : If 'metric' is not a string.
    ValueError : If the number of embedding dimensions is equal to or exceeds the size of the signal (dims >= signal_size).
    ValueError : If the time-delay equal to or exceeds half the size of the signal (dims >= signal_size/2).
    ValueError : If 'metric' is not a valid value to be used for the conversion. Valid values are:
                 - CPU: All metrics accepted by scipy.spatial.distance.pdist()
                 - GPU: 'euclidean'
    """  

    if not isinstance(conv_args, dict):
        raise TypeError("Please specify the conversion arguments as a dictionary.")

    dims = 4
    if "dims" not in conv_args.keys():
        warnings.warn("The number of embedding dimensions was not specified for RCP conversion. Will default to {}".format(dims))
    else:
        if isinstance(conv_args["dims"], int):
            dims = conv_args["dims"]
        else:
            raise TypeError("The embedding dimensions has to be specified as an integer."
    )
    if dims >= signal_size:
        raise ValueError("The number of embedding dimensions is equal to or exceeds the size of the signal {} vs {}.".format(dims, signal_size))

    delay = 2
    if "delay" not in conv_args.keys():
        warnings.warn("The time-delay was not specified for RCP conversion. Will default to {}".format(delay))
    else:
        if isinstance(conv_args["delay"], int):
            dims = conv_args["delay"]
        else:
            raise TypeError("The time-delay has to be specified as an integer."
    )
    if delay >= signal_size/2:
        raise ValueError("The time delay is too large  {} >= {}/2.".format(dims, signal_size))

    metric = "euclidean"
    if "metric" not in conv_args.keys():
        warnings.warn("The metric for the distance calculations in the RCP conversion was not specified. Will default to '{}'".format(delay))
    if not isinstance(conv_args["metric"], str):
        raise TypeError("The distance metric has to be specified as a string")
    if useGpu:
        if metric not in ["euclidean"]:
            raise ValueError("Only euclidean distance metric is supported on the GPU.")
    else:
        valid_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]
        if metric not in valid_metrics:
            raise ValueError("The specified metric ('{}') is not a valid metric to be used with 'scipy.spatial.distance.pdist()".format(metric))

    return dims, delay, metric


def _check_range_mtf(mtf_range, data_range):
    """Verifies whether the supplied data range is within the specified range for the MTF conversion.
    Prints a warning if the data_range is bigger than the MTF-range.

    ----------
    Parameters:
    mtf_range : tuple of form (min, max)
        The range specified for the mtf-conversion.

    data_range : tuple of form (min, max)
        The actual range within the data.
    """

    if mtf_range[0] > data_range[0]:
        warnings.warn("The actual data range is larger then the specified one. Specified range is {}, data range is {}".format(mtf_range, data_range))

    if mtf_range[1] < data_range[1]:
        warnings.warn("The actual data range is larger then the specified one. Specified range is {}, data range is {}".format(mtf_range, data_range))


def _convert_on_cpu(func, data, conversions, conv_args, imgFilter):
    """Image conversion calculated on the CPU.
    All images specified in 'conversions' are created.

    Parameters:
    func : function
        The function to be called for the conversion. This is either of type 'serial' or 'parallel'.

    data : ndarray, ndim=3
            3-dimensional ndarray of shape num_samples x channels x time_steps.
            Contains the data to be converted. Each channel is converted independently of the others.

    conversions :list of str
            Each value must be one of the following: "gaf", "mtf" or "rcp"
            Alias for the image conversion to call:
            "gaf" -> Creates Gramian Angular Field (GASF & GADF) images.
            "mtf" -> Creates Markov Transition Field images.
            "rcp" -> Creates Recurrence Plots
    
    conv_args : dict
            Containing the parameters used for some of the image conversions.

    imgFilter : ImageFilter
        A filter to be applied to the converted images.
        If None, the raw images are returned.

    ----------
    Returns:
    converted_data : dict
        Contains the result of the conversion. Dictionary containing at least one of the following keys (dpending on the values specified in 'conversions'):
        'gasf' -> Gramian Angular Summation Field (GASF)
        'gadf' -> Gramian Angular Difference Field (GADF)
        'mtf'  -> Markov Transition Field (MTF)
        'rcp'  -> Recurrence Plots (RCP)
         If an 'imgFilter' is passed, the filter is applied to each image within the data.

    Raises:
    ValueError : If 'gaf' is specified as a conversion and the minimum value within the data is smaller than -1.
    """

    converted_data = {}
    for image in conversions:
        if image == "gaf":
            min_value = np.min(data)
            max_value = np.max(data)
            if min_value < -1:
                raise ValueError("Minimum value in the data is {} but arccos can only handle values within [-1, 1].".format(min_value))
            if max_value > 1:
                warnings.warn("Maximum value in the data is {}. All values >1 will be clipped to 1".format(max_value))

            gafs = func(_calc_gafs, axis=2, arr=data, imgFilter=imgFilter, args=None)
            converted_data["gasf"] = gafs[:, :, :, :, 0]
            converted_data["gadf"] = gafs[:, :, :, :, 1]

        elif image == "mtf":
            num_bins, range_min, range_max = _get_mtf_args(conv_args)
            _check_range_mtf((range_min, range_max), (np.min(data), np.max(data)))
            quantile_borders = np.linspace(range_min, range_max, num_bins, endpoint=False)[1:]
            converted_data["mtf"] = func(_calc_mtf, axis=2, arr=data, imgFilter=imgFilter, args=[quantile_borders])

        elif image == "rcp":
            dims, delay, metric = _get_rcp_args(conv_args, data.shape[2], False)
            converted_data["rcp"] = func(_calc_rcp, axis=2, arr=data, imgFilter=imgFilter, args=[dims, delay, metric])

        else:
            raise ValueError("No conversion defined for '{}'. Has to be one of 'gaf'/'mtf'/'rcp'.".format(image))

    return converted_data
                

def _apply_along_axis_parallel(func1d, axis, arr, imgFilter, args):
    """Like numpy.apply_along_axis(), but takes advantage of multiple cores (if available).
    The data is divided into (almost) equal chunks and distributed along the cores.
    Each core then applies the function along the specified axis for its subset of the data.

    Parameters:
    func1d : function
        The function to apply on the data.

    axis : int
        The axis along which to apply the function.
    
    arr : ndarray
        The array on which to apply the function.
    
    imgFilter : ImageFilter
        A filter to be applied to the converted images.
        If None, the raw images are returned.

    args : optional arguments to be passed as input to 'func1d'.
    
    ----------
    Returns:
    ndarray : The concatenated results of the computation from each core.
    """

    chunks = [(func1d, axis, sub_arr, imgFilter, args) for sub_arr in np.array_split(arr, mp.cpu_count())]
    pool = mp.Pool()
    partial_results = pool.map(_apply_along_axis_wrapper, chunks)
    pool.close()
    pool.join()

    return np.concatenate(partial_results)


def _apply_along_axis_wrapper(args):
    """Wrapper function for parallelisation of numpy.apply_along_axis().
    This is necessary because multiplrocessing.Pool().map() allows for only 1 argument to be passed to the function.

    Parameters:
    args : tupel (func1d, axis, data, imgFilter, args)
        Each element of the tupel encodes an argument for numpy.apply_along_axis().
        func1d : func, the function to apply
        axis : int, the axis along which to apply the function
        data : ndarray, the array on which to apply the function
        imgFilter : ImageFilter, the filter that should be applied to the converted images
        args : optional arguments to be passed as input to 'func1d'

    ----------
    Returns:
    ndarray : The array after the function has been applied.
    """

    (func1d, axis, data, imgFilter, args) = args
    return np.apply_along_axis(func1d, axis, data, imgFilter, args)


def _calc_gafs(signal, imgFilter, args):
    """Calculates both, the Gramian Angular Summation Field (GASF) and the
    Gramian Angular Difference Field (GADF) of the provided signal.

    Parameters:
    signal : 1-dimensional ndarray
        Contains the time-series to be converted.

    imgFilter : imageFilter
        A filter to be applied to the converted images.
        If None, the raw images are returned.

    args : list of arguments for the computation
        Disregarded for GAF calculation.

    ----------
    Returns:
    gafs : 3-dimensional ndarray of shape signal_size x signal_size x 2
        The last dimension conains the actual GASF(0) and GADF(1) values.
        If an 'imgFilter' is passed, this filter is applied to each image and the shape might change.
    """
    
    # because of precision errors, values can be slightly bigger than 1 which would lead to problems with arccos
    angle = np.arccos(np.minimum(1, signal))
            
    time_steps = signal.shape[0]
    gafs = np.empty([time_steps, time_steps, 2])

    for i in range(time_steps):
        for j in range(i+1):
            gafs[i, j, 0] = gafs[j, i, 0] = np.cos(angle[i] + angle[j])

            gafs[i, j, 1] = np.sin(angle[i] - angle[j])
            # equal to np.sin(signal[j] - signal[i]), but faster
            gafs[j, i, 1] = -gafs[i, j, 1]

    # apply Filter
    gafs = np.stack((_apply_filter(gafs[:, : , 0], imgFilter), _apply_filter(gafs[:, : , 1], imgFilter)), axis=-1)

    return gafs

def _calc_mtf(signal, imgFilter, args):
    """Calculates the Markov Transition Field (MTF) of the provided signal. 

    Parameters:
    signal : 1-dimensional ndarray
        Contains the time-series to be converted.

    imgFilter : imageFilter
        A filter to be applied to the converted images.
        If None, the raw images are returned.

    args : list of arguments for the computation 
        Must contain a 1-dimensional ndarray with the border values for each quantile bin.

    ----------
    Returns:
    mtf : 1-dimensional ndarray of shape signal_size x signal_size
        Contains the corresponding MTF-image to the input signal.
        If an 'imgFilter' is passed, this filter is applied to the image and the shape might change.
    """

    quantile_borders = args[0]
    # there is no border for the last bin
    num_bins = quantile_borders.shape[0] + 1
    bin_assignment = np.digitize(signal, quantile_borders)
    adj_mat = np.zeros([num_bins, num_bins], dtype=np.float32)

    # last value must be excluded (no transition)
    for i in range(signal.shape[0]-1):
        adj_mat[bin_assignment[i], bin_assignment[i+1]] += 1

    # normalize by column - maximum is needed to avoid div0
    adj_mat = adj_mat/np.maximum(1, adj_mat.sum(axis=0))

    mtf = np.zeros([signal.shape[0], signal.shape[0]], dtype=np.float32)
    for i in range(signal.shape[0]):
        for j in range(signal.shape[0]):
            mtf[i, j] = adj_mat[bin_assignment[i], bin_assignment[j]]

    return _apply_filter(mtf, imgFilter)


def _calc_rcp(signal, imgFilter, args):
    """Calculates the Recurrence Plot (RCP) of the provided signal. 

    Parameters:
    signal : 1-dimensional ndarray
        Contains the time-series to be converted.

    imgFilter : imageFilter
        A filter to be applied to the converted images.
        If None, the raw images are returned.

    args : list of arguments for the computation
        Must contain the embedding dimensions as the first value and
        the time-delay as the second value.
        The third value encodes the metric used for distance calculations.
        Refer to 'docs.scipy.org' for a list of possible values.

    ----------
    Returns:
    rcp : 1-dimensional ndarray of shape num_states x num_states
        The dimension num_states is defined by num_states = signal_size - (dims-1) * delay
        Contains the corresponding MTF-image to the input signal.
        If an 'imgFilter' is passed, this filter is applied to the image and the shape might change.
    """

    dims = args[0]
    delay = args[1]
    metric = args[2]
    num_states = signal.shape[0] - (dims-1) * delay

    # creating an array where each row corresponds to a single state (8 is the number of bytes for a float32 value)
    states = np.lib.stride_tricks.as_strided(signal, (num_states, dims), (8, dims*8))

    # using scipy to compute pairwise distances
    distances = pdist(states, metric=metric)

    # distances are stored in a triangular fashion
    # get indices of upper triangle
    upper_indices = np.triu_indices(distances, 1)
    # get indices of lower triangle
    lower_indices = np.trilu_indices(distances, 1)

    rcp = np.zeros(num_states, num_states)

    # fill with values
    rcp[upper_indices] = distances
    rcp[lower_indices] = distances

    return _apply_filter(rcp, imgFilter)


def _convert_on_gpu(data, conversions, conv_args, imgFilter):
    """Image conversion calculated on the GPU.
    All images specified in 'conversions' are created.

    data : ndarray, ndim=3
            3-dimensional ndarray of shape num_samples x channels x time_steps .
            Contains the data to be converted. Each channel is converted independently of the others.

    conversions :list of str
            Each value must be one of the following: "gaf", "mtf" or "rcp"
            Alias for the image conversion to call:
            "gaf" -> Creates Gramian Angular Field (GASF & GADF) images.
            "mtf" -> Creates Markov Transition Field images.
            "rcp" -> Creates Recurrence Plots
    
    conv_args : dict
            Containing the parameters used for some of the image conversions.

    imgFilter : ImageFilter
        A filter to be applied to the converted images.
        If None, the raw images are returned.

    ----------
    Returns:
    converted_data : dict
        Contains the result of the conversion. Dictionary containing at least one of the following keys (dpending on the values specified in 'conversions'):
        'gasf' -> Gramian Angular Summation Field (GASF)
        'gadf' -> Gramian Angular Difference Field (GADF)
        'mtf'  -> Markov Transition Field (MTF)
        'rcp'  -> Recurrence Plots (RCP)
         If an 'imgFilter' is passed, the filter is applied to each image within the data.
    """

    converted_data = {}
    for image in conversions:
        if image == "gaf":
            data = cp.asarray(data)
            min_value = cp.min(data)
            max_value = cp.max(data)
            if min_value < -1:
                raise ValueError("Minimum value in the data is {} but arccos can only handle values within [-1, 1].".format(min_value))
            if max_value > 1:
                warnings.warn("Maximum value in the data is {}. All values >1 will be clipped to 1".format(max_value))
            converted_data["gasf"], converted_data["gadf"] = _convert_to_gaf_gpu(data, imgFilter)

        elif image == "mtf":
            num_bins, range_min, range_max = _get_mtf_args(conv_args)
            _check_range_mtf((range_min, range_max), (np.min(data), np.max(data)))
            quantile_borders = np.linspace(range_min, range_max, num_bins, endpoint=False)[1:]
            converted_data["mtf"] = _convert_to_mtf_gpu(data, imgFilter, quantile_borders)

        elif image == "rcp":
            dims, delay, _ = _get_rcp_args(conv_args, data.shape[2], True)
            converted_data = _convert_to_rcp_gpu(cp.asarray(data), imgFilter, dims, delay)

        else:
            raise ValueError("No conversion defined for '{}'. Has to be one of 'gaf'/'mtf'/'rcp'.".format(image))

    return converted_data


def _convert_to_gaf_gpu(data, imgFilter):
    """Converts the provided data into Gramian Angular Field (GAF) images using the GPU for processing.

    Parameters:
    data : cupy array with ndim=3 and of shape num_samples x channels x time_steps.
        Contains the data to be converted. Each channel is converted independently of the others.

    imgFilter : ImageFilter
        A blur filter to be applied to each converted image.

    ----------
    Returns:
    gasf : ndarray of shape num_samples x channels x time_steps x time_steps
        Containing the GASF conversion of the data
    gadf : ndarray of shape num_samples x channels  x time_steps x time_steps
        Containing the GADF conversion of the data.
    If 'ImgFilter' != None, the filter is applied to each image after conversion.
    """

    # cut values because precision errors sometimes leads to values sligthly bigger than 1 which causes problems for the arccos
    angle = cp.arccos(cp.minimum(1, data))
    
    gaf_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void gaf(const float* angle, float* gasf, float* gadf) {
    int row_id = blockIdx.x;
    int col_id = threadIdx.x;
    gasf[row_id*blockDim.x+col_id] = cos(angle[row_id] + angle[col_id]);
    gadf[row_id*blockDim.x+col_id] = sin(angle[row_id] - angle[col_id]);
    }
    ''', 'gaf')

    gasf_samples = []
    gadf_samples = []
    for sample in angle[:]:
        gasf_channels = []
        gadf_channels = []
        for channel in sample:
            time_steps = angle.shape[2]
            gasf_gpu = cp.zeros((time_steps, time_steps), dtype=cp.float32)
            gadf_gpu = cp.zeros((time_steps, time_steps), dtype=cp.float32)
            gaf_kernel((time_steps,), (time_steps,), (channel, gasf_gpu, gadf_gpu))
            # apply filter
            gasf_channels.append(_apply_filter(cp.asnumpy(gasf_gpu), imgFilter))
            gadf_channels.append(_apply_filter(cp.asnumpy(gadf_gpu), imgFilter))
        gasf_samples.append(gasf_channels)
        gadf_samples.append(gadf_channels)
                
    return np.array(gasf_samples), np.array(gadf_samples)


def _convert_to_mtf_gpu(data, imgFilter, quantile_borders):
    """Converts the provided data into Gramian Angular Field (GAF) images using the GPU for processing.

    Parameters:
    data : ndarray with ndim=3 and of shape num_samples x channels x time_steps.
        Contains the data to be converted. Each channel is converted independently of the others.

    imgFilter : ImageFilter
        A blur filter to be applied to each converted image.

    quantile_borders : 1-dimensional ndarray 
        Contains the boarder values for each quantile bin.

    ----------
    Returns:
    mtf : ndarray of shape num_samples x channels x time_steps x time_steps
        Containing the MTF conversion of the data.
        If 'ImgFilter' != None, the filter is applied to each image after conversion.
    """

    # there is no border for the last bin
    num_bins = quantile_borders.shape[0] + 1

    mtf_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void mtf(const int* bin_assignment, const float* adj_mat, const int numBins, float* mtf) {
    int row_id = blockIdx.x;
    int col_id = threadIdx.x;
    mtf[row_id*blockDim.x+col_id] = adj_mat[bin_assignment[row_id] * numBins + bin_assignment[col_id]];
    }
    ''', 'mtf')

    markov_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void markov(const int* bin_assignment, const int numBins, float* adj_mat) {
    atomicAdd(&adj_mat[bin_assignment[threadIdx.x] * numBins + bin_assignment[threadIdx.x +1]], 1);
    }
    ''', 'markov')

    mtf_samples = []
    for sample in data[:]:
        mtf_channels = []
        for channel in sample:

            time_steps = channel.shape[0]
            # no cupy version available
            bin_assignment = np.digitize(channel, quantile_borders)
            bin_assignment = cp.asarray(bin_assignment, dtype=cp.int32)
            adj_mat = cp.zeros([num_bins, num_bins], dtype=cp.float32)

            # last value must be excluded (no transition)
            markov_kernel((1,), (time_steps-1,), (bin_assignment, num_bins, adj_mat))
            #for i in range(time_steps-1):
            #    adj_mat[bin_assignment[i], bin_assignment[i+1]] += 1

            # normalize by column - maximum is needed to avoid div0
            #adj_mat = cp.asarray(adj_mat)
            adj_mat = adj_mat/cp.maximum(1, adj_mat.sum(axis=0))
            #print(adj_mat)

            #bin_assignment = cp.asarray(bin_assignment, dtype=cp.int32)
            adj_mat = cp.asarray(adj_mat)
            mtf = cp.zeros([time_steps, time_steps], dtype=cp.float32)
            mtf_kernel((time_steps,), (time_steps,), (bin_assignment, adj_mat, num_bins, mtf))
            mtf_channels.append(_apply_filter(cp.asnumpy(mtf), imgFilter))

        mtf_samples.append(mtf_channels)

    return np.array(mtf_samples)


def _convert_to_rcp_gpu(data, imgFilter, dims, delay):
    """Converts the provided data into Recurrence Plots (RCP) images using the GPU for processing.
    Unless the CPU version this function supports only the 'euclidean'-distance metric.

    Parameters:
    data : cupy array with ndim=3 and of shape num_samples x channels x time_steps.
        Contains the data to be converted. Each channel is converted independently of the others.

    imgFilter : ImageFilter
        A blur filter to be applied to each converted image.

    dims : int
        The number of embedding dimensions.

    delay : int
        The time-delay used for the calculation.

    ----------
    Returns:
    rcp : ndarray of shape num_samples x channels x num_states x time_num_states
        Containing the RCP conversion of the data.
        The dimension num_states is defined by num_states = signal_size - (dims-1) * delay
    If 'ImgFilter' != None, the filter is applied to each image after conversion.
    """

    num_states = data.shape[2] - (dims-1) * delay
    
    rcp_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void gaf(const float* signal, const int delay, float* states) {
    int row_id = blockIdx.x;
    int col_id = threadIdx.x;
    states[row_id*blockDim.x+col_id] = signal[row_id+col_id*delay];
    }
    ''', 'rcp')

    rcp_samples = []
    for sample in data:
        rcp_channels = []
        for channel in sample:
            # get matrix of states (num_states x dims)
            states = cp.zeros((num_states, dims), dtype=cp.float32)
            rcp_kernel((num_states,), (dims,), (channel, delay, states))

            # compute pairwise differences
            rcp = states[:, cp.newaxis] - states
            rcp = cp.linalg.norm(rcp, axis=2)
            
            # apply filter
            rcp_channels.append(_apply_filter(cp.asnumpy(rcp), imgFilter))
        rcp_samples.append(rcp_channels)
                
    return np.array(rcp_samples)


def _apply_filter(img, imgFilter):
    """Applies the filter to the image.

    Parameters:
    img : ndarray with ndim=2
        Contains the image to be filtered

    imgFilter : ImageFilter
        The filter to be applied to the image.

    ----------
    Returns :
    ndarray : same shape as the input
        The filtered image, or the original (if no filter is passed).
    """

    if imgFilter is None:
        return img

    return imgFilter.apply(img)



"""
fetcher = DataFetcher("/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC")
converter = GRFImageConverter()
scaler = GRFScaler()
data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=False, scaler=scaler)

test = {
    "affected": data["affected"][0:1, :, :],
    "non_affected": data["non_affected"][0:1, :, :],
    "label": data["label"][0]
}
args ={
    "num_bins": 32,
    "range": (-1, 1)
}
image = converter.convert(data, conversions="mtf", conv_args=args)
print("CONVERSION FINISHED")
converter.enableGpu()
image_gpu = converter.convert(data, conversions="mtf", conv_args=args)

#plotter = GRFPlotter()
#plotter.plot_image(image, keys="affected")
#plotter.plot_image(image_gpu, keys="affected")
"""
"""
blur = ImageFilter("resize", (10,10), output_size=(50,50))
blur_image = converter.convert(test, imgFilter=blur)
plotter.plot_image(blur_image, keys="affected")

#converter.enableGpu()
#image_gpu = converter.convert_to_GAF(data, plot=False)
"""
#Compare GPU and CPU
"""
assert np.allclose(image["affected"]["gasf"], image_gpu["affected"]["gasf"], rtol=1e-05, atol=1e-06)
assert np.allclose(image["affected"]["gadf"], image_gpu["affected"]["gadf"], rtol=1e-05, atol=1e-06)
assert np.allclose(image["non_affected"]["gasf"], image_gpu["non_affected"]["gasf"], rtol=1e-05, atol=1e-06)
assert np.allclose(image["non_affected"]["gadf"], image_gpu["non_affected"]["gadf"], rtol=1e-05, atol=1e-06)
print("GPU SUCCESS!")


filelist = ["GRF_F_V_", "GRF_F_AP_", "GRF_F_ML_", "GRF_COP_AP_", "GRF_COP_ML_"] 
ver_data={}
for item in filelist:
    component_name = item[item.index("_")+1:-1].lower()
    new_data = pd.read_csv("/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"+item+"PRO_left.csv", header=0)
    # SESSION_ID 1338 is the first item in the TRAIN_BALANCED set (left side affected) with 10 Trials
    #ver_data[component_name]=new_data[(new_data["SESSION_ID"]==1338) & (new_data["TRIAL_ID"]==1)].drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]).values
    ver_data[component_name]=new_data[new_data["SESSION_ID"]==1338].drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]).values

ver_data = scaler.transform(ver_data)
keys = list(ver_data.keys())
arr = np.array(list(ver_data.values()))
arr = np.swapaxes(arr, 0, 1)
 


arr_test = np.swapaxes(arr, 1,2)
# Comapre original data to fetched one
#assert np.allclose(arr_test, test["affected"], rtol=1e-05, atol=1e-07)
"""

#Compare GPU
"""
arr_gpu = cp.asarray(arr, dtype=cp.float32)
angle = cp.arccos(arr_gpu)
for i in range(angle.shape[0]):
    j = 0
    for signal in angle[i]:
        n = signal.shape[0]
        #print(signal)
        gasf = cp.zeros((n, n), dtype=cp.float32)
        gadf = cp.zeros((n, n), dtype=cp.float32)
        gaf_kernel((n,), (n,), (signal, gasf, gadf))  # grid, block and arguments
        #print(j)
        #print(image["affected"]["gadf_gpu"][i,:,:,j])
        #print(gadf)
        assert cp.allclose(gasf, image["affected"]["gasf_gpu"][i, :, :, j], rtol=1e-05, atol=1e-06)
        assert cp.allclose(gadf, image["affected"]["gadf_gpu"][i, :, :, j], rtol=1e-05, atol=1e-06)
        j += 1
"""


#Compare CPU
"""
for k in range(5):
    angle = np.arccos(arr[0, k, :])
    num = arr[0, k].shape[0]
    gasf = np.empty([num, num])
    gadf = np.empty([num, num])
    for i in range(num):
        for j in range(i+1):
            gasf[i,j] = gasf[j,i] = np.cos(angle[i]+angle[j])
            gadf[i,j] = np.sin(angle[i]-angle[j])
            gadf[j,i] = np.sin(angle[j]-angle[i])
    #print(gasf)
    #print(image["affected"]["gasf"][0, :, :, k])
    assert np.allclose(image["affected"]["gasf"][0, :, :, k], gasf, rtol=1e-05, atol=1e-06), "Wrong GASF ({})".format(k)
    assert np.allclose(image["affected"]["gadf"][0, :, :, k], gadf, rtol=1e-05, atol=1e-06), "Wrong GADF ({})".format(k)
    #plt.imshow(gasf, cmap="jet")
    #plt.show()
    #plt.imshow(gadf, cmap="jet")
    #plt.show()
"""
