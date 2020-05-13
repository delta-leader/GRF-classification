import numpy as np
import cupy as cp
import multiprocessing as mp

#testing purposes only
from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from GRFPlotter import GRFPlotter
from ImageFilter import ImageFilter
import pandas as pd
import time


class GRFImageConverter(object):
    """Converts the GRF data into images.
    """
    
    def __init__(self, useGpu=False):
        self.useGpu = useGpu


    def enableGpu(self):
        """Turns on GPU-usage for image conversion."""
        #TODO verify for all images
        self.useGpu = True

    def disableGPU(self):
        """Turns off GPU-usage for image conversions."""
        self.useGpu = False
    
    
    def convert_to_GAF(self, data, image="BOTH", imgFilter=None):
        #TODO change function description
        """Converts the provided data into Gramian Angular Field (GAF) images.

        Parameters:
        data : TODO 3-dimensional ndarray of shape num_samples x time_steps x channels.
            Contains the data to be converted. Each channel is converted independently of the others.

        image : TODO Removed because they can be efficiently computed in one go
            Specifies with version of GAF to create (Gramian Angular Summation Field or Gramian Angular Difference field).
            If 'BOTH' is specified, both versions are created.

        imgFilter: ImageFilter
            A blur filter to be applied to each converted image before the final output.

        ----------
        Returns:
        conv_data : dicitonary containing the same keys as the input (except "label").
            Each entry consits of another dictionary containing at least one of the following keys (depending on the value specefied by 'image'):
            'gasf': ndarray of shape num_samples x time_steps x time_steps x channels
                    Containing the GASF conversion of the data.
            'gadf': ndarray of shape num_samples x time_steps x time_steps x channels
                    Containing the GADF conversion of the data.

        ----------
        Raises:
        TypeError: if 'image' is not a string.

        ValueError : If one of the arrays containing GRF-data is not 3-dimensional.
        """
        
        if self.useGpu:
            return _check_keys_and_apply(_convert_to_GAF_GPU, data, imgFilter)
        else:
            return _check_keys_and_apply(_convert_to_GAF_CPU, data, imgFilter)
   

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
    ValueError: If provided dictionary contains less than 3 keys,
                or if it does not contain either 'affected' or 'non_affected'.

    """

    key_list = list(data.keys())

    if len(key_list) < 3:
        raise ValueError("The dictionary does not contain enough keys.")

    for key in ["affected", "non_affected"]:
        if key not in key_list:
            raise ValueError("Key: '{}' not found in the provided dictionary.".format(key))

    key_list.remove("label")

    return key_list


def _calc_gafs(signal, imgFilter):
    """Calculates both, the Gramian Angular Summation Field (GASF) and the
    Gramian Angular Difference Field (GADF) of the provided signal.

    Parameters:
    signal : 1-dimensional ndarray
        Contains the time-series to be converted.

    imgFilter : imageFilter
        A blur filter to be applied to each calculated image.

    ----------
    Returns:
    gafs : 3-dimensional ndarray of shape signal_size x signal_size x 2
        The last dimension conains the actual GASF(0) and GADF(1) values.
        If an 'imgFilter' is passed, this filter is applied to each image.
    """
    
    time_steps = signal.shape[0]
    gafs = np.empty([time_steps, time_steps, 2])

    for i in range(time_steps):
        for j in range(i+1):
            gafs[i, j, 0] = gafs[j, i, 0] = np.cos(signal[i] + signal[j])

            gafs[i, j, 1] = np.sin(signal[i] - signal[j])
            # equal to np.sin(signal[j] - signal[i]), but faster
            gafs[j, i, 1] = -gafs[i, j, 1]

    # apply Filter
    gafs = np.stack((_apply_filter(gafs[:, : , 0], imgFilter), _apply_filter(gafs[:, : , 1], imgFilter)), axis=-1)

    return gafs


def _check_keys_and_apply(func, data, imgFilter):
    """Verifies the dimension of all keys containing GRF-data and applies a function to the corresponding values.

    Parameters:
    func : function
        The function to apply to the GRF-data.

    data : dict
        Dictionary containg the GRF-data.
        The function is only applied to the data stored in any of the following keys: 'affected', 'non_affected', 'affected_val' and 'non_affected_val'.

    imgFilter : ImageFilter
        A blur filter to be applied to the converted image.
        If None, no filter is applied.

    ----------
    Returns:
    processed_data : dict, containing all keys with GRF-data (see above).
        Each key stores the result of applying the function on the corresponding values.
        If none of the above keys exists, an empty dictionary is returned.
        If an 'imgFilter' is passed, the filter is applied to each image within the data.


    ----------
    Raises:
    ValueError : If one of the arrays containing GRF-data is not 3-dimensional.
    """

    processed_data = {}

    start = time.time()
    for key in _get_keys(data):
        if data[key].ndim !=3:
            raise ValueError("Expected data to be stored in a 3-dimensinal array, but found array of dimension {}".format(data[key].ndim))

        processed_data[key] = func(data[key], imgFilter)

    end = time.time()
    print(end-start)

    return processed_data


def _apply_along_axis_parallel(func1d, axis, data, num_processes, imgFilter):
    """Like numpy.apply_along_axis(), but takes advantage of multiple cores (if available).
    The data is divided into (almost) equal chunks and distributed along the cores.
    Each core then applies the function along the specified axis for its subset of the data.

    Parameters:
    func1d : function
        The function to apply on the data.

    axis : int
        The axis along which to apply the function.
    
    data : ndarray
        The array on which to apply the function.
    
    num_processes : int
        The number of parallel processes.

    imgFilter : ImageFilter
        A blur filter to be applied to each converted image.
    
    ----------
    Returns:
    ndarray : The concatenated results of the computation from each core.
    """

    chunks = [(func1d, axis, sub_arr, imgFilter) for sub_arr in np.array_split(data, num_processes)]
    pool = mp.Pool()
    partial_results = pool.map(_apply_along_axis_wrapper, chunks)
    pool.close()
    pool.join()

    return np.concatenate(partial_results)


def _apply_along_axis_wrapper(args):
    """Wrapper function for parallelisation of numpy.apply_along_axis().
    This is necessary because multiplrocessing.Pool().map() allows for only 1 argument to be passed to the function.

    Parameters:
    args : tupel (func1d, axis, data)
        Each element of the tupel encodes an argument for numpy.apply_along_axis().
        func1d : func, the function to apply
        axis : int, the axis along which to apply the function
        data : ndarray, the array on which to apply the function
        imgFilter : ImageFilter, the filter that should be applied to the converted images

    ----------
    Returns:
    ndarray : The array after the function has been applied.
    """

    (func1d, axis, data, imgFilter) = args
    return np.apply_along_axis(func1d, axis, data, imgFilter)


def _convert_to_GAF_CPU(data, imgFilter):
    """Converts the provided data into Gramian Angular Field (GAF) images using the CPU for processing.
    Trys to take advantage of parallel_processing if multiple cores are available.

    Parameters:
    data : ndarray with ndim=3 and of shape num_samples x time_steps x channels.
        Contains the data to be converted. Each channel is converted independently of the others.

    imgFilter : ImageFilter
        A blur filter to be applied to each converted image.

    ----------
    Returns:
    gaf : dict
        Dictionary containing the following keys:
        'gasf': ndarray of shape num_samples x time_steps x time_steps x channels
                Containing the GASF conversion of the data.
        'gadf': ndarray of shape num_samples x time_steps x time_steps x channels
                Containing the GADF conversion of the data.
        If 'ImgFilter' != None, the filter is applied to each image after conversion.
    """
    
    gaf = {}
    # because of precision errors, values can be slightly bigger than 1 which would lead to problems with arccos
    # TODO maybe implement in a better fashion
    angle = np.arccos(np.minimum(1, data))
            
    # swap time_steps and channel so that apply_along_axis can be used
    angle = np.swapaxes(angle, 1, 2)

    # if the amount of data is too low, compute serial, otherwise parallel
    cpu_count = mp.cpu_count()
    if angle.shape[0] < (10 * cpu_count):
        gafs = np.apply_along_axis(_calc_gafs, axis=2, arr=angle, imgFilter=imgFilter)
    else:
        gafs = _apply_along_axis_parallel(_calc_gafs, axis=2, data=angle, num_processes=cpu_count, imgFilter=imgFilter)

    # swap the channel back to the last dimension
    gafs = np.moveaxis(gafs, 1, -1)

    # separate the GASF(0) and GADF(1)
    gaf["gasf"] = gafs[:, :, :, 0, :]
    gaf["gadf"] = gafs[:, :, :, 1, :]

    return gaf


def _convert_to_GAF_GPU(data, imgFilter):
    """Converts the provided data into Gramian Angular Field (GAF) images using the GPU for processing.

    Parameters:
    data : ndarray with ndim=3 and of shape num_samples x time_steps x channels.
        Contains the data to be converted. Each channel is converted independently of the others.

    imgFilter : ImageFilter
        A blur filter to be applied to each converted image.

    ----------
    Returns:
    gaf : dict
        Dictionary containing the following keys:
        'gasf': ndarray of shape num_samples x time_steps x time_steps x channels
                Containing the GASF conversion of the data.
        'gadf': ndarray of shape num_samples x time_steps x time_steps x channels
                Containing the GADF conversion of the data.
        If 'ImgFilter' != None, the filter is applied to each image after conversion.
    """

    gaf = {}
    data = np.swapaxes(data, 1, 2)

    # cut values because precision errors sometimes leads to values sligthly bigger than 1 which causes problems for the arccos
    angle = cp.arccos(cp.minimum(1, cp.asarray(data)))
    
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
                
    gaf["gasf"] = np.moveaxis(np.array(gasf_samples), 1, -1)
    gaf["gadf"] = np.moveaxis(np.array(gadf_samples), 1, -1)

    return gaf


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
image = converter.convert_to_GAF(test)
print("CONVERSION FINISHED")
plotter = GRFPlotter()
plotter.plot_image(image, keys="affected")
blur = ImageFilter("resize", (10,10), output_size=(50,50))
blur_image = converter.convert_to_GAF(test, imgFilter=blur)
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
