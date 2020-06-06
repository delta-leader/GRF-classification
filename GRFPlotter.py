import matplotlib.pyplot as plt

class GRFPlotter(object):
    """Plotter to be used for images created from GRF-data.
    The data will be plotted as a heatmap using 'matplotlib' and the specified colormap.
    The plotter is able to process the data as outputted by 'GRFImageConverter' no further modification is necessary.
    
    Currently 3 types of images are supported:
    - Gramian Angular Field (GAF): both Summation (GASF) and Difference (GADF) images are created.
    - Markov Transition Field: Signals are divided into bins and probabilities of bin transitions are calculated (requieres the setting of #bins).
    - Recurrence Plots (RCP): Requires the setting of an embedding dimension and a time delay.

    Per default, all available data is plotted but the user can make the following selections:
    - Select the data (e.g. "affected", "non_affected)
    - Select the images (e.g. "gasf", "rcp")
    - Select the sample (e.g. 1, 5)
    - Select the channels (e.g. "f_v", "cop_ml")

    Attributes:
    valid_keys : list of string
        Contains the valid keys for the data-dictionary. Currently 'affected', 'non_affected', 'affected_val' and 'non_affected_val'.
        Only data stored under those keys can be plotted.
    
    valid_images : list of string
        Contains the keys the images converted from the data. Currently only 'gasf', 'gadf', 'mtf' and 'rcp' are supported.

    comp_order : list of string
        Contains the names of the force components in the same order as they are stored within the data.

    colormap : string
        Contains the colormap to be used for plotting.
    """
    
    def __init__(self):
        self.valid_keys = ["affected", "non_affected", "affected_val", "non_affected_val"]
        self.valid_images = ["gasf", "gadf", "mtf", "rcp"]
        self.comp_order = ["f_v", "f_ap", "f_ml", "cop_ap", "cop_ml"]
        self.colormap = "jet"

    def set_colormap(self, colormap):
        """Sets the colormap used for plotting.

        Parameters:
        colormap : string
            The colormap to be used.

        ----------
        Raises:
        ValueError : If the specified colormap is not available in 'matplotlib.pyplot.colormaps()'.
        """

        if colormap in plt.colormaps():
            self.colormap = colormap
        else :
            raise ValueError("The specified colormap '{}' is not available.".format(colormap))



    def set_comp_order(self, comp_order):
        """Sets the component order within the data.
        This is used to identify the order in which the force components are stored.

        Parameters:
        comp_order : list of string
            The new component order.

        ----------
        Raises:
        TypeError : If the component order is not a list
        ValueError : If one of the 5 force components if missing from the list.
        ValueError : If a value is included that is not a valid force component.
        """

        if not isinstance(comp_order, list):
            raise TypeError("Component order is not a list.")

        if len(set(comp_order)) != 5:
            raise ValueError ("Component order must include exactly 5 different force components, but found {}".format(len(set(comp_order))))
        for component in comp_order:
            self.__check_key(component, self.comp_order)    
        
        self.comp_order = comp_order
        


    def plot_image(self, data, keys=None, images=None, sampleIndex=None, channels=None, comp_order=None ):
        """Plots the GRF-data that has been converted to images using matplotlib.

        Parameters:
        data : dict
            Dictionary containg the GRF-data.
            Only data stored in any of the following keys can be plotted: 'affected', 'non_affected', 'affected_val' and 'non_affected_val'.    
    
        keys : list of string, string or None, default=None
            Specifies which data to plot. Can either be a list containing the following elements: 'affected', 'non_affected', 'affected_val' and 'non_affected_val',
            or a single element passed as a string.
            If None, all of the elements are plotted (if available).

        images : list of string, string or None, default=None
            Specifies which images to plot. Can either be a list containing the following elements : 'gasf', "gadf', 'mtf' and 'rcp',
            or a single element passed as a string.
            If None, all of the images are plotted (if available)

        sampleIndex : list of int, int or None, default=None
            Specifies the index of the sample to plot. Passing a list creates a plot for each corresponding sample of the provided indexes.
            If None, all samples from the data are plotted.

        channels : list of string, string or None, default=None
            Specifies the force components to plot. Can be either a list of the following elements: 'f_v', 'f_ap', 'f_ml', 'cop_ap' and 'cop_ml',
            or a single element passed as a string.
            If None, all the channels are plotted (if available)

        comp_order : list of string or None, default=None["f_v", "f_ap", "f_ml", "cop_ap", "cop_ml"] 
            The order of the components in the data. The first element corresponds to the first channel and so on.
            Can contain any combination of the following values: 'f_v', 'f_ap', 'f_ml', 'cop_ap' and 'cop_ml'.
            If None, the default settings are used for the component order (see 'set_comp_order()').

        ----------
        Raises:
        TypeError : If 'data' is not a dictionary.
        TypeError : If 'keys' is none of the following: None, string or list of string.
        TypeError : If 'images' is none of the following: None, string or list of string.
        TypeError : If 'sampleIndex' is none of the following: None, int or list of int.
        ValueError : If one of the indices specified in sampleIndex is not available.
        TypeError : If 'compOrder' is none of the following: None, list of string.
        TypeError : If 'channels' is none of the following: None, string, list of string.
        """

        if not isinstance(data, dict):
            raise TypeError("Data must be specified as a dictionary.")

        # Check available keys
        available_keys = self.__get_keys(data)
        if keys is None:
            keys = available_keys

        elif isinstance(keys, str):
            self.__check_key(keys, available_keys)
            keys = [keys]

        elif isinstance(keys, list):
            for key in keys:
                self.__check_key(key, available_keys)

        else:
            raise TypeError("An invalid value was provided for 'keys' ({}).".format(keys))


        # Check available images
        available_images = self.__get_images(data, available_keys)
        if images is None:
            images = available_images

        elif isinstance(images, str):
            self.__check_key(images, available_images)
            images = [images]

        elif isinstance(images, list):
            for image in images:
                self.__check_key(image, available_images)

        else:
            raise TypeError("An invalid value was provided for 'images' ({}).".format(images))


        # Check sampleIndex
        num_samples = self.__get_sample_count(data, available_keys, available_images)
        if sampleIndex is None:
            sampleIndex = range(num_samples)

        elif isinstance(sampleIndex, int):
            sampleIndex = [sampleIndex]

        elif isinstance(sampleIndex, list):
            if all(isinstance(index, int) for index in sampleIndex):
                raise TypeError("All elements specified by 'sampleIndex' have to be integers.")
        
        else:
            raise TypeError("An invalid value was provided for 'sampleIndex' ({}).".format(sampleIndex))
    
        if max(sampleIndex) >= num_samples:
            raise ValueError("Index '{}' is not available across all datasets".format(max(sampleIndex)))


        # Check component order
        if comp_order is None:
            comp_order = self.comp_order

        elif isinstance(comp_order, list):
            for component in comp_order:
                self.__check_key(component, self.comp_order)

        else:
            raise TypeError("An invalid values was provided for 'comp_order' ({}).".format(comp_order))


        # Check channels
        if channels is None:
            channels = self.comp_order

        elif isinstance(channels, str):
            self.__check_key(channels, comp_order)
            channels = [channels]

        elif isinstance(channels, list):
            for channel in channels:
                self.__check_key(channels, comp_order)

        else:
            raise TypeError("An invalid values was provided for 'channels' ({}).".format(channels))

        channelIndices = []
        for channel in channels:
            channelIndices.append(comp_order.index(channel))


        for key in keys:
            for image in images:
                for i in sampleIndex:
                    for j in channelIndices:
                        plt.figure("Component: {}".format(comp_order[j]))
                        plt.imshow(data[key][image][i, :, :, j], cmap=self.colormap)
                        plt.title("{} - '{}' (Sample: {})".format(image.upper(), key, i))
                    plt.show()



    def __get_keys(self, data):
        """Returns the relevant keys from the dictionary (i.e. the ones that store GRF data)

        Parameters:
        data : dict
            Dictinonary containing the GRF data.
            Must contain at least one of the keys specified in 'valid_keys'.

        ----------
        Returns:
        key_list : list of string
            Includes all keys that are available in the data and 'valid_keys'

        ----------
        Raises:
        ValueError: If none of the available keys is a valid key.
        """

        key_list = list(data.keys())

        keys = []
        for key in key_list:
            if key in self.valid_keys:
                keys.append(key)
        
        if len(keys) < 1:
            raise ValueError("No valid keys found in the provided data.")

        return key_list



    def __check_key(self, key, valid_keys):
        """Asserts that the value of key is included in the list of valid keys.

        Parameters:
        keys : string
            The key-value to be verified.

        valid_keys : list of string
            Containing the valid keys for the dataset.

        ----------
        Raises:
        ValueError: If keys is not available in valid_keys.
        """

        if key not in valid_keys:
            raise ValueError("'{}' is not available within the set of available keys ({}).".format(key, valid_keys))



    def __get_images(self, data, keys):
        """Returns the available images from each subset of the data specified by a key.

        Parameters:
        data : dict
            Dictinonary containing the GRF data.
            All subsets of the data specified by the available keys must contain at least 1 common image.

        keys : list of string
            List containing the valid keys to access the subsets of the data.

        ----------
        Returns:
        image_list : list of string
            Includes all images that are available within all subsets of the data specified by 'data[keys]'.

        ----------
        Raises:
        ValueError: If no valid images are contained or no common image is contained within all subsets.
        """

        available_images = []
        for key in keys:
            images = []
            for image in data[key].keys():
                if image in self.valid_images:
                    images.append(image)
            available_images.append(images)   

        #remove duplicates
        common_images = set(available_images)

        common_images = list(common_images)        
        if len(common_images) < 1:
            raise ValueError("No valid images were found that are common accross all subsets of the data.")

        return common_images



    def __get_sample_count(self, data, keys, images):
        """Returns the number of samples included across all images and subsets of the data.

        Parameters:
        data : dict
            Dictinonary containing the GRF data.
            
        keys : list of string
            List containing the valid keys to access the subsets of the data.

        images : list of string
            List containing the available images across all subsets of the data.

        ----------
        Returns:
        max_available : int
            The maximum number of samples that is available within all images of all subsets of the data.

        ----------
        Raises:
        ValueError: If the number of available samples is < 1.
        """

        max_available = data[keys[0]][images[0]].shape[0]
        for key in keys:
            for image in images:
                max_available = min(max_available, data[key][image].shape[0])

        if max_available < 1:
            raise ValueError("There are no samples available to plot.")

        return max_available