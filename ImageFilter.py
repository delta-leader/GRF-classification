import cv2 as cv

class ImageFilter(object):
    """Wrapper to apply one of the supported OpenCV filters to the images created by 'GRFIMAGE' converter.
    Can also be used to resize the image into a specified output format using an averaging approach.

    Currently the following filters are available:
    'cv.blur': Applies a normalized box filter (moving average).
    'cv.GaussianBlur': Instead of a box filter, a gaussian kernel is used.
    'cv.medianBlur': Takes median of all the pixels under the kernel area and replaces the central element with this median value.
    'cv.resize': Resizes the image using an averaging approach.

    Parameters:
    filterType : string, default="avg"
        Specifies the type of the filter. The following options are available:
        "avg" -> cv.blur()
        "gaussian" -> cv.GaussianBlur()
        "median" -> cv.medianBlur()
        "resize" -> cv.resize()

    kernel_size : int or int Tupel of shape(2):
        The size of the kernel. If only one number is provided, a quadratic kernel is applied.

    sigma : double
        Gaussian kernel standard deviation (applied in both x and y direction).
        Ignored if filter != "gaussian".

    output_size : int tupel of shape(2)
        The desired size of the output.
        Using this option resizes the image using an averaging approach.
        If this is set and 'filterType' != "resize", the blurring filter is applied before resizing the image.

    ----------
    Attributes:
    availableFilterst : list of shape (3)
        Exhaustive list of the supported filters ("avg", "gaussian", "median", "resize").
    """

    def __init__(self, filterType="avg", kernel_size=5, sigma=0, output_size=None):
        self.availableFilters = ["avg", "gaussian", "median", "resize"] 
        self.__set_filterType(filterType)
        self.__set_kernel_size(kernel_size)
        self.sigma = sigma
        self.__set_output_size(output_size)
        

    def apply(self, img):
        """Applies the filter to the provided image and returns the result.

        Parameters:
        img : ndarray with ndim=2

        ----------
        Returns:
        filtered_img : ndarray, same shape as the input
            Contains the filtered image.

        ----------
        Raises:
        ValueError: If the input is not 2 dimensional.
        """

        if img.ndim != 2:
            raise ValueError("Expected a 2-dimensional image but found an image with {} dimensions.".format(img.ndim))

        if self.filterType == "avg":
            filtered_img = cv.blur(img, self.kernel_size)
        elif self.filterType == "gaussian":
            filtered_img = cv.GaussianBlur(img, self.kernel_size, self.sigma)
        elif self.filterType == "median":
            filtered_img = cv.medianBlur(img, self.kernel_size)
        elif self.filterType == "resize":
            if self.output_size is None:
                raise ValueError("Filter is set to 'resize', but 'output_shape' was not specified.")
            filtered_img = img
        else:
            raise ValueError("Unexpected value for filterType ({}). This should not have happened.".format(self.filterType))

        if self.output_size is not None:
            filtered_img = cv.resize(filtered_img, self.output_size, interpolation = cv.INTER_AREA)

        return filtered_img


    def __set_filterType(self, filterType):
        """Verifies that the specified filter type is supported and sets the value accordingly.

        Parameters:
        filterType : string
            Specifies the type of the filter. Must be one of 'availableFilters'

        ----------
        Raises:
        ValueError : If the specified filterType is not defined in 'availableFilters'.
        """

        filterType = filterType.lower()
        if filterType not in self.availableFilters:
            raise ValueError("Filter '{}' is not available.".format(filterType))

        self.filterType = filterType

    
    def __set_kernel_size(self, kernel_size):
        """Verifies that the specified kernel dimensions are valid (e.g. non negativ) and sets the value accordingly.

        Parameters:
        kernel_size : int or int Tupel of shape (2)
            The size of the kernel. If only one number is provided, a quadratic kernel is applied.

        ----------
        Raises:
        ValueError : If a tupel is specified that does not have shape (2) or any of the values are 0 or negative.
        TypeError : If the provided values are not Integers.
        """

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        _check_size(kernel_size)

        if self.filterType == "median":
            self.kernel_size = kernel_size[0]
        else:
            self.kernel_size = kernel_size


    def __set_output_size(self, output_size):
        """Verifies that the specified output dimensions are valid (e.g. non negativ) and sets the value accordingly.

        Parameters:
        output_size : int tupel of shape (2):
            The desired size of the output. An averaging approach is used for resizing.

        ----------
        Raises:
        ValueError : If the parameter is not an integer tupel of shape (2).
        ValueError : If any of the values in 'output_shape' are 0 or negative.
        """

        if output_size is None:
            self.output_size = None
            return

        _check_size(output_size)
        self.output_size = output_size


def _check_size(size_tuple):
    """Verifies that the specified tupel represents a valid size for a 2-dimensional array.
    I.e. it has 2 dimensions, and only integer values greater than 0.

    Parameters:
    size_tupel: int tuple of shape(2)

    ----------
    ValueError : If the tuple is not 2 dimensional or any of the values is 0 or negative.
    TypeError : If the provided value is not a tuple or its individual elements are not Integers.
    """

    if not isinstance (size_tuple, tuple):
        raise TypeError("Invalid type expected a tuple for one of the sizes.")
        
    if len(size_tuple) != 2:
        raise ValueError("Expected tuple of shape (2) but found tuple of shape ({}).".format(len(size_tuple)))
    for item in size_tuple:
        if not isinstance(item, int):
            raise TypeError("Provided values for one of the sizes are not of type 'int'")
        if item < 1 :
            raise ValueError("Sizes have to be a positive integer > 0.")