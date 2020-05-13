import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import unittest
import numpy as np
import cv2 as cv
from ImageFilter import ImageFilter

class TestScaler(unittest.TestCase):

    def test_init(self):
        """Test the inital setup of the filter."""

        imgFilter = ImageFilter()
        assert imgFilter.kernel_size == (5,5), "Incorrect inital kernel-size."
        assert imgFilter.filterType == "avg", "Incorrect initial filter type."
        assert imgFilter.sigma == 0, "Incorrect inital value for sigma."
        assert imgFilter.output_size is None, "Incorrect inital value for output-size."
        imgFilter = ImageFilter("median", output_size=(10,10))
        assert imgFilter.kernel_size == 5, "Incorrect kernel-size for median-filter."
        assert imgFilter.filterType == "median", "Filter type was not set correctly."
        assert imgFilter.output_size == (10,10), "Output-size was not set correctly."
        imgFilter = ImageFilter("median", (10,10))
        assert imgFilter.kernel_size == 10, "Incorrect kernel-size for median-filter."
        imgFilter = ImageFilter("gaussian", (3,3), 5, (50,30))
        assert imgFilter.kernel_size == (3,3), "Kernel-size was not set correcly."
        assert imgFilter.filterType == "gaussian", "Filter type was not set correctly."
        assert imgFilter.sigma == 5, "Sigma was not set correctly."
        assert imgFilter.output_size == (50,30), "Output-size was not set correctly."
        imgFilter = ImageFilter("resize", output_size=(1,1))
        assert imgFilter.filterType == "resize", "Filter type was not set correctly."
        assert imgFilter.output_size == (1,1), "Output-size was not set correctly."
        assert imgFilter.kernel_size == (5,5), "Incorrect inital kernel-size."
        imgFilter = ImageFilter("avg", 7, 2, (2,2))
        assert imgFilter.kernel_size == (7,7), "Kernel-size was not set correcly."
        assert imgFilter.filterType == "avg", "Filter type was not set correctly."
        assert imgFilter.sigma == 2, "Sigma was not set correctly."
        assert imgFilter.output_size == (2,2), "Output-size was not set correctly."


        # verify exception handling
        with self.assertRaises(ValueError):
            ImageFilter("test")
            ImageFilter(filterType=5)
            ImageFilter(filterType=None)
            ImageFilter(kernel_size=(10))
            ImageFilter(kernel_size=(1, 1, 1))
            ImageFilter(kernel_size=(0, 1))
            ImageFilter(kernel_size=0)
            ImageFilter(output_size=(1, -1))
            ImageFilter(output_size=(1, 1, 2))
            ImageFilter(output_size=5)
            ImageFilter(output_size=(7))
            

        with self.assertRaises(TypeError):
            ImageFilter(kernel_size="test")
            ImageFilter(kernel_size=0.5)
            ImageFilter(kernel_size=("a", 2))
            ImageFilter(kernel_size=(1, 0.5))
            ImageFilter(output_size=True)
            ImageFilter(output_size=(3.8, 4))
            ImageFilter(output_size=(10, False))
            ImageFilter(output_size=[10, 10])

    
    def test_apply(self):
        """Checks whether or not the filter is correctly applied."""
        image = np.array([[1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2], [3,3,3,3,3,3,3,3,3,3], [4,4,4,4,4,4,4,4,4,4], [5,5,5,5,5,5,5,5,5,5]])
        # image is reflected at borders
        avg_solution = np.array([[2,2,2,2,2,2,2,2,2,2], [2,2,2,2,2,2,2,2,2,2], [3,3,3,3,3,3,3,3,3,3], [4,4,4,4,4,4,4,4,4,4], [4,4,4,4,4,4,4,4,4,4]])
        imgFilter = ImageFilter("avg", (3,3))
        filtered_image = imgFilter.apply(image)
        assert np.array_equal(avg_solution, filtered_image), "Moving average filter did not produce the expected result ({}).".format(filtered_image)

        image = np.array([[1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2], [3,3,3,3,3,3,3,3,3,3], [4,4,4,4,4,4,4,4,4,4], [5,5,5,5,5,5,5,5,5,5], [6,6,6,6,6,6,6,6,6,6], [7,7,7,7,7,7,7,7,7,7],[8,8,8,8,8,8,8,8,8,8], [9,9,9,9,9,9,9,9,9,9]], dtype=np.float32)
        imgFilter = ImageFilter("gaussian")
        filtered_image = imgFilter.apply(image)
        assert np.allclose(filtered_image, cv.GaussianBlur(image, (5,5), 0)), "Gaussian filter did not produce the expected result."
        
        image = np.array([[1,1,1,3,1,1,1,4,1,1], [2,2,4,2,2,2,2,6,2,2], [3,2,3,3,8,3,3,3,6,3], [1,4,4,4,4,4,4,2,4,4], [5,4,5,5,4,5,5,9,5,5], [6,6,6,6,6,6,6,6,6,6], [7,7,7,7,7,7,7,7,7,7],[8,8,8,8,8,8,8,8,8,8], [9,9,9,9,9,9,9,9,9,9]], dtype=np.float32)
        imgFilter = ImageFilter("median")
        filtered_image = imgFilter.apply(image)
        assert np.allclose(filtered_image, cv.medianBlur(image, 5)), "Median filter did not produce the expected result."

        # Verify resize
        image = np.array([[1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2], [3,3,3,3,3,3,3,3,3,3], [4,4,4,4,4,4,4,4,4,4]], dtype=np.float32)
        resize_solution = np.array([[1.5,1.5,1.5,1.5,1.5], [3.5,3.5,3.5,3.5,3.5]])
        imgFilter = ImageFilter("resize", output_size=(5, 2))
        filtered_image = imgFilter.apply(image)
        assert np.array_equal(resize_solution, filtered_image), "Resizing filter did not produce the expected result ({}).".format(filtered_image)
        
        imgFilter = ImageFilter("avg", kernel_size=(2,2), output_size=(5, 2))
        filtered_image = imgFilter.apply(image)
        resize_solution = np.array([[1.5,1.5,1.5,1.5,1.5], [3,3,3,3,3]])
        assert np.array_equal(filtered_image, resize_solution), "Filtering + resizing did not produce the expected result ({}).".format(filtered_image)

        image = np.array([[1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2], [3,3,3,3,3,3,3,3,3,3], [4,4,4,4,4,4,4,4,4,4], [5,5,5,5,5,5,5,5,5,5], [6,6,6,6,6,6,6,6,6,6], [7,7,7,7,7,7,7,7,7,7],[8,8,8,8,8,8,8,8,8,8], [9,9,9,9,9,9,9,9,9,9]], dtype=np.float32)
        imgFilter = ImageFilter("gaussian", kernel_size=(3,3), output_size=(3, 5))
        filtered_image = imgFilter.apply(image)
        assert np.allclose(filtered_image, cv.resize(cv.GaussianBlur(image, (3, 3), 0), (3, 5), interpolation = cv.INTER_AREA)), "Filtering + resizing did not produce the expected result."

        # Verify exceptions
        imgFilter = ImageFilter("resize")
        with self.assertRaises(ValueError):
            imgFilter.apply(image)





if __name__ == "__main__":
    unittest.main()