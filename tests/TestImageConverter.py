import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import unittest
import warnings
import numpy as np
import pandas as pd
from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from GRFImageConverter import GRFImageConverter
from ImageFilter import ImageFilter

filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"

class TestImageConverter(unittest.TestCase):
    
    
    def test_init(self):
        """Verifies the inital configuration of the GRFImageConverter"""

        converter = GRFImageConverter()
        assert converter.available_conv == ["gaf", "mtf", "rcp"], "List of available conversions is not set right."
        assert not converter.useGpu, "GPU is enabled per default"
        assert converter.parallel_threshold == 100 , "Default parallel threshold is not correct."


    
    def test_settings(self):
        """Verfifies whether or not the enabling/disabling of the GPU and
        the setting of the parallel threshold works correctly.
        Additionally checks if appropriate exeptions are thrown.
        """

        converter = GRFImageConverter()

        # test GPU options
        converter.enableGpu()
        assert converter.useGpu, "GPU was not correctly enabled."
        converter.enableGpu()
        converter.enableGpu()
        assert converter.useGpu, "GPU was not correctly enabled after consecutive calls."
        converter.disableGpu()
        assert not converter.useGpu, "GPU was not correctly disabled."
        converter.disableGpu()
        assert not converter.useGpu, "GPU was not correctly disabled after consecutive calls."

        # test parallel threshold
        converter.set_parallel_threshold(500)
        assert converter.parallel_threshold == 500 , "The parallel threshold was not set correctly."
        converter.set_parallel_threshold(0)
        assert converter.parallel_threshold == 0 , "The parallel threshold was not set correctly."
        converter.set_parallel_threshold(-7)
        assert converter.parallel_threshold == -7 , "The parallel threshold was not set correctly."

        with self.assertRaises(TypeError):
            converter.set_parallel_threshold(None)
        with self.assertRaises(TypeError):
            converter.set_parallel_threshold(8.5)
        with self.assertRaises(TypeError):
            converter.set_parallel_threshold("AB")



    def test_conversions(self):
        self.__assert_gaf_conversion()
        self.__assert_mtf_conversion()
        #self.__assert_filtering()        



    def __assert_gaf_conversion(self):
        """Checks whether or not the output of the converter is identicall to a manual conversion.
        Verifies that the results of the GPU and CPU (serial & parallel) are similar to each other.
        Additionally checks that the appropriate exceptions are thrown in case of invalid Parameters.
        """

        # Manual Conversion
        fetcher = DataFetcher(filepath)
        converter = GRFImageConverter()
        scaler = GRFScaler()
        data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=False, scaler=scaler)
        #The first 10 sample in the TRAIN_BALANCED set correspond to the 10 trials of SESSION_ID=1338 (left side affected)
        test_data = {
            "affected": data["affected"][0:10, :, :],
            "non_affected": data["non_affected"][0:10, :, :],
            "label": data["label"][0:10]
        } 

        filelist = ["GRF_F_V_", "GRF_F_AP_", "GRF_F_ML_", "GRF_COP_AP_", "GRF_COP_ML_"] 
        ver_data = {
            "affected": {},
            "non_affected": {}
        }
        for item in filelist:
            component_name = item[item.index("_")+1:-1].lower()

            # affected
            new_data = pd.read_csv("/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"+item+"PRO_left.csv", header=0)
            # SESSION_ID 1338 is the first item in the TRAIN_BALANCED set (left side affected) with 10 Trials
            ver_data["affected"][component_name]=new_data[new_data["SESSION_ID"]==1338].drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]).values

            # non_affected
            new_data = pd.read_csv("/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"+item+"PRO_right.csv", header=0)
            ver_data["non_affected"][component_name]=new_data[new_data["SESSION_ID"]==1338].drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]).values


        for key in ver_data.keys():
            ver_data[key] = scaler.transform(ver_data[key])
            ver_data[key] = np.swapaxes(np.array(list(ver_data[key].values())), 0, 1)
            
            # compare the original data to the fetched one
            assert np.allclose(test_data[key], np.swapaxes(ver_data[key], 1, 2), rtol=1e-05, atol=1e-07), "Original data does not correspond to the fetched one."
            
        # compare conversion result to CPU (serial) and GPU
        image_cpu = converter.convert(test_data, conversions="gaf")
        converter.enableGpu()
        image_gpu = converter.convert(test_data, conversions="gaf")

        # manual conversion
        for key in ver_data.keys():
            for l in range(ver_data[key].shape[0]):
                for k in range(5):
                    angle = np.arccos(ver_data[key][l, k, :])
                    num = ver_data[key][l, k].shape[0]
                    gasf = np.empty([num, num])
                    gadf = np.empty([num, num])
                    for i in range(num):
                        for j in range(i+1):
                            gasf[i,j] = gasf[j,i] = np.cos(angle[i]+angle[j])
                            gadf[i,j] = np.sin(angle[i]-angle[j])
                            gadf[j,i] = np.sin(angle[j]-angle[i])
                    assert np.allclose(image_cpu[key]["gasf"][l, :, :, k], gasf, rtol=1e-05, atol=1e-06), "Wrong GASF - CPU ({}, {})".format(l, k)
                    assert np.allclose(image_cpu[key]["gadf"][l, :, :, k], gadf, rtol=1e-05, atol=1e-06), "Wrong GADF - CPU ({}, {})".format(l, k)
                    assert np.allclose(image_gpu[key]["gasf"][l, :, :, k], gasf, rtol=1e-05, atol=1e-06), "Wrong GASF - GPU ({}, {})".format(l, k)
                    assert np.allclose(image_gpu[key]["gadf"][l, :, :, k], gadf, rtol=1e-05, atol=1e-06), "Wrong GADF - GPU ({}, {})".format(l, k)

        
        # compare conversion result of GPU and CPU (parallel)
        converter.disableGpu()
        test_data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=True, scaler=GRFScaler())
        image_cpu = converter.convert(test_data, conversions="gaf")
        converter.enableGpu()
        image_gpu = converter.convert(test_data, conversions="gaf")

        assert np.allclose(image_cpu["affected"]["gasf"], image_gpu["affected"]["gasf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU (GASF)."
        assert np.allclose(image_cpu["affected"]["gadf"], image_gpu["affected"]["gadf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU (GADF)."
        assert np.allclose(image_cpu["non_affected"]["gasf"], image_gpu["non_affected"]["gasf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU (GASF)."
        assert np.allclose(image_cpu["non_affected"]["gadf"], image_gpu["non_affected"]["gadf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU (GADF)."
        

        # verify exceptions
        test_data["affected"][1, 4, 3] = -5
        converter.disableGpu()
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="gaf")
        converter.enableGpu()
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="gaf")

        test_data["affected"][1, 4, 3] = 0
        test_data["affected"][8, 9, 1] = 2
        with self.assertWarns(UserWarning):
            converter.convert(test_data, conversions="gaf")
        converter.disableGpu()
        with self.assertWarns(UserWarning):
            converter.convert(test_data, conversions="gaf")





    def __assert_mtf_conversion(self):
        """Checks whether or not the output of the converter is identicall to a manual conversion.
        Verifies that the results of the GPU and CPU (serial & parallel) are similar to each other.
        Additionally checks that the appropriate exceptions are thrown in case of invalid Parameters.
        """

        # Manual Conversion
        fetcher = DataFetcher(filepath)
        converter = GRFImageConverter()
        scaler = GRFScaler()
        data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=False, scaler=scaler)
        #The first 10 sample in the TRAIN_BALANCED set correspond to the 10 trials of SESSION_ID=1338 (left side affected)
        test_data = {
            "affected": data["affected"][0:10, :, :],
            "non_affected": data["non_affected"][0:10, :, :],
            "label": data["label"][0:10]
        } 

        filelist = ["GRF_F_V_", "GRF_F_AP_", "GRF_F_ML_", "GRF_COP_AP_", "GRF_COP_ML_"] 
        ver_data = {
            "affected": {},
            "non_affected": {}
        }
        for item in filelist:
            component_name = item[item.index("_")+1:-1].lower()

            # affected
            new_data = pd.read_csv("/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"+item+"PRO_left.csv", header=0)
            # SESSION_ID 1338 is the first item in the TRAIN_BALANCED set (left side affected) with 10 Trials
            ver_data["affected"][component_name]=new_data[new_data["SESSION_ID"]==1338].drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]).values

            # non_affected
            new_data = pd.read_csv("/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"+item+"PRO_right.csv", header=0)
            ver_data["non_affected"][component_name]=new_data[new_data["SESSION_ID"]==1338].drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]).values


        for key in ver_data.keys():
            ver_data[key] = scaler.transform(ver_data[key])
            ver_data[key] = np.swapaxes(np.array(list(ver_data[key].values())), 0, 1)
            
            # compare the original data to the fetched one
            #assert np.allclose(test_data[key], np.swapaxes(ver_data[key], 1, 2), rtol=1e-05, atol=1e-07), "Original data does not correspond to the fetched one."

        conv_args = {
            "num_bins": 32,
            "range": (-1, 1)
        }

        # compare conversion result to CPU (serial) and GPU
        image_cpu = converter.convert(test_data, conversions="mtf", conv_args=conv_args)
        converter.enableGpu()
        image_gpu = converter.convert(test_data, conversions="mtf", conv_args=conv_args)

        # manual conversion
        for key in ver_data.keys():
            for l in range(ver_data[key].shape[0]):
                for k in range(5):
                    w = np.zeros([conv_args["num_bins"], conv_args["num_bins"]])
                    quantiles = np.digitize(ver_data[key][l, k], np.linspace(conv_args["range"][0], conv_args["range"][1], conv_args["num_bins"], endpoint=False)[1:])
                    for j in range(len(quantiles)-1):
                        w[quantiles[j], quantiles[j+1]] += 1
                    col_sum = np.sum(w, axis=0)
                    w = np.divide(w, np.maximum(1, col_sum))

                    num = ver_data[key][l, k].shape[0]
                    mtf = np.zeros([num, num])
                    for i in range(num):
                        for j in range(num):
                            mtf[i,j] = w[quantiles[i], quantiles[j]]

                    assert np.allclose(image_cpu[key]["mtf"][l, :, :, k], mtf), "Wrong MTF - CPU ({}, {})".format(l, k)
                    assert np.allclose(image_gpu[key]["mtf"][l, :, :, k], mtf), "Wrong MTF - GPU ({}, {})".format(l, k)

        
        # compare conversion result of GPU and CPU (parallel)
        converter.disableGpu()
        test_data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=True, scaler=GRFScaler())
        image_cpu = converter.convert(test_data, conversions="mtf", conv_args=conv_args)
        converter.enableGpu()
        image_gpu = converter.convert(test_data, conversions="mtf", conv_args=conv_args)

        assert np.allclose(image_cpu["affected"]["mtf"], image_gpu["affected"]["mtf"]), "Output differs between CPU and GPU (MTF)."
        assert np.allclose(image_cpu["affected"]["mtf"], image_gpu["affected"]["mtf"]), "Output differs between CPU and GPU (MTF)."
        assert np.allclose(image_cpu["non_affected"]["mtf"], image_gpu["non_affected"]["mtf"]), "Output differs between CPU and GPU (MTF)."
        assert np.allclose(image_cpu["non_affected"]["mtf"], image_gpu["non_affected"]["mtf"]), "Output differs between CPU and GPU (MTF)."


        # verify exeptions
        converter.disableGpu()
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args=92)
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args=["d", "ab"])
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": None})
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args={"num_bins": None, "range": (0,1)})
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args={"num_bins": 4.8, "range": (0,1)})
        converter.enableGpu()
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args=None)
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args=False)
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": [3, 4]})
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": 7})
        with self.assertRaises(TypeError):
            converter.convert(test_data, conversions="mtf", conv_args={"num_bins": "a", "range": (0,1)})

        converter.disableGpu()
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": (5,)})
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": (5, 5, 5)})
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": ()})
        converter.enableGpu()
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="mtf", conv_args={"num_bins": 5})
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": (7, 3)})
        with self.assertRaises(ValueError):
            converter.convert(test_data, conversions="mtf", conv_args={"range": (5, 5)})
        
        with self.assertWarns(UserWarning):
            converter.convert(test_data, conversions="mtf", conv_args={"range": (-1, 1)})
        
        test_data["affected"][0, 0, 0] = -5
        with self.assertWarns(UserWarning):
            converter.convert(test_data, conversions="mtf", conv_args={"num_bins": 5, "range": (-1, 1)})
        test_data["affected"][0, 0, 0] = 0
        test_data["affected"][4, 8, 3] = 2
        with self.assertWarns(UserWarning):
            converter.convert(test_data, conversions="mtf", conv_args={"num_bins": 5, "range": (-1, 1)})
            

        


    def __assert_filtering(self):
        """Checks if the specified filter is applied correctly (for both CPU and GPU).
        Verifies both, the blurring and the resizing of the images.
        """

        #TODO check for all conversions
        # CPU (serial) and GPU
        fetcher = DataFetcher(filepath)
        converter = GRFImageConverter()
        data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=False)
        imgFilter = ImageFilter("avg", (5,5))
        test_data = {
            "affected": data["affected"][0:10, :, :],
            "non_affected": data["non_affected"][0:10, :, :],
            "label": data["label"][0]
        } 
        image_cpu = converter.convert(test_data, conversions="gaf")
        image_cpu_filtered = converter.convert(test_data, conversions="gaf", imgFilter=imgFilter)
        converter.enableGpu()
        image_gpu = converter.convert(test_data, conversions="gaf")
        image_gpu_filtered = converter.convert(test_data, conversions="gaf", imgFilter=imgFilter)

        assert image_cpu["affected"]["gasf"].shape == image_cpu_filtered["non_affected"]["gasf"].shape == image_gpu["affected"]["gadf"].shape == image_gpu_filtered["non_affected"]["gadf"].shape, "Wrong output shape after applying blur-filter."
        assert np.allclose(image_cpu["affected"]["gasf"], image_gpu["affected"]["gasf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU (GASF)."
        assert np.allclose(image_cpu_filtered["affected"]["gadf"], image_gpu_filtered["affected"]["gadf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU after filtering (GADF)."
        assert not np.allclose(image_gpu["non_affected"]["gadf"], image_gpu_filtered["non_affected"]["gadf"], rtol=1e-05, atol=1e-06), "Output does not differ after filtering on GPU (GADF)."
        assert not np.allclose(image_cpu["non_affected"]["gasf"], image_cpu_filtered["non_affected"]["gasf"], rtol=1e-05, atol=1e-06), "Output does not differ after filtering on GPU (GASF)."

        # Verify resize
        imgFilter = ImageFilter("gaussian", (7,7), 1, (50, 50))
        converter.disableGpu()
        image_cpu_filtered = converter.convert(test_data, conversions="gaf", imgFilter=imgFilter)
        converter.enableGpu()
        image_gpu_filtered = converter.convert(test_data, conversions="gaf", imgFilter=imgFilter)

        for side in ["affected", "non_affected"]:
            for gaf in ["gasf", "gadf"]:
                assert image_cpu_filtered[side][gaf].shape == image_gpu_filtered[side][gaf].shape == (10, 50, 50 ,5), "Incorrect output-shape after resizing (Expected (10, 50, 50, 5), Received {}).".format(image_cpu_filtered[side][gaf].shape)

        
        # CPU (parallel) and GPU
        fetcher = DataFetcher(filepath)
        converter.disableGpu()
        data = fetcher.fetch_set(dataset="TRAIN_BALANCED", averageTrials=True)
        imgFilter = ImageFilter("avg", (8,8)) 
        image_cpu = converter.convert(data, conversions="gaf")
        image_cpu_filtered = converter.convert(data, conversions="gaf", imgFilter=imgFilter)
        converter.enableGpu()
        image_gpu = converter.convert(data, conversions="gaf")
        image_gpu_filtered = converter.convert(data, conversions="gaf", imgFilter=imgFilter)

        assert image_cpu["affected"]["gasf"].shape == image_cpu_filtered["non_affected"]["gasf"].shape == image_gpu["affected"]["gadf"].shape == image_gpu_filtered["non_affected"]["gadf"].shape, "Wrong output shape after applying blur-filter."
        assert np.allclose(image_cpu["affected"]["gasf"], image_gpu["affected"]["gasf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU (GASF)."
        assert np.allclose(image_cpu_filtered["affected"]["gadf"], image_gpu_filtered["affected"]["gadf"], rtol=1e-05, atol=1e-06), "Output differs between CPU and GPU after filtering (GADF)."
        assert not np.allclose(image_gpu["non_affected"]["gadf"], image_gpu_filtered["non_affected"]["gadf"], rtol=1e-05, atol=1e-06), "Output does not differ after filtering on GPU (GADF)."
        assert not np.allclose(image_cpu["non_affected"]["gasf"], image_cpu_filtered["non_affected"]["gasf"], rtol=1e-05, atol=1e-06), "Output does not differ after filtering on GPU (GASF)."

        # Verify resize
        imgFilter = ImageFilter("resize", (8,8), 1, (32, 32))
        converter.disableGpu()
        image_cpu_filtered = converter.convert(data, conversions="gaf", imgFilter=imgFilter)
        converter.enableGpu()
        image_gpu_filtered = converter.convert(data, conversions="gaf", imgFilter=imgFilter)

        for side in ["affected", "non_affected"]:
            for gaf in ["gasf", "gadf"]:
                assert image_cpu_filtered[side][gaf].shape == image_gpu_filtered[side][gaf].shape == (730, 32, 32, 5), "Incorrect output-shape after resizing (Expected (730, 32, 32, 5), Received {}).".format(image_cpu_filtered[side][gaf].shape)





if __name__ == "__main__":
    unittest.main()