import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import random
import unittest
import numpy as np
import pandas as pd
from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from sklearn.preprocessing import StandardScaler
from DataFetcher import _select_initial_measurements
from DataFetcher import _drop_orthopedics
from DataFetcher import _trim_metadata
from DataFetcher import _select_dataset
from DataFetcher import _sample
from DataFetcher import _average_trials
from DataFetcher import _scale
from DataFetcher import _fit_scaler

filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC"

class TestFetcher(unittest.TestCase):
    
    def test_init(self):
        DataFetcher(filepath)
        DataFetcher(filepath + "/")
        with self.assertRaises(IOError):
            DataFetcher(filepath + "/test")
            DataFetcher("/test")
            DataFetcher("")

    def test_fetch(self):
        fetcher = DataFetcher(filepath)

        #self.__assert_data_arrangment(fetcher)
        #self.__assert_class_labels(fetcher)
        self.__assert_data_selection(fetcher)
        #self.__assert_sampling(fetcher)
        #self.__assert_averaging(fetcher)
        #self.__assert_scaling(fetcher)
        #self.__assert_concat(fetcher)
        #self.__assert_rand(fetcher)
        #self.__assert_comp_order(fetcher)


        
    def __assert_class_labels(self, fetcher):
        """Checks whether or not the class labels are identical to the original ones."""

        label_dict = {"HC":0, "H":1, "K":2, "A":3, "C":4}


        # Verify non-averaged labels
        train, test = fetcher.fetch_data(onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=False)
        metadata = pd.read_csv(filepath+"/GRF_metadata.csv", header=0)
        data = pd.read_csv(filepath+"/GRF_COP_AP_PRO_left.csv", header=0)
        metadata = metadata[metadata["SPEED"]==2]
        train_orig = metadata[metadata["TRAIN"]==1][["SESSION_ID", "CLASS_LABEL", "AFFECTED_SIDE"]].set_index("SESSION_ID")
        test_orig = metadata[metadata["TEST"]==1][["SESSION_ID", "CLASS_LABEL", "AFFECTED_SIDE"]].set_index("SESSION_ID")

        def is_leftSide_affected(x):
            if x == 0:
                return True
            if x == 1:
                return False
            return not random.getrandbits(1)

        random.seed(42)
        train_orig["LEFT_AFFECTED"] = train_orig["AFFECTED_SIDE"].apply(is_leftSide_affected)
        random.seed(42)
        test_orig["LEFT_AFFECTED"] = test_orig["AFFECTED_SIDE"].apply(is_leftSide_affected)

        train_labels = data.join(train_orig, on="SESSION_ID", how="inner").reset_index(drop=True)
        train_labels_ordered = train_labels[train_labels["LEFT_AFFECTED"]==True]["CLASS_LABEL"].map(label_dict).values
        train_labels_ordered = np.append(train_labels_ordered, train_labels[train_labels["LEFT_AFFECTED"]==False]["CLASS_LABEL"].map(label_dict).values, axis=0)    
        assert np.array_equal(train["label"], train_labels_ordered), "Class labels do not match for TRAIN-set (non-averaged)."
        
        test_labels = data.join(test_orig, on="SESSION_ID", how="inner").reset_index(drop=True)
        test_labels_ordered = test_labels[test_labels["LEFT_AFFECTED"]==True]["CLASS_LABEL"].map(label_dict).values
        test_labels_ordered = np.append(test_labels_ordered, test_labels[test_labels["LEFT_AFFECTED"]==False]["CLASS_LABEL"].map(label_dict).values, axis=0)
        assert np.array_equal(test["label"], test_labels_ordered), "Class labels do not match for TEST-set (non-averaged)."


        # Verify averaged labels
        train, test = fetcher.fetch_data(raw=True, onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=True)
        data = data.groupby("SESSION_ID").first()

        train_labels = data.join(train_orig, on="SESSION_ID", how="inner").reset_index(drop=True)
        train_labels_ordered = train_labels[train_labels["LEFT_AFFECTED"]==True]["CLASS_LABEL"].map(label_dict).values
        train_labels_ordered = np.append(train_labels_ordered, train_labels[train_labels["LEFT_AFFECTED"]==False]["CLASS_LABEL"].map(label_dict).values, axis=0)    
        assert np.array_equal(train["label"], train_labels_ordered), "Class labels do not match for TRAIN-set (averaged)."

        test_labels = data.join(test_orig, on="SESSION_ID", how="inner").reset_index(drop=True)
        test_labels_ordered = test_labels[test_labels["LEFT_AFFECTED"]==True]["CLASS_LABEL"].map(label_dict).values
        test_labels_ordered = np.append(test_labels_ordered, test_labels[test_labels["LEFT_AFFECTED"]==False]["CLASS_LABEL"].map(label_dict).values, axis=0)
        assert np.array_equal(test["label"], test_labels_ordered), "Class labels do not match for TEST-set (averaged)."



    def __assert_data_selection(self, fetcher):
        """Check whether the right amound of data is selected (i.e. all corresponding samples).
        Additionally check for correct exceptions if invalid parameters are passed for data-selection.
        """
        """
        # Verify for averaged data
        train, test = fetcher.fetch_data(onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN")
        assert train["affected"].shape[0] == 5817, "Wrong number of measurements in TRAIN. {} vs 5817.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (5817, 101, 5), "Wrong shape of output data in TRAIN. {} vs (5817, 101, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 5817, "Wrong number of labels in TRAIN. {} vs 5817.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 2617, "Wrong number of measurements in TEST. {} vs 2617.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (2617, 101, 5), "Wrong shape of output data in TEST. {} vs (2617, 101, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 2617, "Wrong number of labels in TEST. {} vs 2617.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(raw= True, onlyInitial=True, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN")
        assert train["non_affected"].shape[0] == 1463, "Wrong number of measurements in TRAIN. {} vs 1463.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (1463, 100, 5), "Wrong shape of output data in TRAIN. {} vs (1463, 100, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 1463, "Wrong number of labels in TRAIN. {} vs 1463.".format(train["label"].shape[0])
        assert test["non_affected"].shape[0] == 704, "Wrong number of measurements in TEST. {} vs 704.".format(test["non_affected"].shape[0])
        assert test["affected"].shape == (704, 100, 5), "Wrong shape of output data in TEST. {} vs (704, 100, 5).".format(test["affected"].shape)
        assert test["label"].shape[0] == 704, "Wrong number of labels in TEST. {} vs 704.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(onlyInitial=False, dropOrthopedics="Verified", dropBothSidesAffected=False, dataset="TRAIN")
        assert train["affected"].shape[0] == 4820, "Wrong number of measurements in TRAIN. {} vs 4820.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (4820, 101, 5), "Wrong shape of output data in TRAIN. {} vs (4820, 101, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 4820, "Wrong number of labels in TRAIN. {} vs 4820.".format(train["label"].shape[0])
        assert test["non_affected"].shape[0] == 2270, "Wrong number of measurements in TEST. {} vs 2270.".format(test["non_affected"].shape[0])
        assert test["affected"].shape == (2270, 101, 5), "Wrong shape of output data in TEST. {} vs (2270, 101, 5).".format(test["affected"].shape)
        assert test["label"].shape[0] == 2270, "Wrong number of labels in TEST. {} vs 2270.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(raw=True, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN")
        assert train["non_affected"].shape[0] == 1234, "Wrong number of measurements in TRAIN. {} vs 1234.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (1234, 100, 5), "Wrong shape of output data in TRAIN. {} vs (1234, 100, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 1234, "Wrong number of labels in TRAIN. {} vs 1234.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 688, "Wrong number of measurements in TEST. {} vs 688.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (688, 100, 5), "Wrong shape of output data in TEST. {} vs (688, 100, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 688, "Wrong number of labels in TEST. {} vs 688.".format(test["label"].shape[0])

        train = fetcher.fetch_set(onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED")
        assert train["non_affected"].shape[0] == 730, "Wrong number of measurements in TRAIN_BALANCED. {} vs 730.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (730, 101, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (730, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 730, "Wrong number of labels in TRAIN_BALANCED. {} vs 730.".format(train["label"].shape[0])
        train = fetcher.fetch_set(raw=True, onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN_BALANCED")
        assert train["non_affected"].shape[0] == 730, "Wrong number of measurements in TRAIN_BALANCED. {} vs 730.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (730, 100, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (730, 100, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 730, "Wrong number of labels in TRAIN_BALANCED. {} vs 730.".format(train["label"].shape[0])

        train, test = fetcher.fetch_data(raw= True, onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=True, dataset="TRAIN", averageTrials=True)
        assert train["affected"].shape[0] == 5539, "Wrong number of measurements in TRAIN. {} vs 5539.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (5539, 100, 5), "Wrong shape of output data in TRAIN. {} vs (5539, 100, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 5539, "Wrong number of labels in TRAIN. {} vs 5539.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 2478, "Wrong number of measurements in TEST. {} vs 2478.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (2478, 100, 5), "Wrong shape of output data in TEST. {} vs (2478, 100, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 2478, "Wrong number of labels in TEST. {} vs 2478.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(raw= False, onlyInitial=False, dropOrthopedics="All", dropBothSidesAffected=True, dataset="TRAIN", averageTrials=True)
        assert train["affected"].shape[0] == 4151, "Wrong number of measurements in TRAIN. {} vs 4151.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (4151, 101, 5), "Wrong shape of output data in TRAIN. {} vs (4151, 101, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 4151, "Wrong number of labels in TRAIN. {} vs 4151.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 2079, "Wrong number of measurements in TEST. {} vs 2079.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (2079, 101, 5), "Wrong shape of output data in TEST. {} vs (2079, 101, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 2079, "Wrong number of labels in TEST. {} vs 2079.".format(test["label"].shape[0])

        train = fetcher.fetch_set(onlyInitial=True, dropOrthopedics="Verified", dropBothSidesAffected=True, dataset="TRAIN_BALANCED")
        assert train["non_affected"].shape[0] == 707, "Wrong number of measurements in TRAIN_BALANCED. {} vs 707.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (707, 101, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (707, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 707, "Wrong number of labels in TRAIN_BALANCED. {} vs 707.".format(train["label"].shape[0])


        # Verify exceptions
        with self.assertRaises(ValueError):
            fetcher.fetch_set(onlyInitial=True, dropOrthopedics="", dataset="TRAIN_BALANCED")
            fetcher.fetch_set(onlyInitial=True, dropOrthopedics=None, dataset="TRAIN_BALANCED")
            fetcher.fetch_set(onlyInitial=True, dropOrthopedics="TEST", dataset="TRAIN_BALANCED")
            fetcher.fetch_set(onlyInitial=True, dropOrthopedics="None", dataset="TEST_BALANCED")
            fetcher.fetch_set(onlyInitial=True, dropOrthopedics="None", dataset="")
            fetcher.fetch_set(onlyInitial=True, dropOrthopedics="None", dataset=None)
            fetcher.fetch_data(onlyInitial=True, dropOrthopedics="All", dataset="TEST")

        
        # Verify for non-averaged data
        train, test = fetcher.fetch_data(raw= True, onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN", averageTrials=False)
        assert train["affected"].shape[0] == 48681, "Wrong number of measurements in TRAIN. {} vs 48681.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (48681, 100, 5), "Wrong shape of output data in TRAIN. {} vs (48681, 100, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 48681, "Wrong number of labels in TRAIN. {} vs 48681.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 21776, "Wrong number of measurements in TEST. {} vs 21776.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (21776, 100, 5), "Wrong shape of output data in TEST. {} vs (21776, 100, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 21776, "Wrong number of labels in TEST. {} vs 21776.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(onlyInitial=True, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN", averageTrials=False)
        assert train["non_affected"].shape[0] == 12206, "Wrong number of measurements in TRAIN. {} vs 12206.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (12206, 101, 5), "Wrong shape of output data in TRAIN. {} vs (12206, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 12206, "Wrong number of labels in TRAIN. {} vs 12206.".format(train["label"].shape[0])
        assert test["non_affected"].shape[0] == 5542, "Wrong number of measurements in TEST. {} vs 5542.".format(test["non_affected"].shape[0])
        assert test["affected"].shape == (5542, 101, 5), "Wrong shape of output data in TEST. {} vs (5542, 101, 5).".format(test["affected"].shape)
        assert test["label"].shape[0] == 5542, "Wrong number of labels in TEST. {} vs 704.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(raw=True, onlyInitial=False, dropOrthopedics="Verified", dropBothSidesAffected=False, dataset="TRAIN", averageTrials=False)
        assert train["affected"].shape[0] == 40340, "Wrong number of measurements in TRAIN. {} vs 40340.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (40340, 100, 5), "Wrong shape of output data in TRAIN. {} vs (40340, 100, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 40340, "Wrong number of labels in TRAIN. {} vs 40340.".format(train["label"].shape[0])
        assert test["non_affected"].shape[0] == 18880, "Wrong number of measurements in TEST. {} vs 18880.".format(test["non_affected"].shape[0])
        assert test["affected"].shape == (18880, 100, 5), "Wrong shape of output data in TEST. {} vs (18880, 100, 5).".format(test["affected"].shape)
        assert test["label"].shape[0] == 18880, "Wrong number of labels in TEST. {} vs 18880.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(onlyInitial=True, dropOrthopedics="All", dataset="TRAIN", dropBothSidesAffected=False, averageTrials=False)
        assert train["non_affected"].shape[0] == 10386, "Wrong number of measurements in TRAIN. {} vs 10386.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (10386, 101, 5), "Wrong shape of output data in TRAIN. {} vs (10386, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 10386, "Wrong number of labels in TRAIN. {} vs 10386.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 5408, "Wrong number of measurements in TEST. {} vs 5408.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (5408, 101, 5), "Wrong shape of output data in TEST. {} vs (5408, 101, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 5408, "Wrong number of labels in TEST. {} vs 5408.".format(test["label"].shape[0])

        train = fetcher.fetch_set(raw=True, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", averageTrials=False)
        assert train["non_affected"].shape[0] == 6400, "Wrong number of measurements in TRAIN_BALANCED. {} vs 6400.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (6400, 100, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (6400, 100, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 6400, "Wrong number of labels in TRAIN_BALANCED. {} vs 6400.".format(train["label"].shape[0])
        train = fetcher.fetch_set(onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", averageTrials=False)
        assert train["non_affected"].shape[0] == 6400, "Wrong number of measurements in TRAIN_BALANCED. {} vs 6400.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (6400, 101, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (6400, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 6400, "Wrong number of labels in TRAIN_BALANCED. {} vs 6400.".format(train["label"].shape[0])
        """               
        train, test = fetcher.fetch_data(raw= False, onlyInitial=True, dropOrthopedics="None", dropBothSidesAffected=True, dataset="TRAIN", averageTrials=False)
        assert train["affected"].shape[0] == 11665, "Wrong number of measurements in TRAIN. {} vs 11665.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (11665, 101, 5), "Wrong shape of output data in TRAIN. {} vs (11665, 101, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 11665, "Wrong number of labels in TRAIN. {} vs 11665.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 5264, "Wrong number of measurements in TEST. {} vs 5264.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (5264, 101, 5), "Wrong shape of output data in TEST. {} vs (5264, 101, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 5264, "Wrong number of labels in TEST. {} vs 5264.".format(test["label"].shape[0])

        train, test = fetcher.fetch_data(raw= True, onlyInitial=True, dropOrthopedics="Verified", dropBothSidesAffected=True, dataset="TRAIN", averageTrials=False)
        assert train["affected"].shape[0] == 10919, "Wrong number of measurements in TRAIN. {} vs 10919.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (10919, 100, 5), "Wrong shape of output data in TRAIN. {} vs (10919, 100, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 10919, "Wrong number of labels in TRAIN. {} vs 10919.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 5143, "Wrong number of measurements in TEST. {} vs 5143.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (5143, 100, 5), "Wrong shape of output data in TEST. {} vs (5143, 100, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 5143, "Wrong number of labels in TEST. {} vs 5143.".format(test["label"].shape[0])

        train = fetcher.fetch_set(onlyInitial=False, dropOrthopedics="All", dropBothSidesAffected=True, dataset="TRAIN_BALANCED", averageTrials=False)
        assert train["non_affected"].shape[0] == 6205, "Wrong number of measurements in TRAIN_BALANCED. {} vs 6205.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (6205, 101, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (6205, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 6205, "Wrong number of labels in TRAIN_BALANCED. {} vs 6205.".format(train["label"].shape[0])
        


    def __assert_data_arrangment(self, fetcher):
        """Checks whether to correct data is read from the files.
        Additionally verfies the data arrangement and assignment (affected/non-affected) according to some samples.
        """
    
        # default component order
        filelist = ["GRF_F_V_", "GRF_F_AP_", "GRF_F_ML_", "GRF_COP_AP_", "GRF_COP_ML_"] 

        # Verification for processed data
        train = fetcher.fetch_set(onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=False)

        # SESSION_IDs where the left side is affected (413, 1166, 19749 and 20264 only with the default seed)
        leftSideAffected = [31057, 13377, 26461, 413, 841, 1166, 19749, 20264]
        # SESSION_IDs where the right side is affected (1090, 17666, 30882 and 30036 only with the default seed)
        rightSideAffected = [30633, 27998, 15242, 557, 635, 1090, 17666, 30882, 30036]

        verifyRight = pd.read_csv(filepath+"/"+filelist[0]+"PRO_right.csv", header=0)
        verifyLeft = pd.read_csv(filepath+"/"+filelist[0]+"PRO_left.csv", header=0)
        verifyAffectedL = np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyUnaffectedL = np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyAffectedR = np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyUnaffectedR = np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2)

        for datafile in filelist[1:]:
            verifyRight = pd.read_csv(filepath+"/"+datafile+"PRO_right.csv", header=0)
            verifyLeft = pd.read_csv(filepath+"/"+datafile+"PRO_left.csv", header=0)

            verifyAffectedL = np.append(verifyAffectedL, np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyUnaffectedL = np.append(verifyUnaffectedL, np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyAffectedR = np.append(verifyAffectedR, np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyUnaffectedR = np.append(verifyUnaffectedR, np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)

        for affectedSide, unaffectedSide in [(verifyAffectedL, verifyUnaffectedL), (verifyAffectedR, verifyUnaffectedR)]:
            for row in affectedSide:
                assert np.any((train["affected"][:]==row).all(2).all(1)), "Sample: {}\n is missing in the final data for the affected side.".format(row)
                assert not np.any((train["non_affected"][:]==row).all(2).all(1)), "Sample (affected): {}\n is contained in the final data for the unaffected side.".format(row)
            for row in unaffectedSide:
                assert np.any((train["non_affected"][:]==row).all(2).all(1)), "Sample: {}\n is missing in the final data for the unaffected side.".format(row)
                assert not np.any((train["affected"][:]==row).all(2).all(1)), "Sample (unaffected): {}\n is contained in the final data for the affected side.".format(row)
        

        # Verification for raw data
        train = fetcher.fetch_set(raw=True, onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=False)

        # SESSION_IDs where the left side is affected (23321, 42415, 19628 and 29751 only with the default seed)
        leftSideAffected = [18537, 22706, 41633, 12111, 23321, 42415, 19628, 29751]
        # SESSION_IDs where the right side is affected (7601, 33320, 999910373 and 20033 only with the default seed)
        rightSideAffected = [22765, 20307, 15298, 26125, 7601, 33320, 999910373, 20033]

        verifyRight = pd.read_csv(filepath+"/"+filelist[0]+"RAW_right.csv", header=0)
        verifyLeft = pd.read_csv(filepath+"/"+filelist[0]+"RAW_left.csv", header=0)
        verifyAffectedL = verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:]
        verifyUnaffectedL = verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:]
        verifyAffectedR = verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:]
        verifyUnaffectedR = verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:]

        verifyAffectedL_new = []
        verifyAffectedR_new = []
        verifyUnaffectedL_new = []
        verifyUnaffectedR_new = []

        for dataset, new in [(verifyAffectedL, verifyAffectedL_new), (verifyAffectedR, verifyAffectedR_new), (verifyUnaffectedL, verifyUnaffectedL_new), (verifyUnaffectedR, verifyUnaffectedR_new)]:
            for sample in dataset:
                row = sample[~np.isnan(sample)]
                sampling = range(0, row.shape[0])
                sampling_goal = np.arange(0, row.shape[0], row.shape[0]/100)[:100]
                new.append(np.expand_dims(np.interp(sampling_goal, sampling, row), axis=1))
        
        verify_sampled = [np.array(verifyAffectedL_new), np.array(verifyAffectedR_new), np.array(verifyUnaffectedL_new), np.array(verifyUnaffectedR_new)]
       
        for datafile in filelist[1:]:
            verifyRight = pd.read_csv(filepath+"/"+datafile+"RAW_right.csv", header=0)
            verifyLeft = pd.read_csv(filepath+"/"+datafile+"RAW_left.csv", header=0)

            verifyAffectedL = verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:]
            verifyUnaffectedL = verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:]
            verifyAffectedR = verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:]
            verifyUnaffectedR = verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:]

            i = 0
            for dataset in [verifyAffectedL, verifyAffectedR, verifyUnaffectedL, verifyUnaffectedR]:
                temp = []
                for sample in dataset:
                    row = sample[~np.isnan(sample)]
                    sampling = range(0, row.shape[0])
                    sampling_goal = np.arange(0, row.shape[0], row.shape[0]/100)[:100]
                    temp.append(np.expand_dims(np.interp(sampling_goal, sampling, row), axis=1))
                verify_sampled[i] = np.append(verify_sampled[i], np.array(temp), axis=2)
                i += 1

        for affectedSide, unaffectedSide in [(verify_sampled[0], verify_sampled[2]), (verify_sampled[1], verify_sampled[3])]:
            for row in affectedSide:
                assert np.any(np.isclose(train["affected"][:], row).all(2).all(1)), "Sample: {}\n is missing in the final data for the affected side.".format(row)
                assert not np.any(np.isclose(train["non_affected"][:], row).all(2).all(1)), "Sample (affected): {}\n is contained in the final data for the unaffected side.".format(row)
            for row in unaffectedSide:
                assert np.any(np.isclose(train["non_affected"][:], row).all(2).all(1)), "Sample: {}\n is missing in the final data for the unaffected side.".format(row)
                assert not np.any(np.isclose(train["affected"][:], row).all(2).all(1)), "Sample (unaffected): {}\n is contained in the final data for the affected side.".format(row)


    
    def __assert_sampling(self, fetcher):
        """Checks whether the resampling functionality works as expected.
        Additionally checks for the correct exceptions if invalid values are passes as 'stepsize'.
        """

        # Verify for processed data
        train, test = fetcher.fetch_data(raw=False, stepsize=1)
        assert train["affected"].shape == (730, 101, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (730, 101, 5).".format(train["affected"].shape)

        sampled2 = fetcher.fetch_set(raw=False, stepsize=2)
        sampled3 = fetcher.fetch_set(raw=False, stepsize=3)
        sampled10 = fetcher.fetch_set(raw=False, stepsize=10)
        sampled100 = fetcher.fetch_set(raw=False, stepsize=100)
        for leg in ["affected", "non_affected"]:
            assert sampled2[leg].shape == (730, 51, 5), "Wrong output shape after sampling with stepsize=2. {} vs (730, 51, 5).".format(sampled2[leg].shape)
            assert sampled3[leg].shape == (730, 34, 5), "Wrong output shape after sampling with stepsize=3. {} vs (730, 34, 5).".format(sampled3[leg].shape)
            assert sampled10[leg].shape == (730, 11, 5), "Wrong output shape after sampling with stepsize=10. {} vs (730, 11, 5).".format(sampled10[leg].shape)
            assert sampled100[leg].shape == (730, 2, 5), "Wrong output shape after sampling with stepsize=100. {} vs (730, 2, 5).".format(sampled100[leg].shape)

            for i in range(51):
                assert np.array_equal(sampled2[leg][:, i, :], train[leg][:, i*2, :]), "Sampled data does not match original data for stepsize=2."
            for i in range(34):
                assert np.array_equal(sampled3[leg][:, i, :], train[leg][:, i*3, :]), "Sampled data does not match original data for stepsize=3."
            for i in range(11):
                assert np.array_equal(sampled10[leg][:, i, :], train[leg][:, i*10, :]), "Sampled data does not match original data for stepsize=10."
            for i in range(2):
                assert np.array_equal(sampled100[leg][:, i, :], train[leg][:, i*100, :]), "Sampled data does not match original data for stepsize=100."


        # Verify for raw data
        train, test = fetcher.fetch_data(raw=True, stepsize=1)
        assert train["affected"].shape == (730, 100, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (730, 100, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 730, "Wrong number of labels in TRAIN_BALANCED. {} vs 730.".format(train["label"].shape[0])  
        assert test["affected"].shape == (2617, 100, 5), "Wrong shape of output data in TEST. {} vs (2617, 100, 5).".format(test["affected"].shape)
        assert test["label"].shape[0] == 2617, "Wrong number of labels in TEST. {} vs 2617.".format(test["label"].shape[0])  

        sampled2 = fetcher.fetch_set(raw=True, stepsize=2)
        sampled3 = fetcher.fetch_set(raw=True, stepsize=3)
        sampled10 = fetcher.fetch_set(raw=True, stepsize=10)  
        sampled100 = fetcher.fetch_set(raw=True, stepsize=100)  
        sampled333 = fetcher.fetch_set(raw=True, stepsize=33.3)
        for leg in ["affected", "non_affected"]:
            assert sampled2[leg].shape == (730, 50, 5), "Wrong output shape after sampling with stepsize=2. {} vs (730, 50, 5).".format(sampled2[leg].shape)
            assert sampled3[leg].shape == (730, 33, 5), "Wrong output shape after sampling with stepsize=3. {} vs (730, 33, 5).".format(sampled3[leg].shape)
            assert sampled10[leg].shape == (730, 10, 5), "Wrong output shape after sampling with stepsize=10. {} vs (730, 10, 5).".format(sampled10[leg].shape)
            assert sampled100[leg].shape == (730, 1, 5), "Wrong output shape after sampling with stepsize=100. {} vs (730, 1, 5).".format(sampled100[leg].shape)
            assert sampled333[leg].shape == (730, 3, 5), "Wrong output shape after sampling with stepsize=33.3. {} vs (730, 3, 5).".format(sampled333[leg].shape)


        # Verify exceptions
        with self.assertRaises(ValueError):
            fetcher.fetch_set(raw=True, stepsize=0)
            fetcher.fetch_set(raw=True, stepsize=222)
            fetcher.fetch_set(raw=False, stepsize=-3)
            fetcher.fetch_set(raw=False, stepsize=101)

        with self.assertRaises(TypeError):
            fetcher.fetch_set(raw=False, stepsize=2.5)

    

    def __assert_averaging(self, fetcher):
        """Checks whether or not the data is averaged correctly."""

        # Verify for processed data
        train = fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False)
        assert train["non_affected"].shape[0] == 6400, "Wrong number of measurements in TRAIN_BALANCED. {} vs 6400.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (6400, 101, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (6400, 101, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 6400, "Wrong number of labels in TRAIN_BALANCED. {} vs 6400.".format(train["label"].shape[0])

        train, test = fetcher.fetch_data(raw=False, dataset="TRAIN", stepsize=1, averageTrials=False)
        assert train["affected"].shape[0] == 48681, "Wrong number of measurements in TRAIN. {} vs 48681.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (48681, 101, 5), "Wrong shape of output data in TRAIN. {} vs (48681, 101, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 48681, "Wrong number of labels in TRAIN. {} vs 48681.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 21776, "Wrong number of measurements in TEST. {} vs 21776.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (21776, 101, 5), "Wrong shape of output data in TEST. {} vs (21776, 101, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 21776, "Wrong number of labels in TEST. {} vs 21776.".format(test["label"].shape[0])

        trainAVG = fetcher.fetch_set(raw=False, dataset="TRAIN", stepsize=1, averageTrials=True)
        assert trainAVG["non_affected"].shape[0] == 5817, "Wrong number of measurements in TRAIN after averaging. {} vs 5817.".format(train["non_affected"].shape[0])
        assert trainAVG["affected"].shape == (5817, 101, 5), "Wrong shape of output data in TRAIN after averaging. {} vs (5817, 101, 5).".format(train["affected"].shape)
        assert trainAVG["label"].shape[0] == 5817, "Wrong number of labels in TRAIN after averaging. {} vs 5817.".format(train["label"].shape[0])

        # 1st is SESSION_ID=413 gets assigned left side affected (with default seed) and consists of 9 trials
        # 2nd is SESSION_ID=821 with left side affected and consists of 7 trials
        # 3rd is SESSION_ID=841 with left side affected and consists of 4 trials
        for leg in ["affected", "non_affected"]:
            verifyAVG = np.mean(train[leg][:9, :, :], axis=0)
            assert np.allclose(verifyAVG, trainAVG[leg][0, :, :]), "Averaging did not produce the expected values. Expected:\n{}\nProduced:\n{}".format(verifyAVG, trainAVG[leg][0, :, :])
            verifyAVG = np.mean(train[leg][9:16, :, :], axis=0)
            assert np.allclose(verifyAVG, trainAVG[leg][1, :, :]), "Averaging did not produce the expected values. Expected:\n{}\nProduced:\n{}".format(verifyAVG, trainAVG[leg][1, :, :])
            verifyAVG = np.mean(train[leg][16:20, :, :], axis=0)
            assert np.allclose(verifyAVG, trainAVG[leg][2, :, :]), "Averaging did not produce the expected values. Expected:\n{}\nProduced:\n{}".format(verifyAVG, trainAVG[leg][2, :, :])


        #Verify for raw data
        train = fetcher.fetch_set(raw=True, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False)
        assert train["non_affected"].shape[0] == 6400, "Wrong number of measurements in TRAIN_BALANCED. {} vs 6400.".format(train["non_affected"].shape[0])
        assert train["affected"].shape == (6400, 100, 5), "Wrong shape of output data in TRAIN_BALANCED. {} vs (6400, 100, 5).".format(train["affected"].shape)
        assert train["label"].shape[0] == 6400, "Wrong number of labels in TRAIN_BALANCED. {} vs 6400.".format(train["label"].shape[0])

        train, test = fetcher.fetch_data(raw=True, dataset="TRAIN", stepsize=1, averageTrials=False)
        assert train["affected"].shape[0] == 48681, "Wrong number of measurements in TRAIN. {} vs 48681.".format(train["affected"].shape[0])
        assert train["non_affected"].shape == (48681, 100, 5), "Wrong shape of output data in TRAIN. {} vs (48681, 100, 5).".format(train["non_affected"].shape)
        assert train["label"].shape[0] == 48681, "Wrong number of labels in TRAIN. {} vs 48681.".format(train["label"].shape[0])
        assert test["affected"].shape[0] == 21776, "Wrong number of measurements in TEST. {} vs 21776.".format(test["affected"].shape[0])
        assert test["non_affected"].shape == (21776, 100, 5), "Wrong shape of output data in TEST. {} vs (21776, 100, 5).".format(test["non_affected"].shape)
        assert test["label"].shape[0] == 21776, "Wrong number of labels in TEST. {} vs 21776.".format(test["label"].shape[0])

        trainAVG = fetcher.fetch_set(raw=True, dataset="TRAIN", stepsize=1, averageTrials=True)
        assert trainAVG["non_affected"].shape[0] == 5817, "Wrong number of measurements in TRAIN after averaging. {} vs 5817.".format(train["non_affected"].shape[0])
        assert trainAVG["affected"].shape == (5817, 100, 5), "Wrong shape of output data in TRAIN after averaging. {} vs (5817, 100, 5).".format(train["affected"].shape)
        assert trainAVG["label"].shape[0] == 5817, "Wrong number of labels in TRAIN after averaging. {} vs 5817.".format(train["label"].shape[0])

        # 1st is SESSION_ID=413 gets assigned left side affected (with default seed) and consists of 9 trials
        # 2nd is SESSION_ID=821 with left side affected and consists of 7 trials
        # 3rd is SESSION_ID=841 with left side affected and consists of 4 trials
        for leg in ["affected", "non_affected"]:
            verifyAVG = np.mean(train[leg][:9, :, :], axis=0)
            assert np.allclose(verifyAVG, trainAVG[leg][0, :, :]), "Averaging did not produce the expected values. Expected:\n{}\nProduced:\n{}".format(verifyAVG, trainAVG[leg][0, :, :])
            verifyAVG = np.mean(train[leg][9:16, :, :], axis=0)
            assert np.allclose(verifyAVG, trainAVG[leg][1, :, :]), "Averaging did not produce the expected values. Expected:\n{}\nProduced:\n{}".format(verifyAVG, trainAVG[leg][1, :, :])
            verifyAVG = np.mean(train[leg][16:20, :, :], axis=0)
            assert np.allclose(verifyAVG, trainAVG[leg][2, :, :]), "Averaging did not produce the expected values. Expected:\n{}\nProduced:\n{}".format(verifyAVG, trainAVG[leg][2, :, :])



    def __assert_scaling(self, fetcher):
        """Checks whether or not the scaler is applied as intended.
        Additionally verifies that the proper exception is raised if the scaler is not of type 'GRFScaler'.
        """

        # Verify that scaling achieves the expected outcome
        scaler_auto = GRFScaler(featureRange=(0,1))
        train = fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=None)
        train_scaled, test_scaled = fetcher.fetch_data(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler_auto)
        scaler = GRFScaler(scalertype="MinMax", featureRange=(0,1))
        
        def conv_to_dict(dataset):
            data = np.moveaxis(dataset, -1, 0)
            data_dict = {
                "f_v": data[0, :, :],
                "f_ap": data[1, :, :],
                "f_ml": data[2, :, :],
                "cop_ap": data[3, :, :],
                "cop_ml": data[4, :, :]
            }
            return data_dict

        train_dict = {}
        for dataset in ["affected", "non_affected"]:
            train_dict[dataset] = conv_to_dict(train[dataset])
            scaler.partial_fit(train_dict[dataset])
            
        affected_scaled = np.moveaxis(np.asarray(list(scaler.transform(train_dict["affected"]).values()), dtype=np.float32), 0, -1)
        unaffected_scaled = np.moveaxis(np.asarray(list(scaler.transform(train_dict["non_affected"]).values()), dtype=np.float32), 0, -1)
        assert np.allclose(train_scaled["affected"], affected_scaled), "Scaling did not provide the expected outcome for the train-set."
        assert np.allclose(train_scaled["non_affected"], unaffected_scaled), "Scaling did not provide the expected outcome for the train-set."


        # Verify that the test-set was converted using the same scaler
        test = fetcher.fetch_set(raw=False, dataset="TEST", stepsize=1, averageTrials=True, scaler=None)
        test_dict = {}
        for dataset in ["affected", "non_affected"]:
            test_dict[dataset] = conv_to_dict(test[dataset])

        affected_scaled = np.moveaxis(np.asarray(list(scaler.transform(test_dict["affected"]).values()), dtype=np.float32), 0, -1)
        unaffected_scaled = np.moveaxis(np.asarray(list(scaler.transform(test_dict["non_affected"]).values()), dtype=np.float32), 0, -1)
        assert np.allclose(test_scaled["affected"], affected_scaled), "Scaling did not provide the expected outcome for the test-set."
        assert np.allclose(test_scaled["non_affected"], unaffected_scaled), "Scaling did not provide the expected outcome for the test-set."


        # Verify that the scaler is not re-fitted if already fitted
        test = fetcher.fetch_set(raw=False, dataset="TEST", stepsize=1, averageTrials=True, scaler=scaler)
        test_dict = {}
        for dataset in ["affected", "non_affected"]:
            test_dict[dataset] = conv_to_dict(test[dataset])

        assert np.allclose(test_scaled["affected"], test["affected"]), "Scaling did not provide the expected outcome for the test-set."
        assert np.allclose(test_scaled["non_affected"], test["non_affected"]), "Scaling did not provide the expected outcome for the test-set."


        # Verify that passing an unfitted scaler delivers a different result
        scaler = GRFScaler(featureRange=(0,1))
        test_scaled = fetcher.fetch_set(raw=False, dataset="TEST", stepsize=1, averageTrials=True, scaler=scaler)
        assert not np.allclose(test_scaled["affected"], affected_scaled), "Scaling did not provide the expected outcome for the test-set."
        assert not np.allclose(test_scaled["non_affected"], unaffected_scaled), "Scaling did not provide the expected outcome for the test-set."


        # Verify the currect version of fitting and scaling only the test set
        test = fetcher.fetch_set(raw=False, dataset="TEST", stepsize=1, averageTrials=True, scaler=None)
        test_dict = {}
        for dataset in ["affected", "non_affected"]:
            test_dict[dataset] = conv_to_dict(test[dataset])

        affected_scaled = np.moveaxis(np.asarray(list(scaler.transform(test_dict["affected"]).values()), dtype=np.float32), 0, -1)
        unaffected_scaled = np.moveaxis(np.asarray(list(scaler.transform(test_dict["non_affected"]).values()), dtype=np.float32), 0, -1)
        assert np.allclose(test_scaled["affected"], affected_scaled), "Scaling did not provide the expected outcome for the test-set."
        assert np.allclose(test_scaled["non_affected"], unaffected_scaled), "Scaling did not provide the expected outcome for the test-set."


        # Verify exceptions
        with self.assertRaises(ValueError):
            fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler="TEST")
            fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=55)
            fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=StandardScaler())



    def __assert_concat(self, fetcher):
        """Checks whether the concatenation of all force components delivers the expected result."""
        
        # Verify that the data is concateneted in the correct order
        train_concat, test_concat = fetcher.fetch_data(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, concat=True)

        assert train_concat["non_affected"].shape[0] == 6400, "Wrong number of measurements in TRAIN_BALANCED after concatenation. {} vs 6400.".format(train_concat["non_affected"].shape[0])
        assert train_concat["affected"].shape == (6400, 505), "Wrong shape of output data in TRAIN_BALANCED after concatenation. {} vs (6400, 505).".format(train_concat["affected"].shape)
        assert train_concat["label"].shape[0] == 6400, "Wrong number of labels in TRAIN_BALANCED after concatenation. {} vs 6400.".format(train_concat["label"].shape[0])
        assert test_concat["affected"].shape[0] == 21776, "Wrong number of measurements in TEST after concatenation. {} vs 21776.".format(test_concat["affected"].shape[0])
        assert test_concat["non_affected"].shape == (21776, 505), "Wrong shape of output data in TEST after concatenation. {} vs (21776, 505).".format(test_concat["non_affected"].shape)
        assert test_concat["label"].shape[0] == 21776, "Wrong number of labels in TEST. {} vs 21776 after concatenation.".format(test_concat["label"].shape[0])
        
        train = fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, concat=False)
        _concat(train)
        assert np.allclose(train_concat["non_affected"], train["non_affected"]), "Concatenation did not provide the expected outcome."
        test = fetcher.fetch_set(raw=False, dataset="TEST", stepsize=1, averageTrials=False, concat=False)
        _concat(test)
        assert np.allclose(test_concat["affected"], test["affected"]), "Concatenation did not provide the expected outcome."


        # Verify with different Parameters
        train_concat, test_concat = fetcher.fetch_data(raw=True, dataset="TRAIN", stepsize=5, averageTrials=True, concat=True)

        assert train_concat["non_affected"].shape[0] == 5817, "Wrong number of measurements in TRAIN_BALANCED after concatenation. {} vs 5817.".format(train_concat["non_affected"].shape[0])
        assert train_concat["affected"].shape == (5817, 100), "Wrong shape of output data in TRAIN_BALANCED after concatenation. {} vs (5817, 100).".format(train_concat["affected"].shape)
        assert train_concat["label"].shape[0] == 5817, "Wrong number of labels in TRAIN_BALANCED after concatenation. {} vs 5817.".format(train_concat["label"].shape[0])

        train = fetcher.fetch_set(raw=True, dataset="TRAIN", stepsize=5, averageTrials=True, concat=False)
        _concat(train)
        assert np.allclose(train_concat["non_affected"], train["non_affected"]), "Concatenation did not provide the expected outcome."
        assert np.allclose(train_concat["affected"], train["affected"]), "Concatenation did not provide the expected outcome."


        # Verify that even after concatenation, the data remains in the interval specified by the scaler (raises Exception) - tests only for default ordering!
        fetcher.fetch_data(raw=False, dataset="TRAIN", stepsize=1, averageTrials=True, concat=True)
        fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, concat=True)
        fetcher.fetch_data(raw=False, dataset="TRAIN", stepsize=1, averageTrials=False, concat=True)
        fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, concat=True)
        fetcher.fetch_data(raw=True, dataset="TRAIN", stepsize=1, averageTrials=True, concat=True)
        fetcher.fetch_set(raw=True, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, concat=True)
        fetcher.fetch_data(raw=True, dataset="TRAIN", stepsize=1, averageTrials=False, concat=True)
        fetcher.fetch_set(raw=True, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, concat=True)



    def __assert_rand(self, fetcher):
        """Checks if setting 'randSeed' to a different value leads to the expected behaviour.
        Additionally verifies that an exception is raised if trying to set 'randSeed' to a non-integer value.
        """

        # default component order
        filelist = ["GRF_F_V_", "GRF_F_AP_", "GRF_F_ML_", "GRF_COP_AP_", "GRF_COP_ML_"] 
        
        # Verification for processed data
        train = fetcher.fetch_set(onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=False)

        # SESSION_IDs where the left side is affected (with the default seed)
        leftSideAffected = [413, 1166, 19749, 20264]
        # SESSION_IDs where the right side is affected (with the default seed)
        rightSideAffected = [1090, 17666, 30882, 30036]

        verifyRight = pd.read_csv(filepath+"/"+filelist[0]+"PRO_right.csv", header=0)
        verifyLeft = pd.read_csv(filepath+"/"+filelist[0]+"PRO_left.csv", header=0)
        verifyAffectedL = np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyUnaffectedL = np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyAffectedR = np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyUnaffectedR = np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2)

        for datafile in filelist[1:]:
            verifyRight = pd.read_csv(filepath+"/"+datafile+"PRO_right.csv", header=0)
            verifyLeft = pd.read_csv(filepath+"/"+datafile+"PRO_left.csv", header=0)

            verifyAffectedL = np.append(verifyAffectedL, np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyUnaffectedL = np.append(verifyUnaffectedL, np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyAffectedR = np.append(verifyAffectedR, np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyUnaffectedR = np.append(verifyUnaffectedR, np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)

        def verifyTrain(train):
            for affectedSide, unaffectedSide in [(verifyAffectedL, verifyUnaffectedL), (verifyAffectedR, verifyUnaffectedR)]:
                for row in affectedSide:
                    assert np.any((train["affected"][:]==row).all(2).all(1)), "Sample: {}\n is missing in the final data for the affected side.".format(row)
                    assert not np.any((train["non_affected"][:]==row).all(2).all(1)), "Sample (affected): {}\n is contained in the final data for the unaffected side.".format(row)
                for row in unaffectedSide:
                    assert np.any((train["non_affected"][:]==row).all(2).all(1)), "Sample: {}\n is missing in the final data for the unaffected side.".format(row)
                    assert not np.any((train["affected"][:]==row).all(2).all(1)), "Sample (unaffected): {}\n is contained in the final data for the affected side.".format(row)


        # Verify that the seed is set for every call to 'fetch'
        verifyTrain(train)
        train, _ = fetcher.fetch_data(onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=False)
        verifyTrain(train)


        # Verify that a different seed changes the outcome
        fetcher.set_randSeet(0)

        # SESSION_IDs where the left side is affected (with randSeed=0)
        leftSideAffected = [1166, 19749, 20264, 1090, 30882]
        # SESSION_IDs where the right side is affected (with randSeed=0)
        rightSideAffected = [17666, 30036, 413]

        train = fetcher.fetch_set(onlyInitial=False, dropOrthopedics="None", dataset="TRAIN", averageTrials=False)
        
        # Changed outcome (the first SESSIONS have from affected to unaffected and vice-versa in both datasets)
        for affectedSide, unaffectedSide in [(verifyAffectedL, verifyUnaffectedL), (verifyAffectedR, verifyUnaffectedR)]:
            for row in affectedSide[0:3]:
                assert not np.any((train["affected"][:]==row).all(2).all(1)), "Sample: {}\n is missing in the final data for the affected side.".format(row)
                assert np.any((train["non_affected"][:]==row).all(2).all(1)), "Sample (affected): {}\n is contained in the final data for the unaffected side.".format(row)
            for row in unaffectedSide[0:3]:
                assert not np.any((train["non_affected"][:]==row).all(2).all(1)), "Sample: {}\n is missing in the final data for the unaffected side.".format(row)
                assert np.any((train["affected"][:]==row).all(2).all(1)), "Sample (unaffected): {}\n is contained in the final data for the affected side.".format(row)

        # Expected outcome
        verifyRight = pd.read_csv(filepath+"/"+filelist[0]+"PRO_right.csv", header=0)
        verifyLeft = pd.read_csv(filepath+"/"+filelist[0]+"PRO_left.csv", header=0)
        verifyAffectedL = np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyUnaffectedL = np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyAffectedR = np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2)
        verifyUnaffectedR = np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2)

        for datafile in filelist[1:]:
            verifyRight = pd.read_csv(filepath+"/"+datafile+"PRO_right.csv", header=0)
            verifyLeft = pd.read_csv(filepath+"/"+datafile+"PRO_left.csv", header=0)

            verifyAffectedL = np.append(verifyAffectedL, np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyUnaffectedL = np.append(verifyUnaffectedL, np.expand_dims(verifyLeft[verifyLeft["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyAffectedR = np.append(verifyAffectedR, np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(rightSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)
            verifyUnaffectedR = np.append(verifyUnaffectedR, np.expand_dims(verifyRight[verifyRight["SESSION_ID"].isin(leftSideAffected)].astype('float32').values[:, 3:], axis=2), axis=2)

        verifyTrain(train)


        # Verify exception
        with self.assertRaises(TypeError):
            fetcher.set_randSeet("")
            fetcher.set_randSeet("test")
            fetcher.set_randSeet(None)
            fetcher.set_randSeet('A')
            fetcher.set_randSeet(True)
            fetcher.set_randSeet(0.5)



    def __assert_comp_order(self, fetcher):
        """Checks the functionality of setting/getting the component order (including exceptions).
        Additionally verifies that changing the component order also changes the order of the values in the result.
        """

        # Verify the default component order
        order = ["f_v", "f_ap", "f_ml", "cop_ap", "cop_ml"]
        assert order == fetcher.get_comp_order(), "Default order is not correct."

        train = fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", averageTrials=False)
        train_concat = fetcher.fetch_set(raw=True, dataset="TRAIN", averageTrials=True, concat=True)
        train_pre_concat = fetcher.fetch_set(raw=True, dataset="TRAIN", averageTrials=True, concat=False)


        # Verify that the order can be changed
        new_order = ["cop_ap", "cop_ml", "f_v", "f_ap", "f_ml"]
        fetcher.set_comp_order(new_order)
        assert new_order == fetcher.get_comp_order(), "Changing the component order did not work correctly."

        train_reordered = fetcher.fetch_set(raw=False, dataset="TRAIN_BALANCED", averageTrials=False)
        train_concat_reordered = fetcher.fetch_set(raw=True, dataset="TRAIN", averageTrials=True, concat=True)


        # Verify that the order of the components has changed
        for side in ["affected", "non_affected"]:
            assert not np.allclose(train[side], train_reordered[side]), "Component order did not change after reordering (non-concatenated)."
            assert not np.allclose(train_concat[side], train_concat_reordered[side]), "Component order did not change after reordering (concatenated)."
            assert np.array_equal(train[side][:, :, [3, 4, 0, 1, 2]], train_reordered[side]), "Reordering did not produce the expected result (non-concatenated)."
            train_pre_concat[side] = train_pre_concat[side][:, :, [3, 4, 0, 1, 2]]
        
        _concat(train_pre_concat)
        for side in ["affected", "non_affected"]:
            assert np.allclose(train_pre_concat[side], train_concat_reordered[side]), "Reordering did not produce the expected result (concatenated)."
      

        # Verify exceptions
        with self.assertRaises(TypeError):
            fetcher.set_comp_order(None)
            fetcher.set_comp_order("")
            fetcher.set_comp_order(45)
            fetcher.set_comp_order("B")

        with self.assertRaises(ValueError):
            fetcher.set_comp_order(["a", "b"])
            fetcher.set_comp_order(["a", "b", "c", "d", "e"])
            fetcher.set_comp_order(["a", "b", "c", "d", "e", "f"])
            fetcher.set_comp_order(["cop_ap", "cop_ml", "f_v", "f_ap", "f_ml", "f_v"])
            fetcher.set_comp_order(["cop_ap", "cop_ml", "f_v", "f_ap", "f_ml", "cop_ap"])
            fetcher.set_comp_order(["cop_ap", "cop_ml", "f_v", "f_ap"])
            fetcher.set_comp_order(["cop_ap", "f_v", "f_ap", "f_ml"])
            fetcher.set_comp_order(["cop_ap", "cop_ml", "f_v", "f_ap", "f_ap"])
            fetcher.set_comp_order(["cop_ap", "cop_ap", "f_v", "f_ap", "f_ml"])

    



def _concat(dataset):
    for side in ["affected", "non_affected"]:
        num_ts = dataset[side].shape[1]
        dataset[side] = np.swapaxes(dataset[side], 1, 2)
        dataset[side][:,1,:] = dataset[side][:,1,:] + np.repeat((dataset[side][:,0,-1] - dataset[side][:,1,0])[:, np.newaxis], num_ts, axis=1)
        dataset[side][:,2,:] = dataset[side][:,2,:] + np.repeat((dataset[side][:,1,-1] - dataset[side][:,2,0])[:, np.newaxis], num_ts, axis=1)
        dataset[side][:,3,:] = dataset[side][:,3,:] + np.repeat((dataset[side][:,2,-1] - dataset[side][:,3,0])[:, np.newaxis], num_ts, axis=1)
        dataset[side][:,4,:] = dataset[side][:,4,:] + np.repeat((dataset[side][:,3,-1] - dataset[side][:,4,0])[:, np.newaxis], num_ts, axis=1)
        dataset[side] = np.reshape(dataset[side], (-1, 5*num_ts))



if __name__ == "__main__":
    unittest.main()
