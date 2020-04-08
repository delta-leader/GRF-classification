import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import unittest
import numpy as np
import pandas as pd
from DataFetcher import DataFetcher
from GRFScaler import GRFScaler
from DataFetcher import _select_initial_measurements
from DataFetcher import _drop_orthopedics
from DataFetcher import _trim_metadata
from DataFetcher import _select_dataset
from DataFetcher import _sample
from DataFetcher import _average_trials
from DataFetcher import _scale
from DataFetcher import _fit_scaler
from DataFetcher import _arrange_data

filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC"

class TestFetcher(unittest.TestCase):
    
    def test_init(self):
        fetcher = DataFetcher(filepath)
        fetcher = DataFetcher(filepath + "/")
        with self.assertRaises(IOError):
            fetcher = DataFetcher(filepath + "/test")
            fetcher = DataFetcher("/test")
            fetcher = DataFetcher("")

    
    def test_data_selection_and_trim(self):
        metadata = pd.read_csv(filepath+"/GRF_metadata.csv", header=0)
        # only intial measurements
        assert _select_initial_measurements(metadata).shape[0] == 2705, "Wrong number of inital measurements."

        # drop orthopedics
        assert _drop_orthopedics(metadata).shape[0] == 7628, "Wrong number of measurements without orthopedic assistance."

        # trim data
        test1 = _trim_metadata(metadata, False)
        for element in ["SUBJECT_ID", "SESSION_ID", "CLASS_LABEL", "AFFECTED_SIDE", "TRAIN", "TRAIN_BALANCED", "TEST"]:
            assert element in test1.columns, "Column '{}' missing.".format(element)
        for element in ["CLASS_LABEL_DETAILED", "SEX", "AGE", "HEIGHT", "BODY_WEIGHT", "BODY_MASS", "SHOE_SIZE", "SHOD_CONDITON", "ORTHOPEDIC_INSOLE", "SPEED", "READMISSION", "SESSIOM_TYPE", "SESSION_DATE"]:
            assert element not in test1.columns, "Column '{}' still exists after trimming.".format(element)
        test2 = _trim_metadata(metadata, True)
        for element in ["SUBJECT_ID", "SESSION_ID", "CLASS_LABEL", "AFFECTED_SIDE", "TRAIN", "TRAIN_BALANCED", "TEST", "BODY_WEIGHT", "BODY_MASS", "SHOE_SIZE"]:
            assert element in test2.columns, "Column '{}' missing.".format(element)
        for element in ["CLASS_LABEL_DETAILED", "SEX", "AGE", "HEIGHT", "SHOD_CONDITON", "ORTHOPEDIC_INSOLE", "SPEED", "READMISSION", "SESSIOM_TYPE", "SESSION_DATE"]:
            assert element not in test2.columns, "Column '{}' still exists after trimming.".format(element)

        # select dataset
        assert _select_dataset(metadata, "TRAIN").shape[0] == 6197, "Wrong number of samples in TRAIN-dataset."
        assert _select_dataset(metadata, "TRAIN_BALANCED").shape[0] == 730, "Wrong number of samples in TRAIN_BALANCED-dataset."
        assert _select_dataset(metadata, "TEST").shape[0] == 2775, "Wrong number of samples in TEST-dataset."
    
    
    def test_data_fetch_and_sample(self):
        fetcher = DataFetcher(filepath)
        metadata = fetcher._DataFetcher__fetch_metadata() 
        metadata = metadata.sample(n=5)
        
        # test data fetch & selection for processed data
        left, right = fetcher._DataFetcher__fetch_data(metadata, raw=False)
        for component in fetcher.get_comp_order():
            assert component in left.keys() and component in right.keys(), "Error: {} not available in the dictionary.".format(component)
            assert left[component]["SESSION_ID"].isin(metadata["SESSION_ID"]).all() == True, "Fetched data that was not requested for component {}.".format(component)
            assert right[component]["SESSION_ID"].isin(metadata["SESSION_ID"]).all() == True, "Fetched data that was not requested for component {}.".format(component)
        filelist = ["GRF_COP_AP_", "GRF_COP_ML_", "GRF_F_AP_", "GRF_F_ML_", "GRF_F_V_"] 
        for element in filelist:
            component_name = element[element.index("_")+1:-1].lower()
            test_data = pd.read_csv(filepath+"/"+element+"PRO_right.csv")
            test_data = test_data.astype({test_data.columns[3]: 'float64'})
            test_data = test_data[test_data["SESSION_ID"].isin(metadata["SESSION_ID"])].reset_index(drop=True)
            assert test_data.equals(right[component_name]), "Data for {} component differs from original one.".format(component_name)

        # test sampling for processed data
        left_sampled = _sample(left, stepsize=1, raw=False)
        assert _DFdicts_equal(left_sampled, left), "There should be no sampling if stepsize = 1."
        left_sampled1 = _sample(left, stepsize=2, raw=False)
        left_sampled2 = _sample(left, stepsize=3, raw=False)
        left_sampled3 = _sample(left, stepsize=10, raw=False)
        for component in left_sampled.keys():
            assert left_sampled1[component].shape[1] == (int) (np.ceil((left[component].shape[1] - 3) /2)) + 3, "Size after sampling not appropriate."
            assert left_sampled2[component].shape[1] == (int) (np.ceil((left[component].shape[1] - 3) /3)) + 3, "Size after sampling not appropriate."
            assert left_sampled3[component].shape[1] == (int) (np.ceil((left[component].shape[1] - 3) /10)) + 3, "Size after sampling not appropriate."
            time_steps = left_sampled2[component].shape[1]
            for i in range(3, time_steps):
                assert left_sampled2[component].iloc[:, i].equals(left[component].iloc[:, i+2*(i-3)]), "Sampled data does not match original data."
        
        # test data fetch & selection for raw data
        left, right = fetcher._DataFetcher__fetch_data(metadata, raw=True)
        for component in fetcher.get_comp_order():
            assert component in left.keys() and component in right.keys(), "Error: {} not available in the dictionary.".format(component)
            assert left[component]["SESSION_ID"].isin(metadata["SESSION_ID"]).all() == True, "Fetched data that was not requested for component {}.".format(component)
            assert right[component]["SESSION_ID"].isin(metadata["SESSION_ID"]).all() == True, "Fetched data that was not requested for component {}.".format(component)
        filelist = ["GRF_COP_AP_", "GRF_COP_ML_", "GRF_F_AP_", "GRF_F_ML_", "GRF_F_V_"] 
        for element in filelist:
            component_name = element[element.index("_")+1:-1].lower()
            test_data = pd.read_csv(filepath+"/"+element+"RAW_right.csv")
            test_data = test_data[test_data["SESSION_ID"].isin(metadata["SESSION_ID"])].reset_index(drop=True)
            assert test_data.equals(right[component_name]), "Data for {} component differs from original one.".format(component_name)
        
        # test sampling for raw data
        left_sampled1 = _sample(left, stepsize=2, raw=True)
        left_sampled2 = _sample(left, stepsize=3, raw=True)
        left_sampled3 = _sample(left, stepsize=10, raw=True)
        for component in left_sampled1.keys():
            assert left_sampled1[component].shape[1] == (int) (100/2+3), "Size after sampling not appropriate."
            assert left_sampled2[component].shape[1] == (int) (100/3+3), "Size after sampling not appropriate."
            assert left_sampled3[component].shape[1] == (int) (100/10+3), "Size after sampling not appropriate."
    
    
    def test_average_trials_and_scale(self):
        fetcher = DataFetcher(filepath)
        metadata = fetcher._DataFetcher__fetch_metadata() 
        metadata = metadata.sample(n=5)

        # test averaging
        left, right = fetcher._DataFetcher__fetch_data(metadata, raw=False)
        left_avg = _average_trials(left)
        for component in left.keys():
            assert left_avg[component].shape[0] == 5, "Averaging did not reduce data to unique SESSION_IDs."
            id = left_avg[component].iloc[0, :]["SESSION_ID"]
            left[component] = left[component][left[component]["SESSION_ID"] == id]
            left_avg[component] = left_avg[component][left_avg[component]["SESSION_ID"] == id]
            assert left_avg[component].iloc[:, 2:].equals(left[component].iloc[:, 3:].mean(axis=0).to_frame().transpose()), "Averaging did not produce the expected values."

        # test scaling for averaged data
        right_avg = _average_trials(_sample(right, stepsize=10, raw=False))
        left_avg = _average_trials(_sample(left, stepsize=10, raw=False))
        scaler = GRFScaler(featureRange=(0, 1))
        _fit_scaler(scaler, (right_avg, left_avg))
        right_scaled = _scale(scaler, right_avg)
        for component in right_scaled.keys():
            assert right_scaled[component].shape == right_avg[component].shape, "Shape does not match after scaling."
            data1 = right_avg[component].iloc[:, 2:]
            data2 = left_avg[component].iloc[:, 2:]
            dmin = min([data1.values.min(), data2.values.min()])
            dmax = max([data1.values.max(), data2.values.max()])
            data1 = data1.applymap(lambda x: (x - dmin) / (dmax - dmin))
            assert np.allclose(right_scaled[component].iloc[:, 2:].values, data1.values, rtol=1e-4, atol=1e-8), "Scaling does not produce the expected result."
            assert right_scaled[component].iloc[:, :2].equals(right_avg[component].iloc[:, :2]), "Scaling messes up the meta-information columns."
        
        # test scaling for non-averaged data
        scaler = GRFScaler(featureRange=(0, 1))
        _fit_scaler(scaler, (right, left))
        left_scaled = _scale(scaler, left)
        for component in left_scaled.keys():
            assert left_scaled[component].shape == left[component].shape, "Shape does not match after scaling."
            data1 = right[component].iloc[:, 3:]
            data2 = left[component].iloc[:, 3:]
            dmin = min([data1.values.min(), data2.values.min()])
            dmax = max([data1.values.max(), data2.values.max()])
            data2 = data2.applymap(lambda x: (x - dmin) / (dmax - dmin))
            assert np.allclose(left_scaled[component].iloc[:, 3:].values, data2.values, rtol=1e-4, atol=1e-8), "Scaling does not produce the expected result."
            assert left_scaled[component].iloc[:, :3].equals(left[component].iloc[:, :3]), "Scaling messes up the meta-information columns."

    
    def test_concat(self):
        fetcher = DataFetcher(filepath)
        metadata = fetcher._DataFetcher__fetch_metadata() 
        metadata = metadata.sample(n=5)
        left, right = fetcher._DataFetcher__fetch_data(metadata, raw=False)

        # test concat for averaged data
        left_avg = _average_trials(left)
        left_concat = fetcher._DataFetcher__concat(left_avg)
        order = fetcher.get_comp_order()
        assert len(left_concat.keys()) == 1, "Too many keys in dictionary after concatenation."
        assert "concat" in left_concat.keys(), "'concat' is no key in the dictionary."
        assert left_concat["concat"].iloc[:, :103].equals(left_avg[order[0]]), "Component {} does not match in concatenated dict.".format(order[0])
        nextIndex = 103
        lastColumn = left_avg[order[0]].iloc[:, -1]
        for component in order[1:]:
            carry = lastColumn - left_avg[component].iloc[:, 2]
            assert left_concat["concat"].iloc[:, nextIndex:(nextIndex + 101)].equals(left_avg[component].iloc[:, 2:].add(carry, axis="index")), "Component {} does not match in concatenated dict.".format(component)
            nextIndex += 101
            lastColumn = left_avg[component].iloc[:, -1].add(carry, axis="index")

        # test for non-averaged data
        right_concat = fetcher._DataFetcher__concat(right)
        order = fetcher.get_comp_order()
        assert len(right_concat.keys()) == 1, "Too many keys in dictionary after concatenation."
        assert "concat" in right_concat.keys(), "'concat' is no key in the dictionary."
        assert right_concat["concat"].iloc[:, :104].equals(right[order[0]]), "Component {} does not match in concatenated dict.".format(order[0])
        nextIndex = 104
        lastColumn = right[order[0]].iloc[:, -1]
        for component in order[1:]:
            carry = lastColumn - right[component].iloc[:, 3]
            assert right_concat["concat"].iloc[:, nextIndex:(nextIndex + 101)].equals(right[component].iloc[:, 3:].add(carry, axis="index")), "Component {} does not match in concatenated dict.".format(component)
            nextIndex += 101
            lastColumn = right[component].iloc[:, -1].add(carry, axis="index")


    def test_arrange_and_format(self):
        fetcher = DataFetcher(filepath)
        metadata = fetcher._DataFetcher__fetch_metadata() 
        metadata = metadata.sample(n=5)
        left, right = fetcher._DataFetcher__fetch_data(metadata, raw=False)

        affected, non_affected = _arrange_data(left, right, metadata, 1)
        leftSideAffected = metadata[metadata["AFFECTED_SIDE"] == 0]
        rightSideAffected = metadata[metadata["AFFECTED_SIDE"] == 1]
        
        for component in affected.keys():
            assert affected[component][affected[component]["SESSION_ID"].isin(leftSideAffected["SESSION_ID"])].iloc[:, :100].equals(left[component][left[component]["SESSION_ID"].isin(leftSideAffected["SESSION_ID"])].iloc[:, :100]), "Did not assign affected leg correctly (left)."
            assert non_affected[component][non_affected[component]["SESSION_ID"].isin(rightSideAffected["SESSION_ID"])].iloc[:, :100].equals(left[component][left[component]["SESSION_ID"].isin(rightSideAffected["SESSION_ID"])].iloc[:, :100]), "Did not assign unaffected leg correctly (left)."
            assert not (affected[component].iloc[:, :100] == non_affected[component].iloc[:, :100]).all(axis=1).any(), "Affected and unaffected side have the same data."
            assert affected[component][affected[component]["SESSION_ID"].isin(rightSideAffected["SESSION_ID"])].iloc[:, :100].equals(right[component][right[component]["SESSION_ID"].isin(rightSideAffected["SESSION_ID"])].iloc[:, :100]), "Did not assign affected leg correctly (right)."
            assert non_affected[component][non_affected[component]["SESSION_ID"].isin(leftSideAffected["SESSION_ID"])].iloc[:, :100].equals(right[component][right[component]["SESSION_ID"].isin(leftSideAffected["SESSION_ID"])].iloc[:, :100]), "Did not assign unaffected leg correctly (right)."

        data = fetcher._DataFetcher__split_and_format(affected, non_affected) 
        component = list(affected.keys())[0]
        assert np.equal(data["label"], affected[component]["CLASS_LABEL"].map({"HC":0, "H":1, "K":2, "A":3, "C":4}).values).all(), "Class labels do not match."
        assert data["affected"].shape == (affected[component].shape[0], 101, 5), "Output shape incorrect (affected)."
        assert data["non_affected"].shape == (affected[component].shape[0], 101, 5), "Output shape incorrect (non_affected)."

        comp_list = list(affected.keys())
        for i in range(5):
            assert np.equal(data["affected"][:, :, i], np.asarray(affected[comp_list[i]].iloc[:, 3:104]), dtype=np.float32).all(), "Data for {} component does not match".format(comp_list[i])
        #TODO test for averaged data/concatenated data




def _DFdicts_equal(dict1, dict2):
    if len(dict1) != len(dict2):
        return False
        
    for component in dict1.keys():
        if not dict1[component].equals(dict2[component]):
            return False
    return True



#TODO verfiy scale after concatenation!!        
#TODO Verify how to proceed with NANs in ORTHOPEDIC_INSOLES
"""onlyInitial=False
dropOrthopedics=True
dataset="TRAIN_BALANCED"
raw=False
stepsize=1
averageTrials=True
scaler=None
concat=False"""




if __name__ == "__main__":
    unittest.main()
