import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import unittest
import numpy as np
from GRFScaler import GRFScaler
from sklearn.exceptions import NotFittedError

class TestScaler(unittest.TestCase):

    def test_init(self):
        """Test the inital setup of the scaler."""

        scaler = GRFScaler()
        assert scaler.get_featureRange() == (-1,1), "Scaler-type not set properly."
        assert scaler.get_type() == "minmax", "Scaler-type not set properly."
        assert scaler.is_fitted() == False, "Scaler is initally fitted."
        assert GRFScaler(scalertype="Standard").get_type() == "standard", "Scaler-type not set properly."
        assert GRFScaler(scalertype="STANDARD").get_type() == "standard", "Scaler-type not set properly."
        assert GRFScaler(scalertype="minmax").get_type() == "minmax", "Scaler-type not set properly."
        assert GRFScaler(scalertype="MinMax").get_type() == "minmax", "Scaler-type not set properly."
        assert GRFScaler(featureRange=(0, 1)).get_featureRange() == (0, 1), "Scaler-type not set properly."
        scaler = GRFScaler("standard", (-5, 10))
        assert scaler.get_featureRange() == (-5,10), "Scaler-type not set properly."
        assert scaler.get_type() == "standard", "Scaler-type not set properly."
        assert scaler.is_fitted() == False, "Scaler is initally fitted."

        # verify exception handling
        with self.assertRaises(ValueError):
            scaler = GRFScaler(scalertype="")
            scaler = GRFScaler(scalertype="test")
            scaler = GRFScaler(featureRange=(0,0))
            scaler = GRFScaler(featureRange=(1,0))


    def test_fit_transform(self):
        """Test the fit() and transform() methods"""
        data0 = {
            "cop_ap": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "cop_ml": np.array([[1,2,3], [1,2,3], [1,2,3]], dtype=np.float32),
            "f_ap": np.array([[5,4,3], [5,4,6], [2,4,3]], dtype=np.float32),
            "f_ml": np.array([[2,1,0], [2,1,3], [2,4,3]], dtype=np.float32),
            "f_v": np.array([[5,6,7], [5,6,7], [5,6,7]], dtype=np.float32)
        }
        data1 = {
            "cop_ap": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "cop_ml": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "f_ap": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "f_ml": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "f_v": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32)
        }
        # solution for data0 using MinMax(-1, 1)
        solution0 = {
            "cop_ap": np.array([[-1,-1,-1], [0,0,0], [1,1,1]], dtype=np.float32),
            "cop_ml": np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=np.float32),
            "f_ap": np.array([[0.5,0,-0.5], [0.5,0,1], [-1,0,-0.5]], dtype=np.float32),
            "f_ml": np.array([[0,-0.5,-1], [0,-0.5,0.5], [0,1,0.5]], dtype=np.float32),
            "f_v": np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=np.float32)
        }

        # test standard settings
        scaler = GRFScaler()
        scaler.fit(data0)
        assert scaler.is_fitted() == True, "Reports wrong fitted status after call to fit()."
        assert _dict_equal(solution0, scaler.transform(data0)), "Wrong scaling results for MinMax-scaler."

        scaler = GRFScaler(featureRange=(0,1))
        # solution for data0 using MinMax(0, 1)
        solution1 =  {
            "cop_ap": np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]], dtype=np.float32),
            "cop_ml": np.array([[0,0.5,1], [0,0.5,1], [0,0.5,1]], dtype=np.float32),
            "f_ap": np.array([[0.75,0.5,0.25], [0.75,0.5,1], [0,0.5,0.25]], dtype=np.float32),
            "f_ml": np.array([[0.5,0.25,0], [0.5,0.25,0.75], [0.5,1,0.75]], dtype=np.float32),
            "f_v": np.array([[0,0.5,1], [0,0.5,1], [0,0.5,1]], dtype=np.float32)
        }
        # solution for data1 using MinMax(0, 1)
        solution2 =  {
            "cop_ap": np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]], dtype=np.float32),
            "cop_ml": np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]], dtype=np.float32),
            "f_ap": np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]], dtype=np.float32),
            "f_ml": np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]], dtype=np.float32),
            "f_v": np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]], dtype=np.float32)
        }
        # solution for data0 using StandardScaler (mean=0, std=1)
        s = 0.816496581
        solution3 =  {
            "cop_ap": np.array([[-1/s,-1/s,-1/s], [0,0,0], [1/s,1/s,1/s]], dtype=np.float32),
            "cop_ml": np.array([[-1/s,-1/s,-1/s], [0,0,0], [1/s,1/s,1/s]], dtype=np.float32),
            "f_ap": np.array([[-1/s,-1/s,-1/s], [0,0,0], [1/s,1/s,1/s]], dtype=np.float32),
            "f_ml": np.array([[-1/s,-1/s,-1/s], [0,0,0], [1/s,1/s,1/s]], dtype=np.float32),
            "f_v": np.array([[-1/s,-1/s,-1/s], [0,0,0], [1/s,1/s,1/s]], dtype=np.float32)
        }
    
        # test different feature range
        scaler.fit(data0)
        assert _dict_equal(solution1, scaler.transform(data0)), "Wrong scaling results for MinMax-scaler."
        # verify that a new fit overwrites the old one
        scaler.fit(data1)
        assert not _dict_equal(solution1, scaler.transform(data0)), "Scaler did not reset between continous calls to fit()."
        assert _dict_equal(solution2, scaler.transform(data1)), "Wrong scaling results for MinMax-scaler."

        # test standard scaler
        scaler = GRFScaler(scalertype="standard")
        scaler.fit(data1)
        assert _dict_equal(solution3, scaler.transform(data1)), "Wrong scaling results for Standard-scaler."

        # verify exception handling
        del data0["f_ml"]
        del data0["f_ap"]
        del solution0["cop_ap"]
        del solution1["f_v"]
        del solution2["cop_ml"]
        scaler = GRFScaler()
        with self.assertRaises(NotFittedError):
            scaler.transform(data0)
        with self.assertRaises(ValueError):
            scaler.fit(np.array([[1,1,1], [2,2,2], [3,3,3]]))
            scaler.fit(data0)
            scaler.fit(data1)
            scaler.fit(solution0)
            scaler.fit(solution1)
            scaler.fit(solution2)


    def test_partial_fit_reset(self):
        """Test the partial_fit() and reset() methods."""
        data0 = {
            "cop_ap": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "cop_ml": np.array([[1,2,3], [1,2,3], [1,2,3]], dtype=np.float32),
            "f_ap": np.array([[5,4,3], [5,4,6], [2,4,3]], dtype=np.float32),
            "f_ml": np.array([[2,1,0], [2,1,3], [2,4,3]], dtype=np.float32),
            "f_v": np.array([[5,6,7], [5,6,7], [5,6,7]], dtype=np.float32)
        }
        data1 = {
            "cop_ap": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "cop_ml": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "f_ap": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "f_ml": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32),
            "f_v": np.array([[1,1,1], [2,2,2], [3,3,3]], dtype=np.float32)
        }
        # solution for data0 using MinMax(-1, 1)
        solution0 = {
            "cop_ap": np.array([[-1,-1,-1], [0,0,0], [1,1,1]], dtype=np.float32),
            "cop_ml": np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=np.float32),
            "f_ap": np.array([[0.5,0,-0.5], [0.5,0,1], [-1,0,-0.5]], dtype=np.float32),
            "f_ml": np.array([[0,-0.5,-1], [0,-0.5,0.5], [0,1,0.5]], dtype=np.float32),
            "f_v": np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=np.float32)
        }
        # solution for data0+data1 using MinMax(-1, 1)
        solution1 = {
            "cop_ap": np.array([[-1,-1,-1], [0,0,0], [1,1,1]], dtype=np.float32),
            "cop_ml": np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=np.float32),
            "f_ap": np.array([[0.6,0.2,-0.2], [0.6,0.2,1], [-0.6,0.2,-0.2]], dtype=np.float32),
            "f_ml": np.array([[0,-0.5,-1], [0,-0.5,0.5], [0,1,0.5]], dtype=np.float32),
            "f_v": np.array([[4/3-1,5/3-1,1], [4/3-1,5/3-1,1], [4/3-1,5/3-1,1]], dtype=np.float32)
        }
        scaler = GRFScaler()
        scaler.partial_fit(data0)
        
        # assert the intermediate result is equal to a normal fit
        assert scaler.is_fitted() == True, "Reports wrong fitted status after call to partial_fit()."
        assert _dict_equal(solution0, scaler.transform(data0)), "Wrong scaling results for MinMax-scaler."
        scaler.partial_fit(data1)
        
        # assert that the properties of the scaler have changed appropriately
        assert not _dict_equal(solution0, scaler.transform(data0)), "Wrong behaviour on continous calls to partial_fit()"
        assert _dict_equal(solution1, scaler.transform(data0)), "Wrong scaling results for MinMax-scaler (partial_fit)."

        # test reset
        scaler.reset()
        assert scaler.is_fitted() == False, "Wrong status of resetted scaler."
        scaler.partial_fit(data0)
        assert scaler.is_fitted() == True, "Reports wrong fitted status after call to partial_fit()."
        assert _dict_equal(solution0, scaler.transform(data0)), "Wrong scaling results for MinMax-scaler."





def _dict_equal(dict1, dict2):
    comp_list = ["f_v", "f_ap", "f_ml", "cop_ap", "cop_ml"]
    for component in comp_list:
        # equal within a certain tolerance (1e-08) because of rounding
        if not np.allclose(dict1[component], dict2[component]):
            return False
    return True


if __name__ == "__main__":
    unittest.main()
