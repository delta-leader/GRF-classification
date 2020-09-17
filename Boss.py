import pandas as pd
import numpy as np
from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from GRFImageConverter import GRFImageConverter
from ImageFilter import ImageFilter 
from GRFPlotter import GRFPlotter
from nolitsa.delay import dmi
from nolitsa.dimension import fnn
import matplotlib.pyplot as plt
from ModelTester import normalize_per_component
from sktime.classification.dictionary_based import BOSSEnsemble
from sklearn import metrics
from sktime.utils.load_data import load_from_arff_to_dataframe


if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    converter = GRFImageConverter()
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    class_dict = fetcher.get_class_dict()
    plotter = GRFPlotter()
    comp_order = fetcher.get_comp_order()

    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=True, val_setp=0.2, include_info=False, clip=False)

    train_data = np.concatenate([train["affected"], train["non_affected"]], axis=-1)
    val_data = np.concatenate([train["affected_val"], train["non_affected_val"]], axis=-1)

    X, _ = load_from_arff_to_dataframe("train.arff")
    Y, _ = load_from_arff_to_dataframe("test.arff")

    #X = pd.DataFrame([pd.DataFrame(pd.Series(train_data[i,:]),dtype="object") for i in range(train_data.shape[0])])
    #Y = pd.DataFrame([pd.DataFrame(pd.Series(val_data[i,:]),dtype="object") for i in range(val_data.shape[0])])
    #print(X.shape)
    #print(X.iloc[0,:])

    cboss = BOSSEnsemble(randomised_ensemble=True, n_parameter_samples=250, max_ensemble_size=50)
    print("Fitting")
    cboss.fit(X, train["label"])
    print("Predicting")
    cboss_preds = cboss.predict(Y)
    print("cBOSS Accuracy: "+str(metrics.accuracy_score(train["label_val"], cboss_preds)))