import os
from collections import namedtuple

from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from ModelTester import ModelTester, resetRand
from GRFImageConverter import GRFImageConverter
from models.MLP import create_MLP
from models.CNN1D import create_1DCNN
from models.CNN2D import create_2DCNN
from models.Alharthi import create_LSTM
from models.FCN import create_FCN
from models.Hatami import create_IMG

class Classifier(object):
    """Wrapper class for saving model configurations and settings.

    Stores configurations and weights for the best models to facilitate easy prediction and training.

    Attributes:
    models : dict
        Containing the models that can be used.
    """

    def __init__(self):
        self.models ={
            "MLP30": {
                "file": "models/saved_models/MLP30.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 1,
                    "neurons0": 30,
                    "batch_normalization": False,
                    "dropout": None,
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "MLP90": {
                "file": "models/saved_models/MLP90.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 1,
                    "neurons0": 90,
                    "batch_normalization": False,
                    "dropout": None,
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "MLP-D": {
                "file": "models/saved_models/MLPD.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 1,
                    "neurons0": 253,
                    "batch_normalization": True,
                    "dropout": 0.24681225107340823,
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "MLP2": {
                "file": "models/saved_models/MLP2.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 2,
                    "neurons0": 50,
                    "neurons1": 80,
                    "batch_normalization": False,
                    "dropout": None,
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.0018516894394818243,
                    "beta_1": 0.934566655519663,
                    "beta_2": 0.7533619902033079,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 92,
                    "epochs": 293
                }
            },
            "1DCNN-strided": {
                "file": "models/saved_models/1DCNN-strided.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 1,
                    "class_number": 5,
                    "filters0": 175,
                    "kernel0": 5,
                    "stride0": 5,
                    "dilation0": 1,
                    "batch_normalization": False,
                    "pool_type": "max",
                    "pool_size": 4,
                    "pool_stride": None,
                    "neurons": 185,
                    "dropout_cnn": 0.38775318271939535,
                    "dropout_mlp": 0.31552753454565335,
                    "separable": False,
                    "skipConnections": True,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "1DCNN-dilated": {
                "file": "models/saved_models/1DCNN-dilated.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 2,
                    "class_number": 5,
                    "filters0": 131,
                    "filters1": 31,
                    "kernel0": 18,
                    "kernel1": 6,
                    "stride0": 1,
                    "stride1": 1,
                    "dilation0": 12,
                    "dilation1": 14,
                    "batch_normalization": False,
                    "pool_type": "max",
                    "pool_size": 3,
                    "pool_stride": None,
                    "neurons": 46,
                    "dropout_cnn": 0.14204535896572307,
                    "dropout_mlp": 0.43700599895787645,
                    "separable": False,
                    "skipConnections": False,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "2DCNN-dilated": {
                "file": "models/saved_models/2DCNN-dilated.h5",
                "shape": "2D_TS1",
                "images": None,
                "non_affected": True,
                "config": {
                    "input_shape": "2D_TS1",
                    "layers": 2,
                    "filters0": 92,
                    "filters1": 134,
                    "kernel0_0": 14,
                    "kernel0_1": 5,
                    "kernel1_0": 3,
                    "kernel1_1": 3,
                    "stride0_0": 1,
                    "stride0_1": 1,
                    "stride1_0": 1,
                    "stride1_1": 1,
                    "dilation0_0": 18,
                    "dilation0_1": 1,
                    "dilation1_0": 9,
                    "dilation1_1": 2,
                    "batch_normalization": False,
                    "pool_type": "max",
                    "pool_size0": 2,
                    "pool_size1": 3,
                    "pool_stride": None,
                    "neurons": 154,
                    "dropout_cnn": 0.300265463619231,
                    "dropout_mlp": 0.2766440793017383,
                    "skipConnections": False,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "2DCNN-1DKernels": {
                "file": "models/saved_models/2DCNN-1DKernels.h5",
                "shape": "2D_TS1",
                "images": None,
                "non_affected": True,
                "config": {
                    "input_shape": "TS1",
                    "layers": 2,
                    "filters0": 63,
                    "filters1": 20,
                    "kernel0_0": 4,
                    "kernel0_1": 1,
                    "kernel1_0": 1,
                    "kernel1_1": 4,
                    "stride0_0": 1,
                    "stride0_1": 1,
                    "stride1_0": 1,
                    "stride1_1": 1,
                    "dilation0_0": 1,
                    "dilation0_1": 1,
                    "dilation1_0": 1,
                    "dilation1_1": 1,
                    "batch_normalization": True,
                    "pool_type": "max",
                    "pool_size0": 5,
                    "pool_size1": 2,
                    "pool_stride": None,
                    "neurons": 162,
                    "dropout_cnn": 0.2036748468533795,
                    "dropout_mlp": 0.1947303088468182,
                    "skipConnections": True,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.004603233045654738,
                    "beta_1": 0.7660373574383992,
                    "beta_2": 0.8063268410553384,
                    "epsilon": 1e-07,
                    "amsgrad": True,
                    "batch_size": 130,
                    "epochs": 181
                }
            },
            "LSTM": {
                "file": "models/saved_models/LSTM.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 2,
                    "units0": 100,
                    "units1": 40,
                    "neurons": 20,
                    "dropout_lstm": 0.20009757746800913,
                    "dropout_mlp": 0.4426838139700136,
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.0012235921850476889,
                    "beta_1": 0.7891835618160651,
                    "beta_2": 0.8693688573107843,
                    "epsilon": 9.058896335050494e-07,
                    "amsgrad": False,
                    "batch_size": 357,
                    "epochs": 111
                }
            },
            "FCN-tuned": {
                "file": "models/saved_models/FCN-tuned.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 3,
                    "filters0": 128,
                    "filters1": 256,
                    "filters2": 128,
                    "kernel0": 8,
                    "kernel1": 5,
                    "kernel2": 3,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.005561347878870108,
                    "beta_1": 0.7027414181263054,
                    "beta_2": 0.9611914226924457,
                    "epsilon": 1e-08,
                    "amsgrad": True,
                    "batch_size": 23,
                    "epochs": 222
                }
            },
            "FCN-original": {
                "file": "models/saved_models/FCN-original.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "layers": 3,
                    "filters0": 128,
                    "filters1": 256,
                    "filters2": 128,
                    "kernel0": 8,
                    "kernel1": 5,
                    "kernel2": 3,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-08,
                    "amsgrad": False,
                    "batch_size": 32,
                    "epochs": 100
                }
            },
            "ResNet": {
                "file": "models/saved_models/ResNet.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "blocks": 3,
                    "layers": 3,
                    "filters0": 64,
                    "filters1": 128,
                    "filters2": 128,
                    "kernel0": 8,
                    "kernel1": 5,
                    "kernel2": 3,
                    "padding": "same",
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.004467472809235925,
                    "beta_1": 0.6774131799301878,
                    "beta_2": 0.8608678157238772,
                    "epsilon": 1e-08,
                    "amsgrad": True,
                    "batch_size": 18,
                    "epochs": 210
                }
            },
            "InceptionTime": {
                "file": "models/saved_models/InceptionTime.h5",
                "shape": "1D",
                "images": None,
                "non_affected": True,
                "config": {
                    "blocks": 2,
                    "layers": 3,
                    "bottleneck_size": 5, 
                    "nb_filters": 32,
                    "kernel_sizes": [10, 20, 40],
                    "stride": 1,
                    "pool_size": 3,
                    "padding": "same",
                    "activation": "linear",
                    "activation_out": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "use_bias": False,
                    "optimizer": "adam",
                    "learning_rate": 0.006321686332796202,
                    "beta_1": 0.816970790375692,
                    "beta_2": 0.6258454624275634,
                    "epsilon": 1e-08,
                    "amsgrad": True,
                    "batch_size": 38,
                    "epochs": 136
                }
            },
            "IMG-original": {
                "file": "models/saved_models/IMG-original.h5",
                "shape": "IMG_STACK",
                "images": ["gadf"],
                "non_affected": True,
                "conv_args": {
                    "num_bins": 25,
                    "range": (-1, 1),
                    "dims": 2,
                    "delay": 3,
                    "metric": "euclidean"
                },
                "config": {
                    "layers": 2,
                    "filters0": 32,
                    "filters1": 32,
                    "kernel0": 3,
                    "kernel1": 3,
                    "padding": "valid",
                    "pool_size": 2,
                    "dropout_cnn": 0.25, 
                    "neurons": 128,
                    "dropout_mlp": 0.5,
                    "activation": "relu",
                    "final_activation": "softmax",
                    "regularizer": None,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-08,
                    "amsgrad": True,
                    "batch_size": 32,
                    "epochs": 100,
                }
            },
        }
        self.generic_models = ["MLP", "1DCNN", "FCN", "IMG"]
        self.mlps = ["MLP", "MLP30", "MLP90", "MLP2", "MLP-D"]
        self.fcns = ["FCN", "FCN-tuned", "FCN-original"]
        self.images =["IMG-original"]
        self.cnns1d = ["1DCNN-strided", "1DCNN-dilated"]
        self.cnns2d = ["2DCNN-1DKernels", "2DCNN-dilated"]
        self.lstms = ["LSTM"]

    def train_and_predict(self, model, data, test=None, boosting=False, config=None, shape=None, images=None, useNonAffected=True, deterministic=True, name=None, log=True, save_plot=False, show_plot=True, plot_architecture=False, loss=None, metrics=None, class_dict=None, filepath=None):
        
        model, shape, images, useNonAffected, data = self.train(model, data, config=config, shape=shape, images=images, useNonAffected=useNonAffected, deterministic=deterministic, name=name, store=True, log=log, save_plot=save_plot, show_plot=show_plot, plot_architecture=plot_architecture, loss=loss, metrics=metrics, class_dict=class_dict, filepath=filepath)
        if test is None:
            self.predict(model=model, data=data, val_set=True, boosting=boosting, shape=shape, images=images, useNonAffected=useNonAffected, loss=loss, metrics=metrics, class_dict=class_dict, filepath=filepath)
        else:
            self.predict(model=model, data=test, val_set=False, boosting=boosting, shape=shape, images=images, useNonAffected=useNonAffected, loss=loss, metrics=metrics, class_dict=class_dict, filepath=filepath)


    def predict(self, model, data, val_set=False, boosting=False, shape="1D", images=[], useNonAffected=True, loss=None, metrics=None, class_dict=None, filepath=None):

        if not isinstance(model, str):
            raise TypeError("Please specify the model to use as a string containing either the name of a pre-defined model or the filepath to a self-created one.")

        tester = self.__create_tester(loss, metrics, class_dict, filepath)

        if model in self.models.keys():
            model = self.models[model]
            tester.predict_model(model["file"], data, val_set, boosting, model["shape"], model["images"], model["non_affected"])
        else:
            tester.predict_model(model, data, val_set, boosting, shape, images, useNonAffected)



    def train(self, model, data, config=None, shape=None, images=None, useNonAffected=True, deterministic=True, name=None, store=True, log=True, save_plot=False, show_plot=True, plot_architecture=False, loss=None, metrics=None, class_dict=None, filepath=None):

        if not isinstance(model, str):
            raise TypeError("Please specify the model to use as a string containing either the name of a pre-defined model or the filepath to a self-created one.")

        if name is None:
            name = model
        if model in self.models.keys():
            shape = self.models[model]["shape"]
            images = self.models[model]["images"]
            useNonAffected = self.models[model]["non_affected"]
            config = self.models[model]["config"]
        else:   
            if model in self.generic_models:
                if config is None:
                    raise ValueError("A configuration needs to be passed in order to create a generic "+model+"-model.")
            else:
                raise ValueError("Model of type '{}' is not known to the classifier.".format(model))

        config = namedtuple("Config", config.keys())(*config.values())
        if deterministic:
            resetRand()
        
        keras_model = None
        if model in self.mlps:
            keras_model = create_MLP(input_shape=(train["affected"].shape[1]*2,), config=config)
            if shape != "1D":
                raise ValueError("A MLP can only be trained with shape='1D'.")
            if images is not None:
                raise ValueError("Images have been specified, but the MLP can not be used to classify image-data.")

        if model in self.cnns1d:
            keras_model = create_1DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            if shape != "1D":
                raise ValueError("A 1DCNN can only be trained with shape='1D'.")
            if images is not None:
                raise ValueError("Images have been specified, but the 1DCNN can not be used to classify image-data.")
        if model in self.lstms:
            keras_model = create_LSTM(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            if shape != "1D":
                raise ValueError("A LSTM can only be trained with shape='1D'.")
            if images is not None:
                raise ValueError("Images have been specified, but the LSTM can not be used to classify image-data.")
        if model in self.cnns2d:
            keras_model = create_2DCNN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2, 1), config=config)
            if shape != "2D_TS1":
                raise ValueError("A 2DCNN can only be trained with shape='2D_TS1'.")
            if images is not None:
                raise ValueError("Images have been specified, but the 2DCNN can not be used to classify image-data.")
        if model in self.fcns:
            keras_model = create_FCN(input_shape=(train["affected"].shape[1], train["affected"].shape[2]*2), config=config)
            if shape != "1D":
                raise ValueError("A FCN can only be trained with shape='1D'.")
            if images is not None:
                raise ValueError("Images have been specified, but the FCN can not be used to classify image-data.")
        if model in self.images:
            converter = GRFImageConverter()
            img_data = converter.convert(data, conversions=["gaf"], conv_args=self.models[model]["conv_args"])
            for key in ["affected", "non_affected", "affected_val", "non_affected_val"]:
                data[key] = img_data[key]
            img = images[0]
            count = len(images)
            keras_model = create_IMG(input_shape=(data["affected"][img].shape[1], data["affected"][img].shape[2], data["affected"][img].shape[3]*count*2), config=config)
            

        if filepath is None:
            filepath = "models/output/"+name+"/"
        tester = self.__create_tester(loss, metrics, class_dict, filepath)
        logfile = name + ".dat"   
        storepath = None
        if store:
            storepath=filepath+name+".h5"
        if plot_architecture:
            tester.save_model_plot(keras_model, name+".png")   

        tester.test_model(keras_model, train=data, config=config, shape=shape, images=images, useNonAffected=useNonAffected, logfile=logfile, model_name=name, plot_name=name+".png", create_plot=save_plot, show_plot=show_plot, store_model=storepath, boost=False)
        if not log:
            os.remove(filepath+logfile)

        if log or save_plot or store or plot_architecture:
            print("All files have been saved to '{}'.".format(filepath))
        else:
            try:
                os.rmdir(filepath)
            except OSError:
                return
        if store:
            print("The trained model can be loaded from '{}'.".format(storepath))

        return storepath, shape, images, useNonAffected, data


    def __create_tester(self, loss, metrics, class_dict, filepath):

        if loss is None:       
            loss="categorical_crossentropy"
        if metrics is None:
            metrics=["accuracy"]
        if class_dict is None:
            class_dict={"HC":0, "H":1, "K":2, "A":3, "C":4}
        if filepath is None:
            filepath="models/output/"

        return ModelTester(loss, metrics, class_dict, filepath)


if __name__ == "__main__":
    #filepath = "../"
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    #scaler = GRFScaler(scalertype="standard")
    #class_dict = {"HC":0, "H":1, "K":1, "A":2, "C":2}
    #fetcher.set_class_dict(class_dict)
    #val = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False, clip=False)
    test = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TEST", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0, include_info=False, clip=False)
    #train = set_valSet(train, val, parse="SESSION_ID")

    classifier = Classifier()

    converter = GRFImageConverter()
    conv_args ={
                    "num_bins": 25,
                    "range": (-1, 1),
                    "dims": 2,
                    "delay": 3,
                    "metric": "euclidean"
                },
    #img_data = converter.convert(test, conversions=["gaf"], conv_args=conv_args)
    #for key in ["affected", "non_affected"]:
    #    test[key] = img_data[key]

    classifier.predict("1DCNN-strided", test, images=[], val_set=False, boosting=False)
    #classifier.train_and_predict("MLP30", train, test, name=None, log=False, save_plot=False, show_plot=False, plot_architecture=False, boosting=False)
    #train(self, model, data, deterministic=True, name=None, store=True, log=True, save_plot=False, show_plot=True, plot_architecture=False, loss=None, metrics=None, class_dict=None, filepath=None):
    #classifier.predict("models/output/MLP1/MLP1.h5", train, val_set=True, boosting=False)
    