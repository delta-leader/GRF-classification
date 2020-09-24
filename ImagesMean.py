import pandas as pd
import numpy as np
import arff
from DataFetcher import DataFetcher, set_valSet
from GRFScaler import GRFScaler
from GRFImageConverter import GRFImageConverter
from ImageFilter import ImageFilter 
from GRFPlotter import GRFPlotter
from nolitsa.delay import dmi
from nolitsa.dimension import fnn
import matplotlib.pyplot as plt
from ModelTester import normalize_per_component, create_heatmap


if __name__ == "__main__":
    filepath = "/media/thomas/Data/TT/Masterarbeit/final_data/GAITREC/"
    fetcher = DataFetcher(filepath)
    converter = GRFImageConverter()
    scaler = GRFScaler(scalertype="MinMax", featureRange=(-1,1))
    class_dict = fetcher.get_class_dict()
    plotter = GRFPlotter()
    comp_order = fetcher.get_comp_order()

    acc = np.array([[0.68,0.68,0.68,0.69,0.68,0.69,0.66,0.66],
                [0.68,0.67,0.67,0.65,0.68,0.66,0.64,0.62],
                [0.66,0.66,0.67,0.67,0.68,0.67,0.68,0.67],
                [0.66,0.66,0.68,0.66,0.67,0.66,0.65,0.64],
                [0.72,0.67,0.67,0.69,0.68,0.68,0.66,0.66],
                [0.70,0.67,0.69,0.67,0.68,0.68,0.66,0.60],
                [0.68,0.68,0.69,0.69,0.70,0.68,0.70,0.66],
                [0.68,0.68,0.68,0.69,0.68,0.68,0.67,0.69]])
    y=[(11,2),(11,3),(11,4),(11,5),(11,6),(11,7),(11,8),(11,9)]
    x=[8,16,32,64,128,256,512,1024]
    create_heatmap(acc, y, x, "2DCNN_heatmap")


    train = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=True, val_setp=0.2, include_info=False, clip=False)

    train_data = np.concatenate([train["affected"], train["non_affected"], np.expand_dims(train["label"], axis=-1)], axis=-1)
    val_data = np.concatenate([train["affected_val"], train["non_affected_val"], np.expand_dims(train["label_val"], axis=-1)], axis=-1)
    train_arff = {
        "descprition": "Training data for GRF measurements",
        "relation": "TRAIN_BALANCED",
        "attributes": [("attr"+str(i), "NUMERIC") for i in range(train_data.shape[1]-1)] + [("label", "INTEGER")],
        "data": train_data
    }
    val_arff = {
        "descprition": "Training data for GRF measurements",
        "relation": "TRAIN_BALANCED",
        "attributes": [("attr"+str(i), "NUMERIC") for i in range(val_data.shape[1]-1)] + [("label", "INTEGER")],
        "data": val_data
    }

    with open("train.arff", "w", encoding="utf8") as f:
        arff.dump(train_arff, f)
    with open("test.arff", "w", encoding="utf8") as f:
        arff.dump(val_arff, f)

    """
    train1 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=False, clip=True)
    #train2 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=2, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    #train3 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=3, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    train5 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=5, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    #train5 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=5, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    train10 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=10, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    train15 = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=15, averageTrials=True, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)

    line_labels=["resampled_5", "resampled_10", "resampled_15", "original"]
    fig, (ax1, ax2, ax3)= plt.subplots(3,1)
    fig.suptitle("Resamplings of the F_V component", fontweight="bold", fontsize=14)
    #t = range(0,101,1)
    #plt.plot(t,train1["non_affected"][1,:,0], label="original", color="orangered", linewidth=2, linestyle="dashed")
    #t = range(0,101,2)
    #plt.plot(t,train2["non_affected"][1,:,0], label="resampled_2", color="#023e8a", linewidth=2)
    #t = range(0,101,3)
    #plt.plot(t,train3["non_affected"][1,:,0], label="resampled_3", color="#0077b6", linewidth=2)
    #t = range(0,101,4)
    #plt.plot(t,train4["non_affected"][1,:,0], label="resampled_4", color="#023e8a", linewidth=4)
    t = range(0,101,5)
    l1,=ax1.plot(t,train5["non_affected"][1,:,0], color="#1d3557", linewidth=3)
    t = range(0,101,10)
    l2,=ax2.plot(t,train10["non_affected"][1,:,0], color="#457b9d", linewidth=3)
    t = range(0,101,15)
    l3,=ax3.plot(t,train15["non_affected"][1,:,0], color="#a8dadc", linewidth=3)
    #plt.title("Resamplings of the F_V component", fontweight="bold", fontsize=14)
    t = range(0,101,1)
    l4, = ax1.plot(t,train1["non_affected"][1,:,0], color="red", linewidth=2, linestyle="dashed")
    ax2.plot(t,train1["non_affected"][1,:,0], label="original", color="red", linewidth=2, linestyle="dashed")
    ax3.plot(t,train1["non_affected"][1,:,0], color="red", linewidth=2, linestyle="dashed")
    plt.xlabel("Time")
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax3.set_xlim(0, 100)
    #ax = plt.gca()
    #plt.set_xlim(0, 100)
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.85, box.height * 0.85])
    #plt.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.)
    fig.legend(handles=[l1, l2, l3, l4], labels=line_labels, loc="center right", borderaxespad=0.1, bbox_to_anchor=(0.95,0.5))
    plt.subplots_adjust(right=0.65)
    plt.show()
   
    
    train = fetcher.fetch_data(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=False, scaler=scaler, concat=False, val_setp=0.2, include_info=True, clip=True)
    train_all = fetcher.fetch_set(raw=False, onlyInitial=True, dropOrthopedics="All", dropBothSidesAffected=False, dataset="TRAIN", stepsize=1, averageTrials=False, scaler=scaler, concat=False, include_info=True, clip=True)
    print(train_all["affected"].shape)
    print(train_all["info"])
    train_all = set_valSet(train_all, train, 'SUBJECT_ID')
    print(train_all["affected"].shape)
    print(train_all["non_affected"].shape)
    print(train_all["label"].shape)
    print(train_all["info"])
    print(train_all["info_val"][train_all["info_val"]["SUBJECT_ID"].isin(train_all["info"]["SUBJECT_ID"])])
    print(train_all["affected_val"].shape)
    print(train_all["non_affected_val"].shape)
    print(train_all["label_val"].shape)

    #train = normalize_per_component(test, ["affected"])

    converter.enableGpu()
    conv_args = {
        "num_bins": 20,
         "range": (-1, 1),
         "dims": 2,
         "delay": 3,
         "metric": "euclidean"
    }
    imgFilter = ImageFilter("avg", (7,7))
    gaf_images = converter.convert(train, conversions="rcp", conv_args=conv_args, imgFilter=None)

    sample = train["affected"][73]
    num_states = 101 - (2-1) * 3
    #plt.plot(sample[:,1], color="#faC05e", label="F_ML", linewidth=2)
    #plt.title("F_ML", fontweight="bold", fontsize=14)
    #plt.xlabel("Time")
    #ax = plt.gca()
    #ax.set_xlim(0, 100)
    #plt.show()
    #plt.figure()
    states = np.lib.stride_tricks.as_strided(sample[:,2], (num_states, 2), (4, 3*4))
    x = states[:,0]
    y=states[:,1]
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, alpha=0.7, color="#faC05e")
    ax = plt.gca()

    for i in range(0, 98, 5):
        ax.annotate(r"S$_{"+str(i)+r"}$", (x[i], y[i]), fontweight="bold", color="#3f7cac" )

    
    plt.title("2D phase space trajectory", fontweight="bold", fontsize=14)

    plt.show()
    #angle = np.arccos(np.maximum(-1, np.minimum(1, sample[:,0])))
    #r = np.array(range(101))
    #plt.polar(angle,r)
    #plt.show()

    plotter.plot_image(gaf_images, sampleIndex=73, keys="affected", images=["rcp"], comp_order=fetcher.get_comp_order(), show=True, save=False, vmin=0, vmax=1)
    #gaf_images = converter.convert(train, conversions="mtf", conv_args=conv_args, imgFilter=imgFilter)
    #plotter.plot_image(gaf_images, sampleIndex=128, keys="affected", images=["mtf"], comp_order=fetcher.get_comp_order(), show=True, save=False, vmin=0, vmax=1)

    #plotting pictures for paper
    #indices = np.where(train["label"] == class_dict.get("HC"))[0]
    #data = np.take(train["affected"], indices, axis=0)
    #info = np.take(train["info"], indices, axis=0)
    #plt.plot(data[365, :, 0], color="#3f7cac", label="F_V", linewidth=2)
    #plt.plot(data[365, :, 1], color="#ee6352", label="F_AP", linewidth=2)
    #plt.plot(data[365, :, 2], color="#faC05e", label="F_ML", linewidth=2)
    #plt.plot(data[365, :, 3], color="#59cd90", label="COP_AP", linewidth=2)
    #plt.plot(data[365, :, 4], color="#631d76", label="COP_ML", linewidth=2)
    #plt.title("GRF-components", fontweight="bold", fontsize=14)
    #plt.xlabel("Time")
    #ax = plt.gca()
    #ax.set_xlim(0, 100)
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.85, box.height * 0.85])
    #plt.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.)
    
    
    #plt.show()
    """
    
    
    """
    train_mean = np.mean(train["affected"], axis=0)
    print(train_mean.shape)
    for i in range(5):
        #delay = dmi(train_mean[:,i], maxtau=100)
        for j in range(4):
            f1, f2, f3 = fnn(train_mean[:,i], dim=[1+j], tau=5)
            print(f1)
            print(f2)
            print(f3)
            print("")
        print("Next Signal")
        #plt.plot(f1)
    #plt.show()
    
    mean_col = ["red", "darkorange", "green", "blue", "mediumorchid"]
    std_col =["tomato", "wheat", "mediumseagreen", "cornflowerblue", "violet"]

    converter.enableGpu()
    conv_args = {
        "num_bins": 32,
         "range": (-1, 1),
         "dims": 3,
         "delay": 3,
         "metric": "euclidean"
    }
    imgFilter = ImageFilter("avg", (5,5))
    #gaf_images = converter.convert(train, conversions="mtf", conv_args=conv_args, imgFilter=imgFilter)

    for injury_class in ["A", "C"]:
        indices = np.where(train["label"] == class_dict.get(injury_class))[0]
        data = np.take(train["affected"], indices, axis=0)
        info = np.take(train["info"], indices, axis=0)
        instances = 0
        for session in info["SESSION_ID"].unique():
            session_indices = np.where(info["SESSION_ID"]==session)[0]
            single_sess = np.take(data, session_indices, axis=0)
            passed = 0
            not_passed = 0
            #print(single_sess.shape)
            for trial in single_sess:
                if (trial[89,1] <= trial[89,3]):     
                    passed +=1
                else:
                    not_passed +=1
            #print("{} passed, {} not passed".format(passed, not_passed))
            if passed >= not_passed:
                instances += 1
        print("{} has {} instances out of {}".format(injury_class, instances, info["SESSION_ID"].unique().shape[0]))
    """
    """
        for i in range(data.shape[0]):
            if not (data[i,89,1] <= data[i,89,3] and data[i,74,1] > 0.265):
                    #if img[74,3] <= img[74,4]:
                    instances +=1
            #if img[89,1] <= img[89,3]:
            #if img[86,0] <= img[86,4]:
            #if img[76,0] <= img[76,1]:
            #if img[30,0] <= img[30,2]:
            #if img[74,1] > 0.265:
            #if img[74,3] <= img[74,4]:
            #    instances +=1
        print("{} has {} instances out of {}".format(injury_class, instances, data.shape[0]))
    """
    """
        plt.figure("class: {}".format(injury_class))
        for i in range(5):

            mean = np.mean(data[:, :, i], axis=0)
            std = np.std(data[:, :, i], axis=0)

            
            plt.plot(mean, color=mean_col[i])
            plt.plot(mean+std, color=std_col[i])
            plt.plot(mean-std, color=std_col[i])

    plt.show()
    """

    """
    converter.enableGpu()
    conv_args = {
        "num_bins": 32,
         "range": (-1, 1),
         "dims": 3,
         "delay": 3,
         "metric": "euclidean"
    }
    #imgFilter = ImageFilter("avg", (5,5))
    #gaf_images = converter.convert(train, conversions="mtf", conv_args=conv_args, imgFilter=imgFilter)
            
    #gaf_images["affected"]["gasf"]
    #gaf_images["affected"]["gadf"]

    for injury_class in ["A", "C"]:
        indices = np.where(train["label"] == class_dict.get(injury_class))[0]
        #gasf = np.take(gaf_images["affected"]["gasf"], indices, axis=0)
        #gadf = np.take(gaf_images["affected"]["gadf"], indices, axis=0)
        mtf = np.take(gaf_images["affected"]["mtf"], indices, axis=0)

        instances = 0
        for img in mtf:
            if img[55, 59, 4] > 0.85 and img[75, 75, 1] > 0.5:
                #if img[35, 35, 3] > 0.7 or img[75, 75, 1] > 0.5:
                instances +=1
        print("{} has {} instances out of {}".format(injury_class, instances, mtf.shape[0]))
        mtf_mean = np.mean(mtf, axis=0)
        mtf_std = np.std(mtf, axis=0)

        #gadf_mean = np.mean(gadf, axis=0)
        #gadf_std = np.std(gadf, axis=0)

        plot_data = {}
        plot_data["affected"] = {}
        plot_data["affected"]["mtf"] = np.expand_dims(mtf_mean, axis=0)
        #plot_data["affected"]["gadf"] = np.expand_dims(gadf_std/abs(gadf_mean), axis=0)

        print("Plotting group {}".format(injury_class))
        #plotter.plot_image(plot_data, keys="affected", images=["mtf"], comp_order=fetcher.get_comp_order(), show=True, save=False, prefix=injury_class, folder="Std", vmin=0, vmax=1)


        #gadf = np.take(gaf_images["affected"]["gadf"], indices, axis=0)
    """

    """
    result = np.where(arr == 15)

        not_in_val_set = ~data["info"]["SUBJECT_ID"].isin(filter_data["info_val"]["SUBJECT_ID"])
    valid_indices = np.where(not_in_val_set.values)[0]

    result = {}
    for key in data.keys():
        if key != "info":
            result[key] = np.take(data[key], valid_indices, axis=0)
            print(result[key].shape)

    """
