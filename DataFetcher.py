import os
import random
import math
import numpy as np
import pandas as pd
from errno import ENOENT
from GRFScaler import GRFScaler

# testing purposes only
# import matplotlib.pyplot as plt

class DataFetcher(object):
    """Reads the GRF data from the provided files and takes care of the preprocessing steps.

    Only the unmodified original files are accepted and all files need to exist within the same directory.
    This class is able to use both the normalized and raw versions of the data.
    Preprocessing can include any of the following steps: 
    - selection of the relevant data (characteristics of the data)
    - resampling (reducing/increasing the amount of timesteps)
    - scaling (using the provided GRFScaler)
    - averaging (across multiple trials in the same session)
    - concatenating of all force components

    Parameters:
    filpath : string
        Provides the path to the directory where the files are stored.

    ----------
    Attributes:
    filelist : list of shape (5)
        Contains the prefix to the filenames for all 5 force components.

    metadata : string
        Name of the file containing the metadata information.

    filepath : string
        The path to the directory where the files are stored.

    class_dict : dictionary of class labels
        Contains the mapping of class labels to integers.

    comp_order : list of shape (5)
        Exhaustive list of the used force components ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        Defines the order of the force components in the output tensor (= ordering within the first axis),
        or the order of the components when concatenated.

    randSeed : int
        The seed for the random number generator used to determine the affected side in ambigous cases (e.g. healthy control)

    randSeedVal : int
        The seed for the random number generator used to select the validation set.

    ----------
    Raises:
    IOError : If one of the necessary files can not be found in the specified directory.
    """
    
    def __init__(self, filepath):
        self.filelist = ["GRF_COP_AP_", "GRF_COP_ML_", "GRF_F_AP_", "GRF_F_ML_", "GRF_F_V_"] 
        self.metadata = "GRF_metadata.csv"
        if len(filepath) > 0:
            if filepath[-1] != "/":
                filepath += "/"
        self.__test_filepath(filepath)
        self.filepath = filepath
        self.class_dict = {"HC":0, "H":1, "K":2, "A":3, "C":4}
        self.comp_order = ["f_v", "f_ap", "f_ml", "cop_ap", "cop_ml"] 
        self.randSeed = 42
        self.randSeedVal = 11

    
    def get_class_dict(self):
        """Returns the dictionary used to convert the class-lables into integer values

        Returns:
        class_dict : dict
            Dictionary containing the encoding of the class-labels.
        """

        return self.class_dict


    def set_randSeed(self, seed):
        """Sets the seed for the random number generator used to determine the affected side in ambigous cases (e.g. healthy control)

        Parameters:
        seed: int
            Seed for the random number generator.

        ----------
        Raises:
        TypeError: If the parameter passed is not an integer value.
        """

        if type(seed) is not int:
            raise TypeError("Seed is not an integer.")
        else:
            self.randSeed = seed


    def set_randSeedVal(self, seed):
        """Sets the seed for the random number generator used to determine the validation set.

        Parameters:
        seed: int
            Seed for the random number generator.

        ----------
        Raises:
        TypeError: If the parameter passed is not an integer value.
        """

        if type(seed) is not int:
            raise TypeError("Seed is not an integer.")
        else:
            self.randSeedVal = seed


    def get_comp_order(self):
        """Returns a list of all force components.
        The order in the list corresponds to the order of the force components in the final output.
        I.e. the first element is orderd first, then the second and so on (either concatenated or along the first axis).

        Returns:
        comp_order : list of shape (5)
        The component order used in the final output.
        """

        return self.comp_order


    def set_comp_order(self, comp_order):
        """Sets the component order used for the force components in the final output.
        I.e. the first element is orderd first, then the second and so on (either concatenated or along the first axis).

        Parameters:
        comp_order : list of shape (5)
            The component order that should be used. Must include 'f_v', 'f_ap', 'f_ml', 'cop_ap' and 'cop_ml'.

        ----------
        Raises:
        TypeError: If the provided 'comp_order' is not a list.

        ValueError : If one of the force component is not included in the new list or if the same component appears more than once.
        """

        if type(comp_order) is not list:
            raise TypeError("The provided component-order is not a list.")

        if len(comp_order) != len(self.comp_order):
            raise ValueError("Length of the new component order does not match the old one. {} vs {}". format(len(comp_order), len(self.comp_order)))
        
        for component in self.comp_order:
            if component not in comp_order:
                raise ValueError("{} component is not provided in the new component order".format(component))

        if len(list(dict.fromkeys(comp_order))) != len(self.comp_order):
            raise ValueError("Duplicate values were provided in component order")

        # if none of the above is True, accept new component order
        self.comp_order = comp_order


    def fetch_data(self, raw=False, onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=None, concat=False, val_setp=0, include_info=False, clip=False):
        """Reads and preprocesses all 5 force components for the specified dataset AND the test set.
        Both the specified dataset and the test set are processed in the same manner, except that there will be no validation split created for the TEST set.

        Parameters:
        raw : bool, default=False
            If True the files containing the raw data are used instead of the normalized ones.

        OnlyInital : bool, default=False
            If True, only inital measurements (excluding control measurements and readmissions) are returned.

        dropOrthopedics : string, default="None"
            Patients using orthopedic aids are ommitted according to this specification. 
            Possible values are:
            "None":     No patients are omitted.
            "Verified": All patients using orthpedic aids are omitted.
            "All":      Only patients using no orthopedic aids are returned (i.e. NaNs are omitted).

        dropBothSidesAffected : bool, default=False
            If True, paritipants with injuries in both legs are excluded (i.e. only returns measurements where either one or no side is affected).

        dataset : string, default="TRAIN_BALANCED"
            Only measurements included in the specified dataset AND the test set are returned.
            Possible values are "TRAIN_BALANCED" and "TRAIN".

        stepsize : int, double, default=1
            Has to be int if working with processed data. Specifies the interval at which to sample the data.
            If stepsize = 1 all datapoints are used, if stepsize = 2, every second datapoint is used and so on.
            For raw data the number of samples is calculated at a basis of 100.
            The number or resamples = 100/stepsize rounded down.

        averageTrials : bool, default=True
            If True all trial recorded within the same session are reduced to 1 averaged measurement.
            Otherwise no modifications are made.

        scaler : GRFScaler, default=None
            If a scaler is provided, it is used to normalize the data.

        concat : bool, default=False
            If True, the force components are concatenated according to the order specified in 'comp_order'.
            Concatenation modifies the original data in order to provide a continuous signal.

        val_setp : float, default=0
            Contains the percentage of samples to include in the validation set.
            This parameter is ignored for the TEST-set.

        include_info : bool, default=False
            If True, additional information (e.g. SUBJECT_ID, SESSION_ID) is exported along with the data.

        clip : bool, default=False
            If True and scaler is of type 'MinMax', the dataset will be clipped to the range of the scaler.
            This can be desirable in case of rounding errors, or if the scaler was fitted on different data.

        ----------
        Returns:
        train : dictionary containing at least the keys 'label', 'affected' and 'non_affected'.
            'label': numpy array of integers containing the class labels (according to 'class_dict').
            'affected': numpy array of float32 either num_samples x time_steps x 5 (last dimension are the force components) or num_samples x timesteps*5 (if concate=True).
            'non-affected': same as above but contains the data for the unaffected leg.
            If 'include_info' == True:
                'info' : pandas dataframe containing SUBJECT_ID, SESSION_ID and TRIAL_ID (if available) and all metadata information.

            If 'val_setp' > 0:
                'affected_val' numpy array of float32 same format as 'affected' but first dimension is num_samples * val_setp (contains the validation set).
                'non_affected_val' same as above but contains the validation set for the unaffected leg.
                'label_val': numpy array of integers containing the class labels for the validation set.
                If 'include_fino' == True:
                    'info_val ' : pandas dataframe containing SUBJECT_ID, SESSION_ID and TRIAL_ID (if available) for the validation set.
            
            Contains only the data from the specified set.

        test : dictinary, same format as above, but never contains 'affected_val' nor 'non_affected_val'.
            Contains only the data from the test set.
        
        ----------
        Raises:
        ValueError: If 'dataset'  is neither 'TRAIN_BALANCED' nor 'TRAIN'.
        """

        if dataset not in ["TRAIN", "TRAIN_BALANCED"]:
            raise ValueError("Dataset {} does not exist. Please use one of 'TRAIN'/'TRAIN_BALANCED'.".format(dataset))

        train = self.fetch_set(raw, onlyInitial, dropOrthopedics, dropBothSidesAffected, dataset, stepsize, averageTrials, scaler, concat, val_setp, include_info, clip)
        test = self.fetch_set(raw, onlyInitial, dropOrthopedics, dropBothSidesAffected, "TEST", stepsize, averageTrials, scaler, concat, val_setp=None, include_info=include_info, clip=clip)

        return train, test


    def fetch_set(self, raw=False, onlyInitial=False, dropOrthopedics="None", dropBothSidesAffected=False, dataset="TRAIN_BALANCED", stepsize=1, averageTrials=True, scaler=None, concat=False, val_setp=0, include_info=False, clip=False):
        """Reads and preprocesses all 5 force components for the specified dataset.

        Parameters:
        raw : bool, default=False
            If True the files containing the raw data are used instead of the normalized ones.

        OnlyInital : bool, default=False
            If True, only inital measurements (excluding control measurements and readmissions) are returned.

        dropOrthopedics : string, default="None"
            Patients using orthopedic aids are ommitted according to this specification. 
            Possible values are:
            "None":     No patients are omitted.
            "Verified": All patients using orthpedic aids are omitted.
            "All":      Only patients using no orthopedic aids are returned (i.e. NaNs are omitted).

        dropBothSidesAffected : bool, default=False
            If True, paritipants with injuries in both legs are excluded (i.e. only returns measurements where either one or no side is affected).

        dataset : string, default="TRAIN_BALANCED"
            Only measurements included in the specified dataset are returned.
            Possible values are "TRAIN_BALANCED", "TRAIN" and "TEST".

        stepsize : int, double, default=1
            Has to be int if working with processed data. Specifies the interval at which to sample the data.
            If stepsize = 1 all datapoints are used, if stepsize = 2, every second datapoint is used and so on.
            For raw data the number of samples is calculated at a basis of 100.
            The number or resamples = 100/stepsize rounded down.

        averageTrials : bool, default=True
            If True all trial recorded within the same session are reduced to 1 averaged measurement.
            Otherwise no modifications are made.

        scaler : GRFScaler, default=None
            If a scaler is provided, it is used to normalize the data.

        concat : bool, default=False
            If True, the force components are concatenated according to the order specified in 'comp_order'.
            Concatenation modifies the original data in order to provide a continuous signal.

        val_setp : float, default=0
            Contains the percentage of samples to include in the validation set.

        include_info : bool, default=False
            If True, additional information (e.g. SUBJECT_ID, SESSION_ID) is exported along with the data.

        clip : bool, default=False
            If True and scaler is of type 'MinMax', the dataset will be clipped to the range of the scaler.
            This can be desirable in case of rounding errors, or if the scaler was fitted on different data (e.g. the TRAIN-set, and is now used on the TEST-set).

        ----------
        Returns:
        data : dictionary containing at least the keys 'label', 'affected' and 'non_affected'.
            'label': numpy array of integers containing the class labels (according to 'class_dict').
            'affected': numpy array of float32 either num_samples x time_steps x 5 (last dimension are the force components) or num_samples x timesteps*5 (if concate=True).
            'non-affected': same as above but contains the data for the unaffected leg.
            If 'include_info' == True:
                'info' : pandas dataframe containing SUBJECT_ID, SESSION_ID and TRIAL_ID (if available) and all metadata information.

            If 'val_setp' > 0:
                'affected_val' numpy array of float32 same format as 'affected' but first dimension is num_samples * val_setp (contains the validation set).
                'non_affected_val' same as above but contains the validation set for the unaffected leg.
                'label_val': numpy array of integers containing the class labels for the validation set.
                If 'include_info' == True:
                    'info_val ' : pandas dataframe containing SUBJECT_ID, SESSION_ID and TRIAL_ID (if available) and all metadata information for the validation set.
        
        ----------
        Raises:
        ValueError: If either 'dataset' or 'dropOrthopedics' are not within the valid range of data.
        ValueError: If a scaler is neither an instance of GRFScaler nor None.
        ValueError: If clip=True and a scaler different with a different type than 'MinMax' is used (or no scaler is used at all).
        """

        if dataset not in ["TEST", "TRAIN", "TRAIN_BALANCED"]:
            raise ValueError("Dataset {} does not exist. Please use one of 'TEST'/'TRAIN'/'TRAIN_BALANCED'.".format(dataset))
        if dropOrthopedics not in ["None", "Verified", "All"]:
            raise ValueError("{} is not an option for dropOrthopedics. Please use one of 'None'/'Verified'/'All'.".format(dropOrthopedics))
        metadata = self.__fetch_metadata()

        if onlyInitial:
            metadata = _select_initial_measurements(metadata)
        if dropBothSidesAffected:
            metadata = _drop_both_sides_affected(metadata)
        metadata = _drop_orthopedics(metadata, dropOrthopedics)

        # keep information if it needs to be exported
        if not include_info:
            metadata = _trim_metadata(metadata, keepNormParams=raw)
        metadata = _select_dataset(metadata, dataset)

        left, right = self.__fetch_data(metadata, raw)

        left, right = [_sample(leg, stepsize, raw) for leg in (left, right)]
        if averageTrials:
            left, right = [_average_trials(leg) for leg in (left, right)]

        if scaler != None:
            if not isinstance(scaler, GRFScaler):
                raise ValueError("Scaler needs to be a GRFScaler or None.")
            if not scaler.is_fitted():
                _fit_scaler(scaler, (left, right))

        if clip:
            if scaler is None or scaler.get_type() != "minmax":
                raise ValueError("Clipping can only take affect if the scaler is of type 'MinMax'. Please change clip to 'False' or choose a different scaler.")

        if scaler != None:
            left, right = [_scale(scaler, leg) for leg in (left, right)]

        if concat:
            left, right = [self.__concat(leg) for leg in (left, right)]

            # testing purposes only - assert that the range is still valid
            """
            for leg in (left, right):
                _assert_scale(leg, scaler)
            """

        affected, non_affected = self.__arrange_data(left, right, metadata)

        dataRange = None
        if clip:
            dataRange=scaler.get_featureRange()
        data = self.__split_and_format(affected, non_affected, metadata, val_setp, include_info, dataRange)
       
        #TODO remove print
        print("Exported dataset with shape: {}".format(data["affected"].shape))
        if "affected_val" in data.keys():
           print("Validation-set shape: {}".format(data["affected_val"].shape))

        return data     
        

    def __test_filepath(self, filepath):
        """Checks if all necessary files exist at the provided directory.

        Parameters:
        filpath : string
            Provides the path to the directory where the files are stored.

        ----------
        Raises:
        IOError : If one of the necessary files can not be found in the specified directory.
        """

        # check metadata
        _check_file(filepath + self.metadata)

        # check files
        for filename in self.filelist:
            for info in ["PRO_", "RAW_"]:
                for ending in ["left.csv", "right.csv"]:
                    _check_file(filepath + filename + info + ending)
        return
                    

    def __fetch_metadata(self):
        """Reads the metadata CSV-file and returns its contents.
        Note that the HC group was measured with different walkings speeds in contrast to all other groups.
        Therefore data with predetermined walking speed (i.e. not self-selected) is removed in order to make the groups equal.

        Returns:
        metadata : DataFrame
            Containing all the metadata information.

        ----------
        Raises:
        AssertionError : if the file does not contain the expected amount of content.
        """

        metadata = pd.read_csv(self.filepath + self.metadata, header=0)
        assert metadata.shape[0] == 8972, "Metadata contains {} elements, but expected 8972.".format(metadata.shape[0])

        # Choose only "self-selected" (only applies to HC group)
        metadata = metadata[metadata["SPEED"] == 2]
        assert metadata.shape[0] == 8434, "Only {} elements with self-selected walking speed, but expected 8434.".format(metadata.shape[0])

        return metadata


    def __fetch_data(self, metadata, raw):
        """Fetches all GRF measurments for a specefied subset of the data.
        Reads and returns all 5 components for both legs.

        Parameters:
        metadata : DataFrame
            The subset to select from the data.

        raw : bool
            Whether to read the processed or the original (i.e. raw) dataset.

        ----------
        Returns:
        left, right : dicitionaries containing all force components ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
            Represent the right and left leg respectively, using the force components as keys to the actual data.
        """
        
        left = {}
        right = {}
        for filename in self.filelist:
            # extract the component name
            component_name = filename[filename.index("_")+1:-1].lower()

            if raw:
                filename += "RAW_"
            else:
                filename += "PRO_"
            # do not check length for raw data
            assertLength = not raw
            left[component_name] = _read_and_select(self.filepath + filename + "left.csv", metadata, assertLength)
            right[component_name] = _read_and_select(self.filepath + filename + "right.csv", metadata, assertLength)

        return left, right


    def __concat(self, data_dict):
        """Transforms the 5 measurements into a single continuous signal by concatenating.
        The order of the force components in the concatenation is given by 'comp_order'.
        Discontinuites are eliminated by adding the last value of a component to the whole next component.

        Parameters:
        data_dict : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
            Contains the data to resample.

        ----------
        Returns:
        data_dict : dictionary with a single key ('concat').
            Contains the concatenated continuous signal.
        """

        
        concat_series = data_dict[self.comp_order[0]]

        # eliminate discontinuities in the concatenated series
        for component in self.comp_order[1:]:
            end_values = concat_series.iloc[:, -1]
            data = _get_data_part(data_dict[component])
            carry = end_values - data.iloc[:, 0]
            concat_series = pd.concat([concat_series, data.add(carry, axis="index")], axis=1)

        # testing purposes only
        """
        # plt.plot(_get_data_part(concat_series).values[0])
        # plt.show()
        len_series = data_dict[self.comp_order[0]].shape[1]
        len_info = _get_info_part(data_dict[self.comp_order[0]]).shape[1]
        assert concat_series.shape[0] == data_dict[self.comp_order[0]].shape[0], "Amount of samples does not match after concatenating {} vs {}".format(concat_series.shape[0], data_dict[self.comp_order[0]].shape[0])
        assert concat_series.shape[1] == len_series*5-len_info*4, "Length does not match after concatenating {} vs {}".format(concat_series.shape[1], len_series*5-len_info*4)
        """

        return {"concat": concat_series}


    def __split_and_format(self, affected, non_affected, metadata, val_setp, include_info, dataRange):
        """Splits the provided data into values and labels.
        Values are formatted into single-precision numpy-arrays.
        Labels are transformed into integers (specified in 'class_dict').
        If the input signals are concatenated the output is of shape num_samples x len_concat_series
        Otherwise it is of shape num_samples x len_series x 5
        If 'val_set' is specified, the provided data is split into a training and validation set.

        Parameters:
        affected : dictionary containing all five force components, either concatenated or not.
            The data for the affected side.

        non_affected : dictionary containing all five force components, either concatenated or not.
            The data for the unaffected side.

        metadata : DataFrame
            Containing all the metadata information (e.g. the class label).

        val_setp : float
            Contains the percentage of samples to include in the validation set.

        include_info : bool
            Whether or not to include additional information in the returned dictionary.

        DataRange : tuple of type (min, max)
            Range of the data if clipping should be applied, None otherwise.

        ----------
        Returns:
        data : dictionary containing the keys 'label', 'affected' and 'non_affected' (plus 'affected_val' and 'non_affected_val' if 'val_set' is specified).
            Contains the labels (as integers) and the data for the affected/unaffected side (as float32).
            If 'include_info' is True, and 'info' (and 'info_val' if 'val_set' is specified) is added to the dictionary containing additional information
            such as SUBJECT_ID, SESSION_ID and TRIAL_ID (if available) plus all metadata information available.

        ----------
        Raises:
        ValueError : If the percentage for the validation set is not within 0-1.
        TypeError : If 'val_setp' is none of float/int/None.
        """

        val_set = None
        keys = affected.keys()

        if val_setp != None and val_setp != 0:
            if not (isinstance(val_setp, float) or isinstance(val_setp, int)):
                raise TypeError("Please specify 'val_setp' as a floating-point value.")
            if val_setp < 0 or val_setp > 1:
                raise ValueError("Please specify the validation set between 0 and 1 (Current: {}).".format(val_setp))
            val_set = self.__get_indices_of_val_set(affected[list(keys)[0]], val_setp)

        label_info = metadata[["SESSION_ID", "CLASS_LABEL",]].set_index("SESSION_ID")
        labels = affected[list(keys)[0]].join(label_info, on="SESSION_ID")["CLASS_LABEL"].map(self.class_dict)
        data = {"label": labels.values}

        affected_formatted = {}
        non_affected_formatted = {}

        if include_info:
            # just take the info from the first component because it is the same across all components
            info = _get_info_part(affected[list(affected.keys())[0]]).reset_index(drop=True)
            metadata_info = metadata.drop(columns=["SUBJECT_ID"]).set_index("SESSION_ID")
            data["info"] = info.join(metadata_info, on="SESSION_ID").reset_index(drop=True)

        for component in affected:
            affected_formatted[component] = _format_data(affected[component], dataRange)
            non_affected_formatted[component] = _format_data(non_affected[component], dataRange)

        if "concat" in keys:
            data["affected"] = affected_formatted["concat"]
            data["non_affected"] = non_affected_formatted["concat"]
        else:
            affected_formatted = np.asarray(list(affected_formatted.values()), dtype=np.float32)
            non_affected_formatted = np.asarray(list(non_affected_formatted.values()), dtype=np.float32)
            data["affected"] = np.moveaxis(affected_formatted, 0, -1)
            data["non_affected"] = np.moveaxis(non_affected_formatted, 0, -1)

        # Extract the validation set
        if val_set is not None:
            data["affected_val"] = np.take(data["affected"], val_set, axis=0)
            data["non_affected_val"] = np.take(data["non_affected"], val_set, axis=0)
            data["label_val"] = np.take(data["label"], val_set, axis=0)
            data["affected"] = np.delete(data["affected"], val_set, axis=0)
            data["non_affected"] = np.delete(data["non_affected"], val_set, axis=0)
            data["label"] = np.delete(data["label"], val_set, axis=0)

            if include_info:
                data["info_val"] = data["info"].take(val_set, axis=0)
                data["info"] = data["info"].drop(val_set)

            # testing purposes only - Verify that the train- and validation-set are mutally exclusive on SESSION_ID & SUBJECT_ID
            """
            for component in affected:
                affected[component] =  affected[component].reset_index(drop=True)
                val_test = affected[component].iloc[val_set]
                train_test = affected[component].iloc[~affected[component].index.isin(val_set)]
                assert not val_test["SESSION_ID"].isin(train_test["SESSION_ID"]).any(), "Something went wrong during the selection of the validation set."
                assert not val_test["SUBJECT_ID"].isin(train_test["SUBJECT_ID"]).any(), "Something went wrong during the selection of the validation set."
            """

        return data


    def __arrange_data(self, left_dict, right_dict, metadata):
        """Combines the data from both legs and seperates them in signals for the affected and unaffected side.
        If the affected side can not be determined unambiguously (e.g. both sides none are affected), the signal used
        for the affected side is choosen at random.

        Parameters:
        left_dict : dictionary containing all five force components, either concatenated or not.
            The data for the left leg.

        right_dict : dictionary containing all five force components, either concatenated or not.
            The data for the right leg.

        metadata : DataFrame
            Containing all the metadata information (e.g. the information about which side is affected).

        ----------
        Returns:
        affected : dictionary, same format as input.
            Contains all the data for the affected side (including the class label).
            The order of the components in the dictionary corresponds to the order set in 'comp_order'.

        non_affected : dictionary, same format as input.
            Contains all the data for the unaffected side (including the class label).
            The order of the components in the dictionary corresponds to the order set in 'comp_order'.
        """

        if "concat" in left_dict.keys():
            key_order = ["concat"]
        else:
            key_order = self.comp_order

        leftSide_affected = self.__determine_affected_side(left_dict[key_order[0]], metadata)
        rightSide_affected =  np.invert(leftSide_affected)
        
        affected = {}
        non_affected = {}
        for component in key_order:       
            affected[component] = left_dict[component][leftSide_affected]
            affected[component] = affected[component].append(right_dict[component][rightSide_affected], sort=False)
            non_affected[component] = right_dict[component][leftSide_affected]
            non_affected[component] = non_affected[component].append(left_dict[component][rightSide_affected], sort=False)

            # testing purposes only
            """
            assert affected[component].shape == non_affected[component].shape == left_dict[component].shape, "Length does not match after arranging the data."
            # assert again
            if "TRIAL_ID" in affected[component].columns:
                assert affected[component][["SESSION_ID", "TRIAL_ID"]].equals(non_affected[component][["SESSION_ID", "TRIAL_ID"]]), "The order of the data is not preserved."
                assert affected[list(affected.keys())[0]][["SESSION_ID", "TRIAL_ID"]].equals(non_affected[component][["SESSION_ID", "TRIAL_ID"]]), "The order of the data is not preserved across components."
            else:
                assert affected[component]["SESSION_ID"].equals(non_affected[component]["SESSION_ID"]), "The order of the data is not preserved."
                assert affected[list(affected.keys())[0]]["SESSION_ID"].equals(non_affected[component]["SESSION_ID"]), "The order of the data is not preserved across components."

            """

        return affected, non_affected

    
    def __determine_affected_side(self, data, metadata):
        """Determines the affected Side of the provided data. If the affected side can not be determined unambiguously
        (e.g. both sides are affected/non-affected), the affected side is choosen at random.

        Parameters:
        data: DataFrame
            Contains a sampling of the data for which the affected side needs to be determined. It does not matter which leg or force-component
            is choosen because the inital ordering (i.e. in the files) is the same.

        metadata: DataFrame
            Containing all the metadata information (e.g. the information about which side is affected).

        ----------
        Returns:
        leftSide_affected: ndarray
            Same length as as data, boolean numpy array containing one value for each sample (True if the left side is the affected side, False otherwise).
        """

        # this assures affected side is choosen per Session (i.e. all Trials have the same affected side).
        # no coherency is guaranteed across Sessions (i.e. the affected side for a health person might change across sessions)
        affected_info = metadata[["SESSION_ID", "AFFECTED_SIDE"]].set_index("SESSION_ID")
        random.seed(self.randSeed)

        def is_leftSide_affected(x):
            if x == 0:
                return True
            if x == 1:
                return False
            return not random.getrandbits(1)

        affected_info["LEFT_AFFECTED"] = affected_info["AFFECTED_SIDE"].apply(is_leftSide_affected)
        data = data.join(affected_info, on="SESSION_ID")

        return data["LEFT_AFFECTED"].values


    def __get_indices_of_val_set(self, dataset, val_setp):
        """Randomly samples the dataset until the number of samples included in the validation set is greater than total_samples * val_setp.
        If a person is determined to be part of the validation set, all corresponding samples have to be part of the validation set.
        Therefore it is only guaranteed that the validation set contains at least the specified number of samples.
        Samples are determined randomly using 'randSeedVal' as the seed for the random number generator.

        Parameters:
        dataset : DataFrame
            Containing the complete data of the set for which to determine the validation set.
            (I.e. only the data for one force component or the concatenated series).

        val_setp :  float
            The percentage of the total samples to be included in the validation set

        ----------
        Returns:
        indices : ndarray (int)
            Numpy array containing the indices of the samples to be indluded in the validation set.
        """

        # Guarantee at least val_setp samples
        num_val = math.ceil(dataset.shape[0]*val_setp)
        randGen = np.random.RandomState(seed=self.randSeedVal)
        val_set_ids = dataset.sample(n=1, frac=None, random_state=randGen, axis=0)["SUBJECT_ID"].values
        indices = np.where(dataset["SUBJECT_ID"].values == val_set_ids)

        while len(indices) < num_val:
            randSample = dataset.sample(n=1, frac=None, random_state=randGen, axis=0)["SUBJECT_ID"].values
            # re-draw in case the sample has been selected previously
            while np.isin(randSample, val_set_ids).any():
                randSample = dataset.sample(n=1, frac=None, random_state=randGen, axis=0)["SUBJECT_ID"].values
            indices = np.append(indices, np.where(dataset["SUBJECT_ID"].values == randSample))
            val_set_ids = np.append(val_set_ids, randSample)
        
        return indices





def _select_initial_measurements(metadata):
    """Selects and returns only the inital measurements from the dataset.

    Parameters:
    metadata : DataFrame
        Containing the metadata information from which to select.

    Returns:
        DataFrame
        Containing only inital measurements.
    """

    return metadata[metadata["SESSION_TYPE"] == 1]
        

def _drop_orthopedics(metadata, dropOrthopedics):
    """Selects and returns only data that was recorded without orthopedic equipment (e.g. orthopedic shoes or inlays).

    Parameters:
    metadata : DataFrame
        Containing the metadata information from which to select.

    dropOrthopedics : string
        Patients using orthopedic aids are ommitted according to this specification. 
        Possible values are:
        "None":     No patients are omitted.
        "Verified": All patients using orthpedic aids are omitted.
        "All":      Only patients using no orthopedic aids are returned (i.e. NaNs are omitted).

    ----------
    Returns:
        DataFrame
        Containing only measurments taken in accordance with the provided specification.
    """
    
    if dropOrthopedics == "Verified":
        return metadata[(metadata["SHOD_CONDITION"] < 2) & (metadata["ORTHOPEDIC_INSOLE"] != 1)]

    if dropOrthopedics == "All":
        return metadata[(metadata["SHOD_CONDITION"] < 2) & (metadata["ORTHOPEDIC_INSOLE"] < 1)]

    # if none of the above is True, don't change the data
    return metadata


def _drop_both_sides_affected(metadata):
    """Selects and returns only measurements where either one or no leg is injured.

    Parameters:
    metadata : DataFrame
        Containing the metadata information from which to select.

    Returns:
        DataFrame
        Containing only measurements where one or no leg is affectted.
    """

    return metadata[metadata["AFFECTED_SIDE"] != 2]


def _trim_metadata(metadata, keepNormParams):
    """Removes unnecessary columns from the Dataframe.

    Parameters:
    metadata : DataFrame
        Containing the metadata information to be reduced.

    keepNormParams : bool
        Whether to keep parameters that could be used for normalization (only valuable for raw data).

    Returns:
        DataFrame
        DataFrame containing only the necessary columns.
    """
    if keepNormParams:
        return metadata[["SUBJECT_ID", "SESSION_ID", "CLASS_LABEL", "BODY_WEIGHT", "BODY_MASS", "SHOE_SIZE", "AFFECTED_SIDE", "TRAIN", "TRAIN_BALANCED", "TEST"]]
    else:
        return metadata[["SUBJECT_ID", "SESSION_ID", "CLASS_LABEL", "AFFECTED_SIDE", "TRAIN", "TRAIN_BALANCED", "TEST"]]


def _select_dataset(data, dataset):
    """Selects only the data belonging to one of the predefined datasets.

    Parameters:
    data : DataFrama
        The dataset from which to select.
    dataset : string
        Has to be on of the predefined datasets (i.e. one from TEST, TRAIN, TRAIN_BALANCED).

    ----------
    Returns:
        DataFrame
        The data belonging to the specified set.
    """

    return data[data[dataset] == 1]


def _read_and_select(filename, metadata, assertLength):
    """Reads one of the files containing the actual GRF measurements and selects the appropriate subset.

    Parameters:
    filename : string
        The name of the file to read.

    metadata : DataFrame
        DataFrame containing the SESSION_IDs of the subset that should be extracted.

    checkLength : bool
        Whether or not to assert the length of the time series when reading.

    ----------
    Returns:
        DataFrame
        The data for a single file corresponding to the specified subset.
        The index of the new dataframe starts with 0.

    ----------
    Raises:
    AssertionError : if the file does not contain the expected number of samples
                     or the length of one sample != 104 and assertLength = True.
    """

    #TODO maybe change to float32?
    data = pd.read_csv(filename, header=0, dtype="float64")
    data["SESSION_ID"] = data["SESSION_ID"].astype("int64")
    data["SUBJECT_ID"] = data["SUBJECT_ID"].astype("int64")
    data["TRIAL_ID"] = data["TRIAL_ID"].astype("int64")

    # assure that the file contains the original (complete) data
    assert data.shape[0] == 75569, "{} contains {} elements, but expected 75569.".format(filename, data.shape[0])
    if assertLength:
        assert data.shape[1] == 104, "{} contains {} entries per row, but expected 104.".format(filename, data.shape[1])

    data = data[data["SESSION_ID"].isin(metadata["SESSION_ID"])]
    return data.reset_index(drop=True)


def _sample(data_dict, stepsize, raw):
    """Resamples the data to match the specified stepsize.

    Parameters:
    data_dict : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        Contains the data to resample.

    stepsize : int, double
        Has to be int if working with processed data. Specifies the interval at which to sample the data.
        If stepsize = 1 all datapoints are used, if stepsize = 2, every second datapoint is used and so on.
        For raw data the number of samples is calculated at a basis of 100.
        The number or resamples = 100/stepsize rounded down.

    raw : bool
        Speciefies whether to use the sampling method for raw or processed data.

    ----------
    Returns:
    sampled_dict : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        The resampled data, same format as the input.
        If stepsize=1 and raw=False, the original dicitonary is returned unchanged.

    ----------
    Raises:
    ValueError : If the stepsize is not within 1-100.

    TypeError : If the stepsize is not an integer (only for processed data).
    """ 

    if stepsize < 1 or stepsize > 100: 
        raise ValueError("Current stepsize is {}, but hast to be between 0-100.".format(stepsize))

    sampled_dict = {}
    if raw:
        num_samples = (int)(100/stepsize)
        for component in data_dict:
            info = _get_info_part(data_dict[component])
            data = _get_data_part(data_dict[component])
            data = np.apply_along_axis(func1d=_interp_with_Nans, axis=1, arr=data, num_samples=num_samples)

            # testing purposes only
            """
            assert np.isnan(data).any() == False, "There are NaNs remaining within the dataset after sampling."
            """

            data = pd.DataFrame(data, dtype=np.float32)
            sampled_dict[component] = pd.concat([info, data], axis=1)

            # testing purposes only
            """
            assert sampled_dict[component].shape[0] == info.shape[0], "{} contains {} samples, but expected {}.".format(component, sampled_dict[component].shape[0], info.shape[0])
            assert sampled_dict[component].shape[1] == num_samples+3, "{} contains {} entries per row, but expected {}.".format(component, sampled_dict[component].shape[1], num_samples+3)
            # fig, (ax1, ax2) = plt.subplots(2)
            # fig.suptitle('Comparison orginal (top) vs resampled')
            # ax1.plot(data_dict[component].iloc[0, 3:])
            # ax2.plot(sampled_dict[component].iloc[0, 3:])
            # plt.show()
            """

    else:
        if type(stepsize) is not int:
            raise TypeError("Stepsize is not an integer.")
    
        # processed data sampling simply skips steps
        if stepsize > 1:
            usecols = [0, 1, 2] + [x for x in range(3, 104, stepsize)]
            for component in data_dict:
                sampled_dict[component] = data_dict[component].iloc[:, usecols]
                # testing purposes only
                """
                assert sampled_dict[component].shape[1] == len(usecols), "{} contains {} entries per row, but expected {}.".format(component, sampled_dict[component].shape[1], len(usecols))
                """
        else:
            # return original dict
            return data_dict

    return sampled_dict


def _average_trials(data_dict):
    """Calculates the mean over all trials recorded within the same measuring session.

    Parameters:
    data_dict : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        Contains the data to be averaged (i.e. all trials for each session contained)

    ----------
    Returns:
    avg_dict : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        Contains the averaged data. Each sample has the same size as in the input, but the numbers of samples is reduced.
        Only unique SESSION_IDs remain after averaging.
    """

    avg_dict = {}
    for component in data_dict:
        avg_dict[component] = data_dict[component].drop(columns="TRIAL_ID").groupby(["SUBJECT_ID", "SESSION_ID"], as_index=False, sort=False).mean()

        # testing purposes only
        """
        assert avg_dict[component]["SESSION_ID"].is_unique, "There was an error when averaging the Trials, duplicate SESSION_IDs remain."
        """

    return avg_dict


def _format_data(data, dataRange):
    """Removes all columns from the DataFrame that do not represent GRF-measurements.
    Comverts the DataFrame into a 'float32' numpy array.

    Parameters:
    data : DataFrame
        The data to be converted.

    DataRange : tuple of type (min, max)
            Range of the data if clipping should be applied, None otherwise.

    ----------
    Returns:
        ndarray
        'float32' numpy array containing only data from the actual measurement.
    """

    # only existing columns are dropped
    data = data.drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID", "CLASS_LABEL", "AFFECTED_SIDE"], errors="ignore") 

    data = data.astype('float32').values
    if dataRange is not None:
        data = np.clip(data, dataRange[0], dataRange[1])
    
    return data


def _check_file(filename):
    """Verifies whether the specified file exists or not.

    Parameters:
    filename : String
        The full filename to be verified.

    ----------
    Raises:
    IOError : If the file does not exist.
    """
    
    if not os.path.exists(filename):
        raise IOError(ENOENT, "No such file or directory.", filename)


def _fit_scaler(scaler, data):
    """Fits the provided GRFScaler on the data of both legs.

    Parameters:
    scaler : GRFScaler
        The scaler to be used on the data.
    
    data : tuple of form (left, right)
        Each element in the tuple contains a dictionary with all force components for each leg.
    """

    for leg in data:
        scaler.partial_fit(_to_scaler_format(leg))


def _scale(scaler, leg):
    """Scales all force components using the provided scaler.

    Parameters:
    scaler : GRFScaler
        The scaler to be used on the data.
    
    leg : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        Contains the data to be scaled.
    
    ----------
    Returns:
    scaled_dict : dictionary, same format as the input.
        Contains the scaled data.
    """

    scaled_data = scaler.transform(_to_scaler_format(leg))
    scaled_leg = {}
    for component in leg:
        info = _get_info_part(leg[component])
        scaled_leg[component] = pd.concat([info, pd.DataFrame(scaled_data[component])], axis=1)
    
    return scaled_leg


def _to_scaler_format(data_dict):
    """Converts all force components to the format excpected by the GRFScaler.

    Parameters:
    data_dict : dictionary containing all five force components  ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
        Contains the data to be scaled.

    ----------
    Returns:
    formatted_data : dictionary, same format as the input.
        Contains the formatted data.
    """

    formatted_data = {}
    for component in data_dict:
        formatted_data[component] = _format_data(data_dict[component], None)
    return formatted_data
            
            
def _interp_with_Nans(series, num_samples):
    """Removes all NaN-values from the given series and then
    interpolates the series according to the specified sampling using numpy's 'interp()' function.

    Parameters:
    series : 1-D ndarray
        The series to be interpolated.

    num_samples: int
        The number of samples in the interpolated series.

    ----------
    Returns:
        ndarray
        The interpolated values, of shape(num_samples).
    """

    # remove NaNs
    series = series[~np.isnan(series)]
    sampling = range(0, series.shape[0])

    # cut is needed because arange sometimes includes the last value
    desired_sampling = np.arange(0, series.shape[0], series.shape[0]/num_samples)[:num_samples]
    test = np.interp(desired_sampling, sampling, series)
    
    return test


def _get_info_part(data):
    """Extracts only the columns containing meta-information (i.e. not actual measurements) from the data.
    This means 'SUBJECT_ID', 'SESSION_ID' and in case of non-averaged data also 'TRIAL_ID'.

    Parameters:
    data : DataFrame
        The data from which to extract the meta information.

    ----------
    Returns:
    info : DataFrame
        Containing only the columns 'SUBJECT_ID', 'SESSION_ID' and in case of non-averaged data also 'TRIAL_ID'.
    """

    if "TRIAL_ID" in data.columns:
        return data[["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]]
    else:
        return data[["SUBJECT_ID", "SESSION_ID"]]


def _get_data_part(data):
    """Extracts only the columns containing the GRF-data (i.e. the actual measurements) from the dataframe.
    This metadata-information like 'SUBJECT_ID', 'SESSION_ID' and 'TRIAL_ID' are removed.

    Parameters:
    data : DataFrame
        The data from which to extract the GRF-data.

    ----------
    Returns:
        DataFrame
        Containing only the actual GRF-mesurements without any additinal meta-information.
    """

    # only existing columns are dropped
    return data.drop(columns=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"], errors="ignore")   


def _assert_scale(concat_data, scaler):
    """Verifies whether the concatenated signal is still within the value range given by the MinMax-Scaler.
    Does nothing for Standard-Scaler.

    Parameters:
    concat_data : dictionary containing the concatenated signal ('concat').
        Contains the data for which the range should be checked.

    scaler : GRFScaler
        The scaler that has been used to normalize the data.

    ----------
    Raises:
    ValueError : If the provided data is not within the range of the MinMax-Scaler.
    """

    #do nothing if no MinMax-Scaler was specified
    if scaler == None or scaler.get_type != "minmax":
        return

    scaler_min, scaler_max = scaler.get_featureRange
    data = _get_data_part(concat_data['concat']).values
    
    data_min = np.min(data)
    if data_min < scaler_min:
        raise ValueError("Concatenation invalidated the range of the data {} < {}".format(data_min, scaler_min))

    data_max = np.max(data)
    if data_max > scaler_max:
        raise ValueError("Concatenation invalidated the range of the data {} > {}".format(data_min, scaler_min))





def set_valSet(data, filter_data, parse=None):
    """Incase that a pre-defined validation set should be used (for example if it has been extracted previously), this function deletes all samples from data that correspond
    to an entry in the validation-set of 'filter_data.' based on the 'SUBJECT_ID'.
    There are three modes of operations: 
    'parse' == None: The validation-of 'filter_data' should become the new validation-set of 'data'.
    'parse' == 'SESSION_ID': The new validation set is parsed from 'data' based on the 'SESSION_ID' of the entries in the validation-set of 'filter_data'.
                             This will result in a new validation set that is identical to the one of 'filter_data' but uses the same format as 'data'
    'parse' == 'SUBJECT_ID': The new validation set is parsed from 'data' based on the 'SUBJECT_ID' of the entries in the validation-set of 'filter_data'.
                             Similar to 'SESSION_ID' but includes all values that were removed from 'data' (i.e. if a subject had multiple sessions in 'data' but only one in 'filter_data', all sessions will be moved to the new validation set). This is likely to result in a larger validation set than the original.

    Parameters:
    data : dict
        Containing only the keys 'affected', 'non_affected', 'labels' and 'info'.
    
    filter_data : dict
        Containing at least the key 'info_val'.

    parse : string, default=None,
        Decides the mode of operation. If None, the validation set from 'filter_data' is used.
        Otherwise the validation-set is parsed from 'data' with 'parse' defining the key to identify similar entries.
        

    ----------
    Returns:
    result :  dict
        Same format as the input.
        Additionally all entries with a 'SUBJECT_ID' corresponding to any of the ones passed in the validation set of 'filter_data' are removed.
        The new validation-set is either the one of 'filter_data' or it is parsed from 'data' based on the entries of the validation-set in 'filter_data' 

    ----------
    Raises:
    TypeError : If 'data' or 'filter_data' are not dictionaries.
    ValueError : If the key 'info' does not exist in 'data'.
    ValueError : If the key 'info_val' does not exist in 'filter_data'
    ValueError : If a validation-set is already specified in 'data' (operation probably does not have the intended effect).
    ValueError : If 'data' used the non_affected side but there is no corresponding validation-set available in 'filter_data'.
    ValueError : If 'parse' is neither of None, 'SUBJECT_ID' or 'SESSION_ID'.
    """

    if type(data) is not dict:
        raise TypeError("'Data' is not a dictionary.")
    if type(filter_data) is not dict:
        raise TypeError("'Filter_data' is not a dictionary.")
    if "info" not in data.keys():
        raise ValueError ("'Info' is not available in 'data', filter cannot be applied because information is missing.")
    if "affected_val" in data.keys():
        raise ValueError("Validation-Set is already specified in 'data', operation will not have the intended effect.")
    if 'info_val' not in filter_data.keys():
        raise ValueError("'Info_val' is not available in 'filter_data', filter cannot be applied because information is missing.")
    if parse is not None:
        if parse not in ["SUBJECT_ID", "SESSION_ID"]:
            raise ValueError("The parameter 'parse' must be one of the following: None, 'SUBJECT_ID' or 'SESSION_ID'.")

    not_in_val_set = ~data["info"]["SUBJECT_ID"].isin(filter_data["info_val"]["SUBJECT_ID"])
    valid_indices = np.where(not_in_val_set.values)[0]

    result = {}
    result["info"] = data["info"][not_in_val_set]
    for key in data.keys():
        if key != "info":
            result[key] = np.take(data[key], valid_indices, axis=0)

    if parse is not None:
        new_val_set = data["info"][parse].isin(filter_data["info_val"][parse])
        val_set_indices = np.where(new_val_set.values)[0]
        
        result["affected_val"] = np.take(data["affected"], val_set_indices, axis=0)
        result["affected_val"] = np.take(data["affected"], val_set_indices, axis=0)
        result["label_val"] = np.take(data["label"], val_set_indices, axis=0)
        
        if _isNonAffectedUsed(data, filter_data):
            result["non_affected_val"] = np.take(data["non_affected"], val_set_indices, axis=0)
        
        result["info_val"] = data["info"][new_val_set]

    else:
        result["affected_val"] = filter_data["affected_val"]
        result["label_val"] = filter_data["label_val"]
        result["info_val"] = filter_data["info_val"]
        
        if _isNonAffectedUsed(data, filter_data):
            result["non_affected_val"] = filter_data["non_affected_val"]

    return result


def _isNonAffectedUsed(data, filter_data):
    """Verifies that if the data for the non_affected side is used in 'filter_data',
    it is also available in 'data'.

    ----------
    Parameters:
    data : dict
        Containing GRF-data
    
    filter_data : dict
        Containing GRF-data with a non-empty validation-set.

    ----------
    Returns:
    bool : False if the data for the non_affected side is not available in 'filter_data'.
           True if it is available in both 'data' and 'filter_data'.

    ----------
    Raises : ValueError : If 'data' used the non_affected side but there is no corresponding validation-set available in 'filter_data'.
    """

    if "non_affected" in data.keys():
        if "non_affected_val" not in filter_data.keys():
            raise ValueError("'Data' used information about the non-affected side, but 'filter-data' does not provide a validation-set for non-affected data.")
        else:
            return True
    else:
        return False



# only maintained for compability issues
def filter_out_val_set(data, filter_data):
    """In case that the validation-set of another data-set should be used, this function filters out all the entried from data that correspond to an entry in the validation-set of 'filter_data',
    based on the 'SUBJECT_ID'.

    Parameters:
    data : dict
        Containing only the keys 'affected', 'non_affected', 'labels' and 'info'.
    
    filter_data : dict
        Containing at least the key 'info_val'.

    ----------
    Returns:
    result :  dict
        Same format as the input but without the 'info' key.
        Additionally all entries with a 'SUBJECT_ID' corresponding to any of the ones passed in the validation set of 'filter_data' are removed as well.
        To ready the data-set for immediate use, the validation-set of 'filter_data' is added to 'data'.

    ----------
    Raises:
    TypeError : If 'data' or 'filter_data' are not dictionaries.
    ValueError : If the key 'info' does not exist in 'data'.
    ValueError : If the key 'info_val' does not exist in 'filter_data'
    ValueError : If a validation-set is already specified in 'data' (operation probably does not have the intended effect).
    ValueError : If 'data' used the non_affected side but there is no corresponding validation-set available in 'filter_data'.
    """

    if type(data) is not dict:
        raise TypeError("'Data' is not a dictionary.")
    if type(filter_data) is not dict:
        raise TypeError("'Filter_data' is not a dictionary.")
    if "info" not in data.keys():
        raise ValueError ("'Info' is not available in 'data', filter cannot be applied because information is missing.")
    if "affected_val" in data.keys():
        raise ValueError("Validation-Set is already specified in 'data', operation will not have the intended effect.")
    if 'info_val' not in filter_data.keys():
        raise ValueError("'Info_val' is not available in 'filter_data', filter cannot be applied because information is missing.")

    not_in_val_set = ~data["info"]["SUBJECT_ID"].isin(filter_data["info_val"]["SUBJECT_ID"])
    valid_indices = np.where(not_in_val_set.values)[0]

    result = {}
    for key in data.keys():
        if key != "info":
            result[key] = np.take(data[key], valid_indices, axis=0)
            print(result[key].shape)

    result["affected_val"] = filter_data["affected_val"]
    result["label_val"] = filter_data["label_val"]
    result["info_val"] = filter_data["info_val"]
    if "non_affected" in data.keys():
        if "non_affected_val" not in filter_data.keys():
            raise ValueError("'Data' used information about the non-affected side, but 'filter-data' does not provide a validation-set for non-affected data.")
        result["non_affected_val"] = filter_data["non_affected_val"]

    return result

#only maintained for compability issus
def filter_out_subjects(data, info, class_dict, injury_classes=["HC", "H", "K", "A", "C"]):
    """Filters out all data corresponding to any 'SUBJECT_ID' listed in 'info' from all datasets in data (i.e. 'affected', 'non_affected', 'label', 'affected_val', 'non_affected_val', 'label_val').
    Additionally provides the option to filter out only patients with selected injuries as specified by injury_classes.
    
    Parameters:
    data : dict
        Containing at least the keys 'affected', 'labels' and 'info'.
    
    filter_data : pd.DataFrame
        Containing the 'SUBJECT_ID' to remove. 

    class_dict: dictionary
        Containing the mapping of injury_classes within the data.

    injury_classes : list, default=["HC", "H", "K", "A", "C"]
        Containing the injury_classes contained in the final data_set

    ----------
    Returns:
    result :  dict
        Same format as the input but without the 'info' and 'info_val' key.
        Additionally all entries with a 'SUBJECT_ID' corresponding to any of the ones passed in the validation set of 'info' are removed as well.

    ----------
    Raises:
    TypeError : If 'data' is not a dictionary.
    TypeError : If 'info' is not a pd.DataFrame.
    TypeError : If 'injury_classes' is not a list.
    ValueError : If the key 'info' (of 'info_val' if a val-set is specified) does not exist in 'data'.
    ValueError : If the column 'SUBJECT_ID' does not exist in 'info'
    """

    if type(data) is not dict:
        raise TypeError("'Data' is not a dictionary.")
    if type(info) is not pd.DataFrame:
        raise TypeError("'info' is not a pd.Dataframe.")
    if type(injury_classes) is not list:
        raise TypeError("'injury_classes' is not a list.")
    if "info" not in data.keys():
        raise ValueError ("'Info' is not available in 'data', filter cannot be applied because information is missing.")
    if "affected_val" in data.keys() and "info_val" not in data.keys():
        raise ValueError("'Info_val' is not available in 'data', filter cannot be applied because information is missing.")
    if 'SUBJECT_ID' not in info.columns:
        raise ValueError("'SUBJECT_ID' is not available in 'info', filter cannot be applied because information is missing.")

    valid_data = ~data["info"]["SUBJECT_ID"].isin(info["SUBJECT_ID"])
    valid_indices = np.where(valid_data.values)[0]

    train_sets = ["affected", "label", "info"]
    val_sets = ["affected_val", "label_val", "info_val"]
    if "non_affected" in data.keys():
        train_sets += ["non_affected"]
        val_sets += ["non_affected_val"]
    
    result = {}
    for key in train_sets:
        result[key] = np.take(data[key], valid_indices, axis=0)

    final_classes =[]
    
    for injury_class in injury_classes:
        final_classes.append(class_dict.get(injury_class))

    new_dict ={}
    for key in injury_classes:
        new_dict[key] = final_classes.index(class_dict.get(key))
    
    indices = np.where(np.isin(result["label"], final_classes))[0]
    for key in train_sets:
        result[key] = np.take(result[key], indices, axis=0)

    if "affected_val" in data.keys():
        valid_data = ~data["info_val"]["SUBJECT_ID"].isin(info["SUBJECT_ID"])
        valid_indices = np.where(valid_data.values)[0]

        for key in val_sets:
            result[key] = np.take(data[key], valid_indices, axis=0)

        indices = np.where(np.isin(result["label_val"], final_classes))[0]
        for key in val_sets:
            result[key] = np.take(result[key], indices, axis=0)

    for key, value in new_dict.items():
        result["label"][result["label"] == class_dict.get(key)] = value
        result["label_val"][result["label_val"] == class_dict.get(key)] = value

    return result, new_dict



