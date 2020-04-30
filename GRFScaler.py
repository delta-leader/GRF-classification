import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

class GRFScaler(object):
    """Wrapper to use the scalers from 'sklearn.preprocessing' for GRF-data.

    One Scaler is used for each of the force components ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').
    Currently only StandardScaler and MinMaxScaler are supported.

    Parameters:
    scalertype : string, default="MinMax"
        Specifies the scaler type to use, has to be one of either "Minmax" or "Standard".
        Type specification is not case-sensitiv.
        "Standard": scales the data to mean=0 and standard deviation=1
        "MinMax": scales the data into the provided 'featureRange'

    featureRange : tuple (min, max), default=(0, 1)
        Desired range of the transformed data.
        Ignored for type "Standard".

    ----------
    Attributes:
    comp_list : list of shape (5)
        Exhaustive list of the used force components ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').

    types : list of shape(2)
        List of available scaler types ('standard', 'minmax')

    isFitted : bool
        Whether the scaler has been fitted on data or not.
    """

    def __init__(self, scalertype="MinMax", featureRange=(-1,1)):
        self.comp_list = ["f_v", "f_ap", "f_ml", "cop_ap", "cop_ml"] 
        self.types = ["standard", "minmax"]
        self.isFitted = False
        self.__set_scalertype(scalertype)
        self.__set_range(featureRange)
        self.__create_scaler()


    def is_fitted(self):
        """Reports whether the scaler has been fitted on data or not.

        Returns:
        isFitted : bool
        """

        return self.isFitted


    def fit(self, GRFData):
        """Compute the necessary values used for later scaling for each force component.

        Parameters:
        GRFData : dictionary containing the data for all force components.
        Input data in the following form:
        'f_v": num_samples x num_dimensions
        'f_ap": num_samples x num_dimensions 
        'f_ml": num_samples x num_dimensions 
        'cop_ap": num_samples x num_dimensions 
        'cop_ml": num_samples x num_dimensions 

        ----------
        Raises:
        ValueError : If GRFData is not a dictionary or does not contain values for one of the force components.
        """

        self.__is_valid_dict(GRFData)
        for component in GRFData.keys():
            self.scaler[component].fit(np.reshape(GRFData[component], (-1, 1)))
        self.isFitted = True


    def partial_fit(self, GRFData):
        """Online computation of mean and standard deviation for later scaling.
        Each component of the GRFData is processed as a single batch.
        This can be convenient to fit the scale to data from both legs.

        Parameters:
        GRFData : dictionary containing the data for all force components.
        Input data in the following form:
        'f_v": num_samples x num_dimensions
        'f_ap": num_samples x num_dimensions 
        'f_ml": num_samples x num_dimensions 
        'cop_ap": num_samples x num_dimensions 
        'cop_ml": num_samples x num_dimensions 

        ----------
        Raises:
        ValueError : If GRFData is not a dictionary or does not contain values for one of the force components.
        """

        self.__is_valid_dict(GRFData)
        for component in GRFData.keys():
            self.scaler[component].partial_fit(np.reshape(GRFData[component], (-1, 1)))
        self.isFitted = True


    def reset(self):
        """Reset internal data-dependent state of the scaler.
        __init__ parameters are not touched.
        """

        self.scaler = None
        self.isFitted = False
        self.__create_scaler()


    def transform(self, GRFData):
        """Scale the values of all force components using the previously fitted scaler.

        Parameters:
        GRFData : dictionary containing the data for all force components.
        Input data in the following form:
        'f_v": num_samples x num_dimensions
        'f_ap": num_samples x num_dimensions 
        'f_ml": num_samples x num_dimensions 
        'cop_ap": num_samples x num_dimensions 
        'cop_ml": num_samples x num_dimensions 

        ----------
        Returns:
        transformed_GRFData : dictionary containing the tranformed values for all force components.
        The output data has the same form as the input data.

        ----------
        Raises:
        NotFittedError : If the scaler has not been fitted to data prior to calling this function.

        ValueError: If GRFData is not a dictionary or does not contain values for one of the force components.
        """
        
        if not self.isFitted:
            raise NotFittedError("The scaler has n ot been fitted to data. Call 'fit()' before calling 'transform()'.")

        self.__is_valid_dict(GRFData)
        transformed_GRFData = {}
        for component in GRFData.keys():
            len_series = GRFData[component].shape[1]
            transformed_data = self.scaler[component].transform(np.reshape(GRFData[component], (-1, 1)))
            transformed_GRFData[component] = np.reshape(transformed_data, (-1, len_series))

        return transformed_GRFData


    def get_type(self):
        """Returns the type of the scaler.

        Returns:
        scalertype : String
        """

        return self.scalertype


    def get_featureRange(self):
        """Returns the feature range of the scaler.

        Returns:
        featureRange : tuple of form (min, max)
        """

        return self.featureRange


    def __set_scalertype(self, scalertype):
        """Sets the type of the scaler to the provided value.
        
        Parameters:
        scalertype : String
        The desired type of the scaler.

        ----------
        Raises:
        ValueError : if scalertype is neither "standard" nor "minmax" (not case-sensitive)
        """

        if not scalertype.lower() in self.types:
            raise ValueError("Scalertype '{}' not available. Use 'Standard' or 'MinMax'.".format(scalertype))
        else:
            self.scalertype = scalertype.lower()


    def __set_range(self, featureRange):
        """Sets the feature-range of the scaler to the desired value.

        Parameters:
        featureRange : tuple (min, max)
        The desired feature-range of the scaler.

        ----------
        Raises:
        ValueError : if the length of the provided tuple is not 2 or min >= max.
        """

        if len(featureRange) != 2:
            raise ValueError("FeatureRange expects a tuple of size 2 but received length {}.".format(len(featureRange)))
        if featureRange[0] >= featureRange[1]:
            raise ValueError("FeatureRange is expected to be (min, max) but received values where min >= max.")
        self.featureRange = featureRange


    def __create_scaler(self):
        """Creates a new scaler for each force component."""
        
        self.scaler = {}
        for component in self.comp_list:
            self.scaler[component] = self.__create_scaler_type()


    def __create_scaler_type(self):
        """Creates a new scaler according to the provided specifications."""

        if self.scalertype == "standard":
            return StandardScaler()
        if self.scalertype == "minmax":
            return MinMaxScaler(feature_range=self.featureRange)
        assert True, "An error occured when creating a scaler of type '{}'".format(self.scalertype)


    def __is_valid_dict(self, GRFData):
        """Check whether the GRFData is a valid dictionary to be used with the scaler.
        A valid dictionary contains entries for all force components ('f_v", 'f_ap', 'f_ml', 'cop_ap', 'cop_ml').

        Parameters:
        GRFData : dictionary 
        The data to be verified.
        ----------
        Raises:
        ValueError : if the received parameter is not a dictionary, or does not contain exactly the same force components as 'comp_list'.
        """

        if type(GRFData) is not dict:
            raise ValueError("Expected GRFData to be of type '{}', but received type '{}'.".format(type(dict), type(GRFData)))

        if len(self.comp_list) != len(GRFData):
            raise ValueError("GRFData contains {} entries, but expected {}.".format(len(GRFData), len(self.comp_list)))

        for component in self.comp_list:
            if component not in GRFData.keys():
                raise ValueError("Component '{}' not found in GRFData.".format(component))
