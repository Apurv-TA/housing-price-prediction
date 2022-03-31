import numpy as np
from sklearn.base import BaseEstimator, 


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


def _custom_data_transform(df, cols2keep=None):
    """Transformation to drop some columns in the data

    Parameters
    ----------
        df - pd.DataFrame
        cols2keep - columns to keep in the dataframe
    """
    cols2keep = cols2keep or []
    if len(cols2keep):
        return (df
                .select_columns(cols2keep))
    else:
        return df


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Class to add some extra features to the dataframe
    The features created are 'rooms_per_household' and
    'population_per_household'
    Attributes
    ----------
    add_bedrooms_per_room : bool, optional
        Public attribute to decide whether to create a feature named
        'bedrooms_per_room' or not
    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """Method to fit the dataframe
        Parameters
        ----------
        X: pandas Dataframe
            Dataframe on which the operation is performed
        y: pd.Dataframe
        """

        return self

    def transform(self, X):
        """Method to transform the dataframe
        This method is used to transform the dataframe and add extra
        features to it. The features added are: 'rooms_per_household',
        'population_per_household' and 'bedrooms_per_room'
        Parameters
        ----------
        X: pd.DataFrame
            The dataframe on which the tranform operation will be performed
        Returns
        --------
        Updated dataframe
        """

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room
            ]
        return np.c_[X, rooms_per_household, population_per_household]