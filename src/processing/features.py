import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class MissValImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for handling missing values in numerical and categorical variables."""
    def __init__(self, numerical_cols, categorical_cols, n_neighbors: int):
        """
        Initialize the imputer for numerical and categorical variables.

        Parameters
        ----------
        num_vars: list
            List of numerical variable names.
        cat_vars: list
            List of categorical variable names.
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.n_neighbors = n_neighbors

        self.numerical_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer to the data.

        This function prepares the KNNImputer for numerical variables 
        and SimpleImputer for categorical variables.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values.
        y : pd.Series, optional
            Target variable (not used here).

        Returns
        -------
        self
            Returns the instance itself.
        """
        self.numerical_imputer.fit(X[self.numerical_cols])
        self.categorical_imputer.fit(X[self.categorical_cols])
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame for both numerical and categorical variables.

        The numerical variables will be imputed using KNNImputer and the categorical variables 
        will be imputed using SimpleImputer with the most frequent strategy.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset with missing values.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with imputed missing values.
        """
        # Impute the missing values
        X[self.numerical_cols] = self.numerical_imputer.transform(X[self.numerical_cols])
        X[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method='median'):
        """
        Initialize OutlierHandler with the given numerical features.

        Parameters
        ----------
        num_vars : list
            List of numerical feature names.
        """
        self.method = method

    def identify_outliers(self, df, clmn):
        """
        Identify outliers in a specified column of a DataFrame using the IQR method.
        This function calculates the first quartile (Q1), third quartile (Q3),
        and interquartile range (IQR) of the specified column. It then determines
        the lower and upper whiskers for outlier detection. The outlier information
        is stored in the `outlier_info` attribute for the given column.
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data.
        clmn : str
            The name of the column in which to identify outliers.
        """
        pass


    def treat_outliers(self, df, clmn, method='mean'):
        """
        Treat outliers in a specified column of a DataFrame by replacing them with a calculated value.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data.
        clmn : str
            The name of the column in which to treat outliers.
        method : str, optional
            The method to use for treating outliers, by default 'mean'.
            Currently, only 'mean' is supported, which replaces outliers
            with the mean value of the column.

        Notes
        -----
        Outliers are defined based on the lower and upper whiskers stored
        in `outlier_info` for the specified column. Values outside these
        whiskers are considered outliers and will be replaced.
        """
        pass

    def fit(self, X, y=None):
        # Identify and store outlier information during fit
        """
        Fit the OutlierHandler by identifying outliers in the numerical features.

        This method identifies and stores outlier information for each numerical
        feature in the dataset using the IQR method. The outlier information
        is used later during the transform step to treat outliers.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing the data with numerical features.
        y : None
            Not used, present here for compatibility with sklearn pipelines.

        Returns
        -------
        self
            Returns the instance with stored outlier information.
        """
        return self

    def transform(self, X):
        # Treat outliers during transform
        """
        Treat outliers in the given DataFrame by replacing them with a calculated value.

        Outliers are defined based on the lower and upper whiskers stored
        in `outlier_info` for each numerical feature. Values outside these
        whiskers are considered outliers and will be replaced.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing the data with numerical features.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with outliers treated.
        """
        X_transformed = X.copy()
        
        # Detect and impute outliers only for numeric columns
        for col in X_transformed.select_dtypes(include=[np.number]).columns:
            Q1 = X_transformed[col].quantile(0.25)
            Q3 = X_transformed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = X_transformed[(X_transformed[col] < lower_bound) | (X_transformed[col] > upper_bound)].index
            
            # Impute the outliers with the specified method
            if len(outliers) > 0:
                if self.method == 'median':
                    impute_value = X_transformed[col].median()
                elif self.method == 'mean':
                    impute_value = X_transformed[col].mean()
                else:
                    raise ValueError("Invalid method. Choose 'median' or 'mean'.")
                
                X_transformed.loc[outliers, col] = impute_value
        
        return X_transformed
        


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        """
        Initialize the CategoricalEncoder with the given categorical feature names.

        Parameters
        ----------
        cat_vars : list
            List of categorical feature names.

        Attributes
        ----------
        cat_vars : list
            List of categorical feature names.
        encoder : None
            The OneHotEncoder object used for encoding categorical features.
        """
        self.categorical_cols = categorical_cols
        self.ohe = OneHotEncoder(drop='first', sparse_output=False)

    def fit(self, X, y=None):
        """
        Fit the OneHotEncoder to the categorical features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing the data with categorical features.
        y : None, optional
            Not used, present here for compatibility with sklearn pipelines.

        Returns
        -------
        self
            Returns the instance with the fitted OneHotEncoder.
        """
        self.ohe.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by one-hot encoding the categorical features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing the data with categorical features.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with the categorical features encoded.
        """
        X[self.categorical_cols] = X[self.categorical_cols].fillna(0)
        encoded_cols = pd.DataFrame(self.ohe.transform(X[self.categorical_cols]), columns=self.ohe.get_feature_names_out(self.categorical_cols))
        X = X.drop(columns=self.categorical_cols)
        X = pd.concat([X, encoded_cols], axis=1)
        return X

# Custom StandardScalerWithExclusion to exclude specific column from scaling
class StandardScalerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_column:str = None):
        self.exclude_column = exclude_column
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        columns_to_scale = X.columns
        self.scaler.fit(X[columns_to_scale])
        
        return self

    def transform(self, X):
        columns_to_scale = X.columns
        X[columns_to_scale] = self.scaler.transform(X[columns_to_scale])
        
        return X
