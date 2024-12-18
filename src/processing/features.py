import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class MissValImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for handling missing values in numerical and categorical variables."""
    def __init__(self, num_vars: list, cat_vars: list, n_neighbors: int = 5):
        """
        Initialize the imputer for numerical and categorical variables.

        Parameters
        ----------
        num_vars: list
            List of numerical variable names.
        cat_vars: list
            List of categorical variable names.
        """
        self.num_vars = num_vars
        self.cat_vars = cat_vars
        self.n_neighbors = n_neighbors

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
        # KNN Imputer for numerical variables
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.knn_imputer.fit(X[self.num_vars])

        # Simple Imputer for categorical variables (using most frequent for simplicity)
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.cat_imputer.fit(X[self.cat_vars])
        
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
        # Impute numerical variables using KNNImputer
        X[self.num_vars] = self.knn_imputer.transform(X[self.num_vars])

        # Impute categorical variables using SimpleImputer (most frequent strategy)
        X[self.cat_vars] = self.cat_imputer.transform(X[self.cat_vars])

        return X


class OutlierHandler:
    def __init__(self, num_vars: list):
        """
        Initialize OutlierHandler with the given numerical features.

        Parameters
        ----------
        num_vars : list
            List of numerical feature names.
        """
        self.num_vars = num_vars


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
        # Identify outliers using IQR
        q1 = df[clmn].quantile(0.25)
        q3 = df[clmn].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[clmn] > lower_bound) & (df[clmn] < upper_bound)]


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
        pass

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
        for clmn in self.num_vars:
            df = self.identify_outliers(df, clmn)
        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_vars: list):
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
        self.categorical_cols = cat_vars
        self.ohe = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' to avoid dummy variable trap

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
        # Apply OneHotEncoding to the categorical columns
        encoded_cols = pd.DataFrame(self.ohe.transform(X[self.categorical_cols]), 
                                    columns=self.ohe.get_feature_names_out(self.categorical_cols))
        
        # Drop original categorical columns and concatenate the encoded columns
        X = X.drop(columns=self.categorical_cols)
        X = pd.concat([X, encoded_cols], axis=1)
        return X

# Custom StandardScalerWithExclusion to exclude specific column from scaling
class StandardScalerWithExclusion(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_column='LoanApproved'):
        self.exclude_column = exclude_column
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        columns_to_scale = X.select_dtypes(include=['number']).drop(columns=[self.exclude_column], errors='ignore').columns
        self.scaler.fit(X[columns_to_scale])
        return self

    def transform(self, X):
        columns_to_scale = X.select_dtypes(include=['number']).drop(columns=[self.exclude_column], errors='ignore').columns
        X[columns_to_scale] = self.scaler.transform(X[columns_to_scale])
        return X

class FeatureTransformer:
    def __init__(self, num_vars: list, cat_vars: list, exclude_column='LoanApproved', n_neighbors=5):
        self.num_vars = num_vars
        self.cat_vars = cat_vars
        self.exclude_column = exclude_column
        self.n_neighbors = n_neighbors

    def create_pipeline(self):
        # Create the components for the numerical and categorical transformations
        miss_val_imputer = MissValImputer(self.num_vars, self.cat_vars, n_neighbors=self.n_neighbors)
        cat_encoder = CategoricalEncoder(self.cat_vars)
        outlier_handler = OutlierHandler(self.num_vars)
        scaler_with_exclusion = StandardScalerWithExclusion(exclude_column=self.exclude_column)

        # Numeric transformation pipeline (missing value imputation, outlier handling, scaling)
        num_transformer = Pipeline([
            ('miss_val_imputer', miss_val_imputer),
            ('outlier_handler', outlier_handler),
            ('scaler', scaler_with_exclusion)
        ])

        # Categorical transformation pipeline (missing value imputation, encoding, outlier handling, and scaling)
        cat_transformer = Pipeline([
            ('miss_val_imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', cat_encoder),
            ('outlier_handler', outlier_handler),
            ('scaler_with_exclusion', scaler_with_exclusion)
        ])

        # ColumnTransformer to apply the respective transformations to numerical and categorical columns
        preprocessor = ColumnTransformer([
            ('num', num_transformer, self.num_vars),
            ('cat', cat_transformer, self.cat_vars)
        ])

        return preprocessor