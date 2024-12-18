import pandas as pd
from sklearn.pipeline import Pipeline
import os
import pickle
from typing import List


def data_sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicate records from the dataset (LoanNumber)
    df = df.drop_duplicates(subset=["LoanNumber"]) 
    # 2. Filter out wrong BorrowerYearsInSchool records like > 1
    df = df[df["BorrowerYearsInSchool"] <= 1]
    # 3. Strip off white space from LoanPurpose and LoanType
    df["LoanPurpose"] = df["LoanPurpose"].str.strip()
    df["LoanType"] = df["LoanType"].str.strip()
    # 4. Reset index
    df = df.reset_index(drop=True)
    return df



def data_transformation(df: pd.DataFrame):
    # 1. Create an Approved column using ApprovalDate if not null else 0
    df['Approved'] = df['ApprovalDate'].notnull().astype(int)
    # 2. Recategorize LoanPurpose column like ['Refinance', 'Purchase'] -> ['Refinance', 'Purchase']
    df['LoanPurpose'] = df['LoanPurpose'].map({'Refinance': 'Refinance', 'Purchase': 'Purchase'})
    # 3. Calculate difference between current date and DateAdded in days and create a new column 'Diff'
    df['Diff'] = (pd.to_datetime('today') - pd.to_datetime(df['DateAdded'])).dt.days
    # 4. Update 'BorrowerOwnRent' column like ['Own', 'Rent'] -> ['Own', 'Rent']
    df['BorrowerOwnRent'] = df['BorrowerOwnRent'].map({'Own': 'Own', 'Rent': 'Rent'})
    return df

def filter_extreme_vals(df: pd.DataFrame):
    # 1. Filter out extreme outliers from the columns like ['CLTV', 'TotalLoanAmount', 'CreditScore', 'TotalIncome']
    for col in ['CLTV', 'TotalLoanAmount', 'CreditScore', 'TotalIncome']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def feature_selection(df: pd.DataFrame):
    # 1. Select relevant features for the model
    df = df[['CoBorrowerTotalMonthlyIncome', 'CoBorrowerAge', 'CoBorrowerYearsInSchool','BorrowerTotalMonthlyIncome', 'BorrowerAge', 
             'DTI', 'CLTV', 'CreditScore', 'TotalLoanAmount', 'LoanApproved', 'LeadSourceGroup','Group','LoanPurpose']]
    return df

def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    # 1. data_sanity_check()
    df = data_sanity_check(df)
    # 2. data_transformation()
    df = data_transformation(df)
    # 3. filter_extreme_vals()
    df = filter_extreme_vals(df)
    # 4. feature_selection()
    df = feature_selection(df)
    return df



def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    # 1. Load the dataset
    df = _load_raw_dataset(file_name=file_name)
    # 2. apply pre_pipeline_preparation
    df = pre_pipeline_preparation(df=df)
    return df


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    model_filename = "trained_model.pkl"
    output_dir = "trained_models"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, model_filename), 'wb') as f:
        pickle.dump(pipeline_to_persist, f)
    print(f"Pipeline saved at {os.path.join(output_dir, model_filename)}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    model_filename = os.path.join("trained_models", file_name)
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Pipeline file {model_filename} not found.")
    
    with open(model_filename, 'rb') as f:
        return pickle.load(f)


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    model_dir = "trained_models"
    all_files = os.listdir(model_dir)
    
    for file in all_files:
        if file not in files_to_keep:
            file_path = os.path.join(model_dir, file)
            os.remove(file_path)
            print(f"Removed old pipeline: {file_path}")