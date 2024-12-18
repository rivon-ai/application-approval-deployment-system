import pandas as pd
from sklearn.pipeline import Pipeline

def data_sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicate records from the dataset (LoanNumber) 
    # 2. Filter out wrong BorrowerYearsInSchool records like > 1
    # 3. Strip off white space from LoanPurpose and LoanType
    # 4. Reset index
    pass



def data_transformation(df: pd.DataFrame):
    # 1. Create an Approved column using ApprovalDate if not null else 0
    # 2. Recategorize LoanPurpose column like ['Refinance', 'Purchase'] -> ['Refinance', 'Purchase']
    # 3. Calculate difference between current date and DateAdded in days and create a new column 'Diff'
    # 4. Update 'BorrowerOwnRent' column like ['Own', 'Rent'] -> ['Own', 'Rent']
    pass

def filter_extreme_vals(df: pd.DataFrame):
    # 1. Filter out extreme outliers from the columns like ['CLTV', 'TotalLoanAmount', 'CreditScore', 'TotalIncome']
    pass


def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    # 1. data_sanity_check()
    # 2. data_transformation()
    # 3. filter_extreme_vals()
    pass



def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    pass


def load_dataset(*, file_name: str) -> pd.DataFrame:
    # 1. Load the dataset
    # 2. apply pre_pipeline_preparation
    pass


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    pass


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    pass


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    pass