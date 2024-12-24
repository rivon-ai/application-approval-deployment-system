import pandas as pd
from sklearn.pipeline import Pipeline
from config.core import ConfigLoader
import os
import joblib
from typing import List


# Loading Configuration from config.yml
config_path = "config.yml"
path = ConfigLoader(config_path)

def data_sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicate records from the dataset (LoanNumber)
    df = df.drop_duplicates(subset=['LoanNumber'], keep='first') 
    # 2. Filter out wrong BorrowerYearsInSchool records like > 1
    # 3. Strip off white space from LoanPurpose and LoanType
    df["LoanPurpose"] = df["LoanPurpose"].str.strip()
    df["LoanType"] = df["LoanType"].str.strip()
    # 4. Reset index
    df = df.reset_index(drop=True)
    return df



def data_transformation(df: pd.DataFrame):
    # 1. Create an Approved column using ApprovalDate if not null else 0
    df['Approved'] = df['ApprovalDate'].apply(lambda x: 1 if pd.notnull(x) else 0)
    df['Approved'] = df['Approved'].astype(int)
    # 2. Recategorize LoanPurpose column like ['Refinance', 'Purchase'] -> ['Refinance', 'Purchase']
    # 3. Calculate difference between current date and DateAdded in days and create a new column 'Diff'
    # 4. Update 'BorrowerOwnRent' column like ['Own', 'Rent'] -> ['Own', 'Rent']
    return df

def filter_extreme_vals(df: pd.DataFrame):
    # 1. Filter out extreme outliers from the columns like ['CLTV', 'TotalLoanAmount', 'CreditScore', 'TotalIncome']
    for col in ['CLTV', 'TotalLoanAmount', 'CreditScore']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    # 1. data_sanity_check()
    df = data_sanity_check(df)
    # 2. data_transformation()
    df = data_transformation(df)
    # 3. filter_extreme_vals()
    # df = filter_extreme_vals(df)
    return df



def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    # 1. Load the dataset
    df = _load_raw_dataset(file_name=file_name)
    # 2. apply pre_pipeline_preparation
    df = pre_pipeline_preparation(df=df)
    return df


def save_pipeline(*, pipeline_to_persist: Pipeline, model_to_persist: object) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    try:
        # Define file paths from config
        pipeline_file = os.path.join(path.config.app_config.save_model_directory, path.config.mode_config.pipeline_name)
        model_file = os.path.join(path.config.app_config.save_model_directory, path.config.mode_config.model_name)
        
        # Save the entire pipeline
        joblib.dump(pipeline_to_persist, pipeline_file)
        print(f"Pipeline saved to {pipeline_file}")
        
        # Save the model separately
        joblib.dump(model_to_persist, model_file)
        print(f"Model saved to {model_file}")
        
    except Exception as e:
        print(f"Error occurred while saving the pipeline and model: {e}")


def load_pipeline(*, pipeline_file: str, model_file: str) -> Pipeline:
    """Load a persisted pipeline."""
    try:
        # Load the pipeline (preprocessing steps)
        pipeline = joblib.load(pipeline_file)
        print(f"Pipeline loaded from {pipeline_file}")
        
        # Load the model (final estimator)
        model = joblib.load(model_file)
        print(f"Model loaded from {model_file}")
        
        return pipeline, model
    except Exception as e:
        print(f"Error occurred while loading the pipeline and model: {e}")
        return None, None


def remove_old_pipelines(*, files_to_keep: List[str], directory: str) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    try:
        # Get all files in the specified directory
        all_files = os.listdir(directory)

        # Loop through all files in the directory
        for file in all_files:
            file_path = os.path.join(directory, file)
            
            # Check if the file is not in the list of files to keep
            if file not in files_to_keep and (file.endswith(path.config.mode_config.file_extension)):  # Assuming .pkl or .joblib extension
                # Remove the old pipeline or model file
                os.remove(file_path)
                print(f"Removed old file: {file}")
        
    except Exception as e:
        print(f"Error occurred while removing old pipelines: {e}")

# Function for user input (same as before)
def get_user_input():
    BorrowerTotalMonthlyIncome = float(input("Enter Borrower Total Monthly Income: "))
    BorrowerAge = float(input("Enter Borrower Age: "))
    CoBorrowerTotalMonthlyIncome = float(input("Enter Co-Borrower Total Monthly Income: "))
    CoBorrowerAge = float(input("Enter Co-Borrower Age: "))
    CoBorrowerYearsInSchool = float(input("Enter Co-Borrower Years In School: "))
    DTI = float(input("Enter DTI (Debt-to-Income Ratio Range ): "))
    CLTV = float(input("Enter CLTV (Loan-to-Value Ratio): "))
    CreditScore = float(input("Enter Credit Score: "))
    TotalLoanAmount = float(input("Enter Total Loan Amount: "))
    LeadSourceGroup = input("Enter Lead Source Group (TV, Self Sourced, Internet, Radio, Referral, Repeat Client, Direct Mail, 3rd Party, Social Media): ")
    Group = input("Enter Group (Admin, Loan Coordinator, Refinance Team - #number): ")
    LoanPurpose = input("Enter Loan Purpose (Purchase, VA IRRRL, Refinance Cash-out, FHA Streamlined Refinance): ")
    return {
        "BorrowerTotalMonthlyIncome": BorrowerTotalMonthlyIncome,
        "BorrowerAge": BorrowerAge,
        "CoBorrowerTotalMonthlyIncome": CoBorrowerTotalMonthlyIncome,
        "CoBorrowerAge": CoBorrowerAge,
        "CoBorrowerYearsInSchool": CoBorrowerYearsInSchool,
        "DTI": DTI,
        "CLTV": CLTV,
        "CreditScore": CreditScore,
        "TotalLoanAmount": TotalLoanAmount,
        "LeadSourceGroup": LeadSourceGroup,
        "Group": Group,
        "LoanPurpose": LoanPurpose,
        }   
