from typing import List, Optional, Tuple, Union
import pandas as pd


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Validate the input DataFrame for model compatibility.

    This function checks the input DataFrame for unprocessable values and 
    prepares it for model input. It performs the following steps:
    1. Prints the shape and columns of the input DataFrame for debugging.
    2. Pre-processes the DataFrame using the pre_pipeline_preparation function.
    3. Validates the processed data against the defined schema using Pydantic.
    4. Returns the validated data and any validation errors encountered.

    Args:
        input_df (pd.DataFrame): The DataFrame containing input data to be validated.

    Returns:
        Tuple[pd.DataFrame, Optional[dict]]: A tuple containing the validated DataFrame 
        and a dictionary of errors if validation fails, otherwise None.
    """


class DataInputSchema(BaseModel):
    BorrowerAge: Optional[int]
    CreditScore: Optional[int]
    LoanPurpose: Optional[str]
    LoanType: Optional[str]
    BorrowerTotalMonthlyIncome: Optional[float]
    CLTV: Optional[int]
    DTI: Optional[int]
    ZipCode: Optional[str]
    LeadSourceGroup: Optional[str]
    Cabin: Optional[Union[str, float]]
    BorrowerOwnRent: Optional[str]
    Education: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
    
    