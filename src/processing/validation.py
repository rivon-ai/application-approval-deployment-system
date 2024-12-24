from typing import List, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, ValidationError

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
    errors = {}

    try:
        # Print the shape and columns for debugging
        print(f"Input Data Shape: {input_df.shape}")
        print(f"Columns: {input_df.columns.tolist()}")

        # Validate against schema using Pydantic
        input_schema = MultipleDataInputs(inputs=[DataInputSchema(**row) for row in input_df.to_dict(orient='records')])

        # Convert the validated input data into a DataFrame to be passed to the model
        validated_input_df = pd.DataFrame([input.dict() for input in input_schema.inputs])

        # Return the validated data (both the original and the validated DataFrame)
        return validated_input_df, None  # Returning None for no errors
    
    except ValidationError as e:
        # If validation fails, capture the errors
        errors = {"validation_errors": e.errors()}
        return input_df, errors


class DataInputSchema(BaseModel):
    CoBorrowerTotalMonthlyIncome: Optional[float]
    CoBorrowerAge: Optional[int]
    CoBorrowerYearsInSchool: Optional[int]
    BorrowerTotalMonthlyIncome: Optional[float]
    BorrowerTotalMonthlyIncome: Optional[float]
    BorrowerAge: Optional[int]
    DTI: Optional[int]
    CLTV: Optional[int]
    CreditScore: Optional[int]
    TotalLoanAmount: Optional[int]
    LeadSourceGroup: Optional[str]
    Group: Optional[str]
    LoanPurpose: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
    
    