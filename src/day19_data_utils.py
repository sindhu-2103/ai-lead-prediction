
import pandas as pd


# Remove duplicate records
def remove_duplicates(data):

    cleaned_data = data.drop_duplicates()

    return cleaned_data.reset_index(drop=True)


# Fill missing values
def fill_missing_values(data):

    cleaned_data = data.copy()

    for column in cleaned_data.columns:

        if pd.api.types.is_numeric_dtype(cleaned_data[column]):

            median_value = cleaned_data[column].median()

            if pd.notna(median_value):
                cleaned_data[column] = (
                    cleaned_data[column].fillna(median_value)
                )

        else:

            mode_values = cleaned_data[column].mode(dropna=True)

            if not mode_values.empty:
                cleaned_data[column] = (
                    cleaned_data[column].fillna(mode_values.iloc[0])
                )

    return cleaned_data


# Validate required columns
def validate_required_columns(data, required_columns):

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True
