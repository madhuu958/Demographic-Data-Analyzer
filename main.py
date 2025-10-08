import pandas as pd

# Define the file path
FILE_PATH = 'adults.csv'

def load_data(file_path):
    """Loads the CSV file into a pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    try:
        # The 'adults.csv' is common, and often lacks a header row, so we'll define them.
        # However, checking the snippet, it looks like it might not have headers or a mix.
        # Let's try loading without explicit headers first.
        df = pd.read_csv(file_path, header=None, skipinitialspace=True)
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Please check the path.")
        return None

def clean_and_name_columns(df):
    """Assigns standard column names and cleans data."""
    print("\nCleaning and naming columns...")

    # Standard column names for the 'Adult' dataset
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income' # This is the target variable
    ]

    # Check if the number of columns matches the expected names
    if df.shape[1] == len(column_names):
        df.columns = column_names
    else:
        # If the number of columns doesn't match, we'll use a generic name
        print(f"WARNING: Expected {len(column_names)} columns but found {df.shape[1]}. Using generic names.")
        df.columns = [f'col_{i}' for i in range(df.shape[1])]


    # Clean up string columns by stripping whitespace (common in this dataset)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        # Also replace '?' with NaN if necessary (common for missing values)
        df[col] = df[col].replace('?', pd.NA)

    return df

def perform_eda(df):
    """Performs basic Exploratory Data Analysis (EDA)."""
    print("\n--- Exploratory Data Analysis (EDA) ---")

    # 1. Display shape and info
    print(f"\nDataFrame Shape: {df.shape}")
    print("\nDataFrame Info:")
    df.info()

    # 2. Display the first few rows
    print("\nFirst 5 Rows of the Data:")
    print(df.head())

    # 3. Check for missing values
    print("\nMissing Values Count:")
    print(df.isnull().sum())

    # 4. Analyze the target variable ('income') distribution if it exists
    if 'income' in df.columns:
        print("\nIncome Distribution:")
        income_counts = df['income'].value_counts(normalize=True) * 100
        print(income_counts)
        print(f"\n{income_counts.loc['>50K']:.2f}% of people earn more than $50K.")

    # 5. Analyze 'age' statistics
    if 'age' in df.columns:
        print("\nAge Statistics:")
        print(df['age'].describe())

def main():
    """Main function to run the data analysis project."""
    df = load_data(FILE_PATH)

    if df is not None:
        df = clean_and_name_columns(df)
        perform_eda(df)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()