import pandas as pd


appliance_to_id_map = {
    'fridge': 1,    
}

def column_extractor(appliance_name, household_id):
    """
    This function extracts the 'Appliance' column from a given CSV file.
    """
    # Load the CSV file
    df = pd.read_csv('data.csv')

    # Extract the 'Age' column
    age_column = df['Age']

    # Display the column
    print(age_column)
