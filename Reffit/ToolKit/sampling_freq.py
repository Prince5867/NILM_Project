import pandas as pd

# Open the output file for writing
with open('C:/Users/niran/OneDrive/Documents/UL/Edge Computing/Project/Datasets/Reffit/average_intervals.txt', 'w') as outfile:
    for i in range(1, 22):
        if i != 14:
            try:
                # Load CSV file
                df = pd.read_csv(f'C:/Users/niran/OneDrive/Documents/UL/Edge Computing/Project/Datasets/Reffit/House_{i}.csv')

                # Extract and clean the Unix timestamps
                timestamps = pd.to_numeric(df['Unix'], errors='coerce').dropna()
                timestamps = timestamps.sort_values().reset_index(drop=True)

                # Compute time differences and average interval
                differences = timestamps.diff().dropna()
                average_interval = differences.mean()

                # Write result to file
                outfile.write(f"Average interval between data points for House {i}: {average_interval:.2f} seconds\n")
            except Exception as e:
                outfile.write(f"Error processing House {i}: {e}\n")
