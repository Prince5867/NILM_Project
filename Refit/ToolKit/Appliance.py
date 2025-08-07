import re
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import math
class Appliance_Manipulation:

    """
    This class provides methods to manipulate and categorize appliances from a dataset.
    It maps various appliance types to unified categories for easier analysis.
    """
    
    def __init__(self):

        self.base_dir = Path(__file__).resolve().parent.parent
        self.data = open(f'{self.base_dir}/ToolKit/house_data.txt', 'r').read()

        # Map similar appliance types to unified categories
        self.appliance_map = {
            'Fridge': ['Fridge', 'Fridge-freezer'],
            'Freezer': ['Freezer', 'Chest Freezer'],
            'Washing Machine': ['Washing Machine'],
            'Washer Dryer': ['Washer Dryer'],
            'Tumble Dryer': ['Tumble Dryer'],
            'Dishwasher': ['Dishwasher'],
            'Microwave': ['Microwave'],
            'Toaster': ['Toaster'],
            'Kettle': ['Kettle'],
            'Computer': ['Computer', 'Desktop Computer', 'Computer Site', 'MJY Computer', 'PGM Computer'],
            'Television': ['Television Site', 'TV/Satellite', 'TV Site', 'Television'],
            'Electric Heater': ['Electric Heater'],
            'Hi-Fi': ['Hi-Fi'],
            'Router': ['Router'],
            'Dehumidifier': ['Dehumidifier'],
            'Bread-maker': ['Bread-maker'],
            'Games Console': ['Games Console'],
            'Network Site': ['Network Site'],
            'Food Mixer': ['Food Mixer', 'K Mix', 'Magimix'],
            'Overhead Fan': ['Overhead Fan'],
            'Vivarium': ['Vivarium'],
            'Pond Pump': ['Pond Pump']
        }

    def map_creator (self):
        """
        This method creates a mapping of appliances to their categories.
        It normalizes the appliance names and prepares them for further processing.
        """
        # Create reverse lookup map
        reverse_map = {}
        for category, variants in self.appliance_map.items():
            for v in variants:
                reverse_map[v] = category

        # Initialize output structure
        appliance_dict = defaultdict(lambda: defaultdict(list))

        # Parse the input line by line
        current_house = None
        for line in self.data.splitlines():
            line = line.strip()
            if line.startswith("House"):
                print(("Current House:", line))
                # Extract house number and normalize it
                current_house = line.replace(" ", "_")  # e.g., "house_1"
            elif current_house and re.match(r"^\d+\.", line):
                # Extract appliance ID and name
                match = re.match(r"^(\d+)\.(.*)", line)
                if match:
                    appliance_id = int(match.group(1))
                    appliance_name = match.group(2).split(',')[0].strip()
                    # print(("Appliance ID:", appliance_id, "Name:", appliance_name))

                    for keyword in reverse_map:
                        # print(("Checking keyword:", keyword))
                        # print(("Appliance Name:", appliance_name))
                        if keyword in appliance_name:
                            # print("passed")
                            normalized = reverse_map[keyword]
                            # print(("Normalized Appliance Name:", normalized))
                            appliance_dict[normalized][current_house].append(appliance_id)
                            break

        # print(("Appliance Dict:", appliance_dict))

        # Convert to regular dict
        final_dict = {k: dict(v) for k, v in appliance_dict.items()}
        # print("Final Dictionary:", final_dict)

        def house_key(house_name):
            return int(house_name.split('_')[1])  # Extracts the number from 'House_XX'

        sorted_final_dict = {}

        for appliance, house_dict in final_dict.items():
            sorted_houses = dict(sorted(house_dict.items(), key=lambda item: house_key(item[0])))
            # print(("Sorted Houses for Appliance:", appliance, "Houses:", sorted_houses))
            sorted_final_dict[appliance] = sorted_houses
            # print(("Sorted Final Dict:", sorted_final_dict))

        return sorted_final_dict
    
    def aggregate_data_extractor(self):
        Refit_path = self.base_dir
        for item in os.listdir(Refit_path):
            if item.startswith('House') and item.endswith('.csv'):
                df = pd.read_csv(f'{Refit_path}/{item}')
                aggregate = df['Aggregate']
                time = df["Unix"] - df["Unix"].iloc[0]

                new_df = pd.DataFrame({
                            'Unix': time,
                            'Aggregate': aggregate,
                        })
                try:
                    os.makedirs(f'{self.base_dir}/Aggregate', exist_ok=True)
                    new_df.to_csv(f'{self.base_dir}/Aggregate/Aggregate_{item}', index=False)
                except Exception as e:
                    print(f"Error processing {item} for Aggregate: {e}")

    
    def column_extractor(self, appliance_name):
        """
        This function extracts the 'Appliance' column from a given CSV file.
        """
        # base_dir = Path(__file__).resolve().parent.parent
        mapping = self.map_creator()
        print(mapping, 'Mapping')
        # print("Hello")
        for appliance, houses in mapping.items():
            print(appliance,houses)
            print('Appliance Name:', appliance_name)
            if appliance == appliance_name:
                appliance_dict = houses
                # print(appliance_dict, 'Heeeeere')
        # print("2")
        for house, appliance_id in appliance_dict.items():
                print(house)
                try:
                        # Load CSV file
                    df = pd.read_csv(f'{self.base_dir}/{house}.csv')

                # Extract and clean the Unix timestamps
                    if len(appliance_id) !=0:
                        appliance_column = df[f'Appliance{appliance_id[0]}']
                        unix_column = df['Unix']
                        new_df = pd.DataFrame({
                            'Unix': unix_column,
                            f'{appliance_name}': appliance_column,
                        })

                    os.makedirs(f'{self.base_dir}/{appliance_name}', exist_ok=True)
                    new_df.to_csv(f'{self.base_dir}/{appliance_name}/{appliance_name}_{house}.csv', index=False)
                except Exception as e:
                    print(f"Error processing {house} for {appliance_name}: {e}")

    def plot_all_appliances_grid(self, appliance_names, max_points=1000):
        """
        Preview each appliance CSV for user confirmation,
        then plot all selected data in a grid at the end.
        """
        import math
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        n = len(appliance_names)
        cols = 3
        rows = math.ceil(n / cols)

        # Store tuples: (appliance_name, df) for approved files
        approved_data = []

        # First: preview and get user confirmation
        for appliance_name in appliance_names:
            folder_path = Path(f'{self.base_dir}/processed/{appliance_name}')
            if not folder_path.exists():
                print(f"Folder not found for appliance: {appliance_name}")
                approved_data.append((appliance_name, None))
                continue

            approved_df = None
            for file in folder_path.glob('*.csv'):
                try:
                    df = pd.read_csv(file)
                    if appliance_name not in df.columns:
                        print(f"Column {appliance_name} not found in {file.name}")
                        continue

                    if df[appliance_name].size < 400:
                        continue

                    preview_df = df.copy()
                    if len(preview_df) > max_points:
                        preview_df = preview_df.iloc[::len(preview_df) // max_points]

                    # Preview plot
                    plt.figure(figsize=(6, 3))
                    plt.plot(preview_df['Unix'], preview_df[appliance_name])
                    plt.title(f"Preview: {file.name}")
                    plt.xlabel("Unix Time")
                    plt.ylabel("Power (W)")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show(block=True)  # waits for window close

                    user_input = input(f"Include {file.name} for {appliance_name}? (y/n): ")
                    plt.close('all')

                    if user_input.lower() == 'y':
                        approved_df = df
                        break  # pick only first approved CSV
                except Exception as e:
                    print(f"Error reading {file.name} for {appliance_name}: {e}")

            approved_data.append((appliance_name, approved_df))

        # Second: create figure and plot all approved data
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), squeeze=False)
        fig.suptitle('Appliance Power Consumption Overview', fontsize=16)

        for idx, (appliance_name, df) in enumerate(approved_data):
            r, c = divmod(idx, cols)
            ax = axes[r][c]

            if df is None:
                ax.set_title(appliance_name.capitalize())
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
                continue

            # Downsample for final plot if needed
            if len(df) > max_points:
                df = df.iloc[::len(df) // max_points]

            ax.plot(df['Unix'], df[appliance_name])
            ax.set_title(appliance_name.capitalize(), fontsize=11)
            ax.set_xlabel('Unix Time')
            ax.set_ylabel('Power (W)')
            ax.grid(True)

        # Remove unused subplots
        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            fig.delaxes(axes[r][c])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def extract_full_house_data(self, window_length,house_no):
        house_path = self.base_dir
        map_houses = self.map_creator()
        df = pd.read_csv(f'{house_path}/{house_no}')

        for appliance,houses in map_houses.items():
            for house,appliance_no in houses.items():
                # print(house, house_no.split(".")[0])
                if house == house_no.split(".")[0]:
                    df = df.rename(columns={f"Appliance{appliance_no[0]}": appliance})
                    df['Unix'] = df['Unix'] - df['Unix'].iloc[0]
                    df['Unix'] = (df['Unix'] // 60) * 60  # round to nearest minute

        df.to_csv(f'{house_path}/Processed/{house_no}', index=False)

    def combining_houses(self, house_list, req_appliances, include_agg=False, resample_freq='1min'):
        import pandas as pd
        import numpy as np
        from functools import reduce

        combined_data_parts = []
        house_path = self.base_dir
        map_houses = self.map_creator()
        used_appliances = set()  # ✅ Track already included appliances

        for house_file in house_list:
            try:
                house_name = house_file.replace(".csv", "")
                df = pd.read_csv(f'{self.base_dir}/{house_file}')

                # ✅ Normalize or convert Unix to datetime for resampling
                if not np.issubdtype(df['Unix'].dtype, np.number):
                    df['Datetime'] = pd.to_datetime(df['Unix'])
                else:
                    df['Datetime'] = pd.to_datetime(df['Unix'], unit='s')
                df.set_index('Datetime', inplace=True)

                keep_cols = []

                if include_agg and 'Aggregate' in df.columns:
                    df.rename(columns={'Aggregate': 'Aggregate'}, inplace=True)
                    keep_cols.append('Aggregate')

                selected_appliances = []

                for appliance, houses in map_houses.items():
                    if appliance not in req_appliances:
                        continue
                    if appliance in used_appliances:
                        continue
                    if house_name not in houses:
                        continue

                    for idx in houses[house_name]:
                        col_name = f'Appliance{idx}'
                        if col_name in df.columns:
                            df.rename(columns={col_name: appliance}, inplace=True)

                    if appliance in df.columns:
                        keep_cols.append(appliance)
                        used_appliances.add(appliance)
                        selected_appliances.append(appliance)

                if selected_appliances:
                    print(f"✔ {house_file} contributed: {selected_appliances}")
                else:
                    print(f"⚠ {house_file} skipped — no new appliances found.")

                # ✅ Only process if useful data found
                if keep_cols:
                    keep_cols.append('Unix') if 'Unix' in df.columns else None
                    temp_df = df[keep_cols].copy()

                    # ✅ Resample to uniform time intervals
                    temp_df_resampled = temp_df.resample(resample_freq).mean()

                    # ✅ Restore Unix from Datetime index
                    temp_df_resampled['Unix'] = temp_df_resampled.index.astype('int64') // 10**9
                    temp_df_resampled.reset_index(drop=True, inplace=True)

                    # ✅ Reorder: move Unix to first column
                    cols = ['Unix'] + [c for c in temp_df_resampled.columns if c != 'Unix']
                    combined_data_parts.append(temp_df_resampled[cols])

            except Exception as e:
                print(f"❌ Error processing {house_file}: {e}")

        if not combined_data_parts:
            return pd.DataFrame()

        combined_df = reduce(lambda left, right: pd.merge(left, right, on='Unix', how='outer'), combined_data_parts)
        combined_df.sort_values('Unix', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        combined_df.to_csv(f'{house_path}/Processed/Synthetic_House.csv', index=False)

        print(f"✅ Final combined appliances: {used_appliances}")
        return combined_df






def main():
    appliance = ['Fridge','Freezer','Washing Machine','Washer Dryer','Tumble Dryer','Dishwasher','Microwave','Toaster','Kettle',
                'Computer','Television','Electric Heater','Hi-Fi','Router','Dehumidifier','Bread-maker',
                'Games Console','Network Site','Food Mixer','Overhead Fan','Vivarium','Pond Pump']
    
    appliance_with_issues = ['Fridge','Freezer','Washing Machine','Dishwasher',
                'Computer','Television','Electric Heater']
    
    appliance_manipulation = Appliance_Manipulation()
    # for appliance in appliance:
        # appliance_map = appliance_manipulation.map_creator()
        # fridge_data = appliance_manipulation.column_extractor(appliance)
    # plot_data = appliance_manipulation.plot_all_appliances_grid(appliance_with_issues)
    # aggregate_data_extractor = appliance_manipulation.aggregate_data_extractor()
    # appliance_manipulation.extract_full_house_data(300,'House_9.csv')
    appliance_manipulation.combining_houses(['House_1.csv','House_20.csv'],['Washer Dryer','Dishwasher','Kettle'], include_agg=False)


if __name__ == "__main__":
    main()