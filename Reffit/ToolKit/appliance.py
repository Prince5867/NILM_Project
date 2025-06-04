import re
from collections import OrderedDict, defaultdict
import pandas as pd
import os
from pathlib import Path
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

                    # else:
                    #     for i in len(appliance_id):
                    #         f'appliance_column{i}' = df[f'Appliance{appliance_id[i]}']
                    #         unix_column = df['Unix']
                    #         new_df = pd.DataFrame({
                    #             'Unix': unix_column,
                    #             f'{appliance_name}{i}': f'appliance_column{i}',
                    #         })
                    #         temp_list.append(new_df)
                    # base_dir = Path(__file__).resolve().parent.parent
                    os.makedirs(f'{self.base_dir}/{appliance_name}', exist_ok=True)
                    new_df.to_csv(f'{self.base_dir}/{appliance_name}/{appliance_name}_{house}.csv', index=False)
                except Exception as e:
                    print(f"Error processing {house} for {appliance_name}: {e}")

def main():
    appliance = ['Fridge','Freezer','Washing Machine','Washer Dryer','Tumble Dryer','Dishwasher','Microwave','Toaster','Kettle',
                'Computer','Television','Electric Heater','Hi-Fi','Router','Dehumidifier','Bread-maker',
                'Games Console','Network Site','Food Mixer','Overhead Fan','Vivarium','Pond Pump']
    
    for appliance in appliance:
        appliance_manipulation = Appliance_Manipulation()
        # appliance_map = appliance_manipulation.map_creator()
        fridge_data = appliance_manipulation.column_extractor(appliance)


if __name__ == "__main__":
    main()