import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import os
def digital_to_analog(df, appliance, time_col="Unix" ):
    # Make sure data is sorted
    df = df.sort_values(time_col)

    # Create interpolation function (linear or cubic for smoother curve)
    interpolator = interp1d(
        df[time_col],
        df[appliance],
        kind="linear",  # or "cubic"
        fill_value="extrapolate"
    )
    return interpolator

def resample_uniform(interpolator, start_time, end_time,appliance, step=10):
    # Generate regular timestamps
    uniform_time = np.arange(start_time, end_time, step)
    uniform_values = interpolator(uniform_time)

    
    # Return as DataFrame
    return pd.DataFrame({
        "Unix": uniform_time,
        appliance: uniform_values
    })

def sampling_frequency(df, appliance):
    signal = df[appliance].values  # or any other appliance
    timestamps = df["Unix"].values
    dt = np.median(np.diff(timestamps))
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs = freqs[1:]  # Exclude the zero frequency component
    fft_values = np.abs(np.fft.rfft(signal))  # Remove DC component
    fft_values = fft_values[1:]  # Exclude the zero frequency component
    # print(fft_values)
    # print(freqs)
    # print(max(freqs))
    # print(int(1/(2*max(freqs))))
    return int(1/(2*max(freqs)))

def main():
    # appliance = ['Fridge','Freezer','Washing Machine','Washer Dryer','Tumble Dryer','Dishwasher','Microwave','Toaster','Kettle',
    #             'Computer','Television','Electric Heater','Hi-Fi','Router','Dehumidifier','Bread-maker',
    #             'Games Console','Network Site','Food Mixer','Overhead Fan','Vivarium','Pond Pump']
    
    # appliance = ['Aggregate']
    appliance = ['Synthetic_House']
    
    for appliance in appliance:
        base_dir = Path(__file__).resolve().parent.parent.parent
        Refit_path = os.path.join(base_dir, 'Refit')
        processed_path = os.path.join(Refit_path, 'Processed')
        folder_path = Path(f'{processed_path}/{appliance}')
        try:
            # folder_path = pd.read_csv(f'{Refit_path}/{appliance}')
            output_path = os.path.join(Refit_path, 'Processed',appliance)
            os.makedirs(output_path, exist_ok=True)
            print("output path    ",output_path)
            step = 4
            for item in folder_path.iterdir():
                if item.is_file():
                    print("File:", item)
                    df = pd.read_csv(item)
                    new_step = sampling_frequency(df, appliance)  # Use the sampling frequency of the appliance
                    if new_step < step:
                        step = new_step
                        print(f"Adjusted step size to {step} based on sampling frequency.")

            for item in folder_path.iterdir():
                if item.is_file():
                    print("File:", item)
                    df = pd.read_csv(item)
                    interpolator = digital_to_analog(df, appliance, time_col="Unix")
                    start = df["Unix"].min()
                    end = df["Unix"].max()
                    resampled_df = resample_uniform(interpolator, start, end, appliance, step)

                    output_path = os.path.join(folder_path,item.name)

                    resampled_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error loading data: {e}")

if __name__ == "__main__":
    main()