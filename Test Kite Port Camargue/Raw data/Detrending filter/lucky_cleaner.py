import pandas as pd
import numpy as np

file_path = 'Transit_imu_log_250606.xlsx'
output_file_path = 'Cleaned_imu_log_250606.xlsx'

try:
    df = pd.read_excel(file_path)

    # Identify load cell columns
    load_cell_columns = [col for col in df.columns if 'LoadCell' in col]

    cleaned_df = df.copy()

    for col in load_cell_columns:
        # Get the data for the current load cell
        data = df[col].values
        time = df['CPU Timestamp (ms)'].values

        # Fit a polynomial to the data. Using a 2nd degree polynomial for now, can be adjusted.
        # The polynomial is fitted to the entire time series to capture the drift.
        poly_coeffs = np.polyfit(time, data, 2)
        drift_baseline = np.polyval(poly_coeffs, time)

        # Subtract the drift baseline from the original data
        detrended_data = data - drift_baseline

        # Adjust for initial offset: make the first point zero
        initial_offset = detrended_data[0]
        cleaned_data = detrended_data - initial_offset

        cleaned_df[col] = cleaned_data

    cleaned_df.to_excel(output_file_path, index=False)
    print(f"Cleaned data saved to {output_file_path}")
    print(cleaned_df.head())

except Exception as e:
    print(f"Error during data cleaning: {e}")

