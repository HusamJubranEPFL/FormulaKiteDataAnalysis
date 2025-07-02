import pandas as pd
import numpy as np
import os

def clean_load_cell_data(input_file_path):
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file_path = os.path.join(os.path.dirname(input_file_path), f"{base_name}cleaned.csv")

    try:
        df = pd.read_csv(input_file_path, delimiter=";")

        # Identify time column (first column)
        time_column = df.columns[0]

        # Identify load cell columns (last 6 columns)
        load_cell_columns = df.columns[-6:].tolist()

        cleaned_df = df.copy()

        for col in load_cell_columns:
            # Ensure the column is numeric before processing
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            data = cleaned_df[col].values
            time = cleaned_df[time_column].values

            # Remove NaN values that might result from coercion
            valid_indices = ~np.isnan(data)
            data_valid = data[valid_indices]
            time_valid = time[valid_indices]

            if len(data_valid) > 2: # Need at least 3 points for a 2nd degree polynomial
                # Fit a 2nd degree polynomial to the data
                poly_coeffs = np.polyfit(time_valid, data_valid, 2)
                drift_baseline = np.polyval(poly_coeffs, time)

                # Subtract the drift baseline from the original data
                detrended_data = data - drift_baseline

                # Adjust for initial offset: make the first point zero
                initial_offset = detrended_data[0]
                cleaned_data = detrended_data - initial_offset

                cleaned_df[col] = cleaned_data
            else:
                print(f"Warning: Not enough valid data points to detrend column {col}. Skipping detrending for this column.")
                cleaned_df[col] = data # Keep original data if not enough points

        cleaned_df.to_csv(output_file_path, index=False, sep=";")
        print(f"Cleaned data saved to {output_file_path}")
        return output_file_path

    except Exception as e:
        print(f"Error during data cleaning for {input_file_path}: {e}")
        return None

if __name__ == "__main__":
    # This part is for testing with a single file. For batch processing, a separate script will call this function.
    test_file = "_imu_log_.csv"
    cleaned_file = clean_load_cell_data(test_file)
    if cleaned_file:
        print(f"Successfully cleaned {test_file} and saved to {cleaned_file}")
    else:
        print(f"Failed to clean {test_file}")

