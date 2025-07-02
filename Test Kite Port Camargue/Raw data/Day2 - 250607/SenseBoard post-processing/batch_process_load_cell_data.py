import os
import sys
from clean_load_cell_data_csv import clean_load_cell_data

def batch_process_data(input_directory):
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory {input_directory} is not valid.")
        return

    print(f"Processing CSV files in directory: {input_directory}")
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv") and not filename.endswith("_cleaned.csv"):
            input_file_path = os.path.join(input_directory, filename)
            print(f"Cleaning {filename}...")
            clean_load_cell_data(input_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 batch_process_load_cell_data.py <input_directory_path>")
    else:
        input_dir = sys.argv[1]
        batch_process_data(input_dir)

