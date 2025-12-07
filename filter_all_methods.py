from pathlib import Path
from shared_code import filter_top_15_stocks_per_month

# Inputs/outputs
input_output_set_paths = [
    (Path("generated_data/VADER_data_output.csv"),    Path("generated_data/VADER_data_filtered.csv")),
    (Path("generated_data/TextBlob_data_output.csv"), Path("generated_data/TextBlob_data_filtered.csv")),
    (Path("generated_data/LM_data_output.csv"),       Path("generated_data/LM_data_filtered.csv")),
]

# Run all filter methods
def main():
    for input_path, output_path in input_output_set_paths:
        filter_top_15_stocks_per_month(input_csv_file = input_path, output_csv_file = output_path, keep_only_top_15_stocks = 15)

if __name__ == "__main__":
    main()
