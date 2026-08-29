import os
import shutil

def clean_synthetic_data():
    data_dir = "data/synthetic"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"Successfully cleaned synthetic data directory: {data_dir}")
    else:
        print(f"Synthetic data directory {data_dir} does not exist.")

if __name__ == "__main__":
    clean_synthetic_data()
