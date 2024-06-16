import os
import sys

# Dodanie ścieżki do katalogu głównego projektu
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(project_dir)

from src.config import TEST_VAR

def main():
    print(f"Imported variable from config.py: {TEST_VAR}")
    raw_data_path = os.getenv('RAW_DATA_PATH')
    processed_data_path = os.getenv('PROCESSED_DATA_PATH')
    print(f"RAW_DATA_PATH: {raw_data_path}")
    print(f"PROCESSED_DATA_PATH: {processed_data_path}")

if __name__ == '__main__':
    main()
