import os
import sys

# Dodanie ścieżki do katalogu głównego projektu
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(project_dir)

from src.config import TEST_VAR

def main():
    print(f"Imported variable from config.py: {TEST_VAR}")

if __name__ == '__main__':
    main()
