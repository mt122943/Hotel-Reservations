import os
import subprocess
from dotenv import load_dotenv

def check_virtual_env():
    """Check if the virtual environment is active."""
    if os.getenv('VIRTUAL_ENV'):
        print("Virtual environment is active.")
    else:
        print("Virtual environment is not active. Please activate the virtual environment.")

def run_test_data():
    """Run the test_data.py script with PYTHONPATH set and provide more diagnostics."""
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    # Load environment variables from .env file
    dotenv_path = os.path.join(project_dir, '.env')
    if os.path.exists(dotenv_path):
        print(f"Attempting to load .env file from: {dotenv_path}")
        load_dotenv(dotenv_path)
        print("Loaded .env file")
    else:
        print(".env file not found")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = project_dir
    print(f"Running script with PYTHONPATH: {env['PYTHONPATH']}")
    result = subprocess.run(
        ['python', os.path.join(project_dir, 'src/data/test_data.py')],
        capture_output=True, text=True, env=env
    )
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

if __name__ == '__main__':
    check_virtual_env()
    run_test_data()
