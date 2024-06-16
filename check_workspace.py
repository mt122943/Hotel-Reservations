import os
import subprocess
import sys
import shutil
from dotenv import load_dotenv, find_dotenv
import pkg_resources

def print_section_title(title):
    """Print section title for better readability in terminal."""
    print("\n" + "-" * 80)
    print(f"- {title}")
    print("-" * 80 + "\n")

def check_virtual_env():
    """Check if the virtual environment is active."""
    print_section_title("CHECK VIRTUAL ENVIRONMENT")
    if os.getenv('VIRTUAL_ENV'):
        print("Virtual environment is active.")
        return True
    else:
        print("Virtual environment is not active. Please activate it.")
        return False

def activate_virtual_env():
    """Activate virtual environment based on the OS."""
    if os.name == 'posix':  # Unix/Linux/macOS
        activate_cmd = 'source .venv/bin/activate'
    elif os.name == 'nt':  # Windows
        activate_cmd = '.venv\\Scripts\\activate'
    else:
        print("Unsupported operating system.")
        return False

    print("Activating virtual environment...")
    subprocess.call(activate_cmd, shell=True)
    return True

def check_installed_packages():
    """Check if required packages are installed."""
    print_section_title("CHECK INSTALLED PACKAGES")
    required_packages = []
    if os.path.isfile('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#') and line.strip() != '-e .']
    else:
        print("requirements.txt not found.")
        return []

    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [pkg.split('==')[0] for pkg in required_packages if pkg.split('==')[0].lower() not in installed_packages]

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
    else:
        print("All required packages are installed.")
    return missing_packages

def install_requirements(missing_packages):
    """Install missing packages."""
    print_section_title("INSTALL MISSING PACKAGES")
    if missing_packages:
        print("Installing missing packages...")
        subprocess.call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    else:
        print("No packages to install.")

def check_env_variables():
    """Check if .env file exists and load variables."""
    print_section_title("CHECK ENVIRONMENT VARIABLES")
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
        print(".env file loaded.")
    else:
        print(".env file not found.")
        return False

    # Add any necessary environment variable checks here
    required_vars = ['RAW_DATA_PATH', 'PROCESSED_DATA_PATH', 'MODEL_SAVE_PATH']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    return True

def check_git_status():
    """Check git repository status."""
    print_section_title("CHECK GIT STATUS")
    if shutil.which("git"):
        print("Git is installed.")
        try:
            result = subprocess.run(['git', 'status'], capture_output=True, text=True, check=True)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("Git repository not initialized or not a git repository.")
            print(e.stderr)
            return False
    else:
        print("Git is not installed.")
        return False

def run_tests():
    """Run tests using tox."""
    print_section_title("RUN TESTS")
    if not shutil.which("tox"):
        print("tox is not installed. Skipping tests.")
        return False

    print("Running tests with tox...")
    result = subprocess.run(['tox'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Tests failed:\n{result.stderr}")
        return False
    else:
        print("All tests passed successfully.")
        return True

def main():
    print_section_title("WORKSPACE STATUS")

    summary = {}

    # Step 1: Check if virtual environment is active
    summary['Virtual Environment'] = check_virtual_env()
    if not summary['Virtual Environment']:
        activate_virtual_env()
        summary['Virtual Environment'] = check_virtual_env()

    # Step 2: Check installed packages and install missing ones
    missing_packages = check_installed_packages()
    summary['Installed Packages'] = len(missing_packages) == 0
    if not summary['Installed Packages']:
        install_requirements(missing_packages)
        missing_packages = check_installed_packages()  # Re-check after installation
        summary['Installed Packages'] = len(missing_packages) == 0

    # Step 3: Check environment variables
    summary['Environment Variables'] = check_env_variables()

    # Step 4: Check git status
    summary['Git Status'] = check_git_status()

    # Step 5: Run tests
    summary['Tests'] = run_tests()

    print_section_title("WORKSPACE CHECK COMPLETE")

    # Print summary
    print_section_title("SUMMARY")
    for key, value in summary.items():
        status = "PASSED" if value else "FAILED"
        print(f"{key}: {status}")

if __name__ == "__main__":
    main()
