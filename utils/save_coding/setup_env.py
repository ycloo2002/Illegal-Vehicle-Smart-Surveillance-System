import os
import subprocess
import sys
import pkg_resources

def check_and_install_packages():
    required_packages = get_required_packages()
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = required_packages - installed_packages

    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        install_packages(missing_packages)
    else:
        print("All required packages are already installed.")

def get_required_packages():
    with open('requirements.txt', 'r') as f:
        packages = {line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')}
    return packages

def install_packages(packages):
    # Create a virtual environment if it doesn't exist
    if not os.path.isdir('env'):
        subprocess.check_call([sys.executable, '-m', 'venv', 'env'])
    
    # Install the missing packages
    pip_path = os.path.join('env', 'Scripts' if os.name == 'nt' else 'bin', 'pip')
    subprocess.check_call([pip_path, 'install'] + list(packages))

if __name__ == '__main__':
    check_and_install_packages()