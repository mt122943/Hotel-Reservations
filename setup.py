from setuptools import setup, find_packages

setup(
    name='hotel_reservations',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pyyaml',
        'python-dotenv',
    ],
)
