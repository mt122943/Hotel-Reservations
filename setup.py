from setuptools import find_packages, setup

setup(
    name='hotel_reservations',  # Nazwa pakietu
    packages=find_packages(where='src'),  # Znajduje wszystkie pakiety w katalogu 'src'
    package_dir={'': 'src'},  # Określa, że pakiety znajdują się w katalogu 'src'
    version='0.1.0',
    description='This is the basic template for data science projects.',
    author='Marcin',
    license='',  # Usunięta licencja
)
