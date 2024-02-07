from setuptools import setup, find_packages

setup(
    name='gym_mtsim',
    version='1.3.0',
    packages=find_packages(),

    author='AminHP',
    author_email='mdan.hagh@gmail.com',

    install_requires=[
        'gymnasium>=0.29.1',
        'numpy>=1.25.2',
        'scipy>=1.11.2',
        'pandas>=2.0.3',
        'matplotlib>=3.8.2',
        'plotly>=5.16.1',
        'nbformat>=5.9.2',
        'pathos>=0.3.1',
        'MetaTrader5>=5.0.45; platform_system == "Windows"',
    ],

    package_data={
        'gym_mtsim': ['data/*.pkl']
    }
)
