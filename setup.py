from setuptools import setup, find_packages

setup(
    name='gym_mtsim',
    version='1.1.0',
    packages=find_packages(),

    author='AminHP',
    author_email='mdan.hagh@gmail.com',

    install_requires=[
        'gym>=0.19.0',
        'numpy>=1.19.5',
        'scipy>=1.7.1',
        'pandas>=1.3.1',
        'matplotlib>=3.4.2',
        'plotly>=5.3.1',
        'nbformat>=5.1.3',
        'pathos>=0.2.8',
        'MetaTrader5>=5.0.35',
    ],

    package_data={
        'gym_mtsim': ['data/*.pkl']
    }
)
