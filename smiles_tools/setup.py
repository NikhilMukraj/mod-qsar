from setuptools import setup
  

requirements = ['numpy', 'pandas', 'rdkit']

setup(name='smiles_tools',
    version='0.1.1',
    description='Some simple tools for SMILES strings',
    packages=['smiles_tools'],
    install_requires=requirements
)
