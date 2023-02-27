from setuptools import setup
  

# specify requirements of your package here
requirements = ['numpy', 'pandas', 'rdkit']

# packages=['tokenization', 'smienumeration'],
# package_dir={"":"smiles_tools"},

# calling the setup function 
setup(name='smiles_tools',
    version='0.1.1',
    description='Some simple tools for SMILES strings',
    packages=['smiles_tools'],
    install_requires=requirements
)
