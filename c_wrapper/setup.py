# https://stackoverflow.com/questions/45121352/how-to-include-a-shared-c-library-in-a-python-package
# https://python-packaging.readthedocs.io/en/latest/minimal.html
from setuptools import setup
  

# specify requirements of your package here
requirements = ['numpy']

# packages=['tokenization', 'smienumeration'],
# package_dir={"":"smiles_tools"},

# calling the setup function 
setup(name='c_wrapper',
    version='0.1.0',
    description='Some helper functions written in C',
    packages=['c_wrapper'],
    include_package_data=True,
    package_data={"" : ["*.sh", "src/*.c", "src/*.so"]},
    install_requires=requirements
)
