from setuptools import setup
  

requirements = ['numpy']

setup(name='c_wrapper',
    version='0.1.0',
    description='Some helper functions written in C',
    packages=['c_wrapper'],
    include_package_data=True,
    package_data={"" : ["*.sh", "src/*.c", "src/*.so"]},
    install_requires=requirements
)
