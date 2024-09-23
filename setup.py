from setuptools import setup, find_packages

import tools
import decorators
import math_tools
import curves

with open('README.md', 'rt', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', 'rt') as file:
    install_requires = file.readlines()

setup(
    name='tools',
    #version='1.0',
    description='lib',
    long_description=long_description,
    author='Daniil Andryushin',
    author_email='',
    url='https://github.com/ParkhomenkoDV/python_scripts',
    packages=find_packages(exclude=['datas', 'tests', 'test.*']),
    python_requires='>=3.8',
    install_requires=install_requires,
)
