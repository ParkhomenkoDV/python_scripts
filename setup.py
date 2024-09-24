from setuptools import setup, find_packages

name = 'python_scripts'

with open('README.md', 'rt', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', 'rt') as file:
    install_requires = file.readlines()

setup(
    name=name,
    version='3.0',
    description='lib',
    long_description=long_description,
    long_description_content_type='text/markdown',  # если long_description = .md
    author='Daniil Andryushin',
    author_email='',
    url='https://github.com/ParkhomenkoDV/python_scripts',
    packages=[name],
    python_requires='>=3.8',
    install_requires=install_requires,
)
