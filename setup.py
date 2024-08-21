import os
from setuptools import setup, find_packages


setup(
    name='iris_evaluation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'pandas',
        'polars',
        'openpyxl',
        'tqdm',
        'torch'
    ],
    author='Rob Hart',
    author_email='robjhart@iu.edu',
    description='A package for evaluating iris images.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.4',
)