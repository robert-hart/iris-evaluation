import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        red = "\033[91m"
        reset = "\033[0m"
        install.run(self)

        base_dir = os.getcwd()

        dirs_to_create = ['analysis', 'analysis/calibration', 'analysis/comparison', 'analysis/reference', 'analysis/results']

        for dir in dirs_to_create:
            os.makedirs(f'{base_dir}/{dir}', exist_ok=True)

        sys.stdout.write(red + f'\n\n########SAVE ANALYSIS DIRECTORY PATH BELOW########\n\n\t{base_dir}/{dirs_to_create[0]}\n\n#################################################\n\n' + reset)
        sys.stdout.flush
setup(
    name='iris_evaluation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'pandas',
        'polars',
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
    cmdclass={
        'install': CustomInstallCommand,
    },
)