from setuptools import setup, find_packages

#TODO change name to iris_evaluation

setup(
    name='generated_iris_evaluation',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'pandas',
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