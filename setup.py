"""Setup for crowdastro.

Based on the setup script from https://github.com/pypa/sampleproject.

Matthew Alger
The Australian National University
2016
"""

from setuptools import setup, find_packages
from os import path

from crowdastro import __version__

here = path.abspath(path.dirname(__file__))

# Load README for long descriptions.
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='crowdastro',
    version=__version__,
    description='Automated cross-identification of radio objects and host '
                'galaxies using crowdsourced labels from the Radio Galaxy Zoo.',
    long_description=long_description,
    url='https://github.com/chengsoonong/crowdastro',
    # Setup scripts don't support multiple authors, so this should be the main
    # author or the author that should be contacted regarding the module.
    author='Matthew Alger',
    author_email='matthew.alger@anu.edu.au',
    license='MIT',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='machine-learning radio astronomy classification',
    packages=['crowdastro'],
    # Should be kept in sync with requirements.txt.
    install_requires=[
        'astropy',
        'h5py',
        'matplotlib',
        'numpy',
        'pandas',
        'pymongo',
        'requests',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        'dev': ['scikit-image'],
        'test': [],
    },
    package_data={
        'crowdastro': ['crowdastro.json'],
    },
)
