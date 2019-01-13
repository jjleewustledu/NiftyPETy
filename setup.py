#!/usr/bin/env python
""" Setup Python package 'NiftyPETy' for namespace package 'respet'.  See also packages nimpa and nipet from https://github.com/pjmark.
"""
__author__      = "John J. Lee"
__copyright__   = "Copyright 2019"
# ---------------------------------------------------------------------------------

from setuptools import setup, find_packages

import os
import sys
import platform
from subprocess import call, Popen, PIPE



print 'i> found those packages:'
print find_packages(exclude=['docs'])

with open('README.rst') as file:
    long_description = file.read()

stdout = sys.stdout
stderr = sys.stderr
log_file = open('setup_niftypety.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

assert ('linux' in sys.platform) , 'system platform is not supported'
fex = '*.so'
    
setup(
    name='NiftyPETy',
    license = 'Apache 2.0',
    version='1',
    description='reconstruction and motion-correction for nipet',
    long_description=long_description,
    author='John J. Lee',
    author_email='16329959+jjleewustledu@users.noreply.github.com ',
    url='https://github.com/jjleewustledu/NiftyPETy',
    keywords='PET image reconstruction and analysis',
    install_requires=['nipet>=1.1.8', 'nimpa>=1.1.0', 'pydicom>=1.0.2,<1.3.0', 'nibabel>=2.2.1, <2.4.0'],
    packages=find_packages(exclude=['docs']),
    package_data={
        'NiftyPETy': ['tests/*'],
        'NiftyPETy.respet': ['auxdata/*'],
        'NiftyPETy.respet': ['matlab/*'],
        'NiftyPETy.respet': ['recon/*'],
        'NiftyPETy.respet': ['resolve/*'],
    },
    zip_safe=False,
)

