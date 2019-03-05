#!/usr/bin/env python                                                                                                                                                                                                                         
""" Setup Python package 'NiftyPETy' for namespace 'respet'.  See also packages nimpa and nipet from https://github.com/pjmark.                                                                                                               
"""
__author__      = "John J. Lee"
__copyright__   = "Copyright 2019"
# ---------------------------------------------------------------------------------                                                                                                                                                           
from setuptools import setup, find_packages
import sys

print 'i> found packages:'
print find_packages(exclude=['docs'])

stdout = sys.stdout
stderr = sys.stderr
log_file = open('setup_niftypety.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

setup(
    name='NiftyPETy',
    version='0.1',
    packages=find_packages(exclude=['docs']),
    package_data={
        'NiftyPETy': ['tests/*'],
        'NiftyPETy.respet': ['auxdata/*'],
        'NiftyPETy.respet': ['recon/*'],
        'NiftyPETy.respet': ['resolve/*'],        
    },
    license='Apache 2.0',
    long_description=open('README.md').read(),
)

