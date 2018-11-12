#!/usr/bin/env python
"""Reconstruction of PET data using Python package NiftyPET."""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2017, University College London"
# ---------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
import time
import scipy.ndimage as ndi
import time
import dicom as dcm
import nipet
# get all the constants and LUTs
Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()
Cnt['SPN']=11
Cnt['BTP']=2

#------------------------------------------------------
# GET ALL THE INPUT
# need to have norm in *.{dcm,bf} or in *.ima; same for listmode; need folder with umap (192 DICOMs)
# optional:   norm files can be in a "norm" folder and listmode in an "LM" folder and umap in "humanumap"; useful for debugging
folderin = '/home2/jjlee/Local/Pawel/FDG_V1_NiftyPETx'
datain = nipet.mmraux.explore_input(folderin, Cnt)
datain['mumapCT'] = '/home2/jjlee/Local/Pawel/FDG_V1_NiftyPETx/mumap_obj/mumapCarney.npy'
datain['mumapUTE'] = '/home2/jjlee/Local/Pawel/FDG_V1_NiftyPETx/mumap_obj/mumapCarney.npy'
Cnt['VERBOSE']=True

#------------------------------------------------------
# image registration/resampling apps; if already not defined in ~/.niftypet/resources.py
#Cnt['RESPATH']	= 'reg_resample'
#Cnt['REGPATH']	= 'reg_aladin'
#Cnt['HMUDIR']	= '/data/nil-bluearc/raichle/jjlee/Local/JSRecon12/hardwareumaps'
#Cnt['DCM2NIIX']	= '/home/usr/jjlee/Local/mricrogl_lx/dcm2niix'

#------------------------------------------------------
# get the object mu-map through resampling and store it in a numpy array for future uses
##mud = nipet.img.mmrimg.obj_mumap(datain, Cnt, store=True)

import nibabel as nib
nim = nib.load('/home2/Shared/jjlee/Local/Pawel/FDG_V1_NiftyPETx/mumap_obj/mumap_fromCT.nii.gz')
mu = nim.get_data()
mu = np.transpose(mu[::-1,::-1,::-1], (2, 1, 0))
#mu = np.zeros_like(mu)
mu = np.float32(mu)
mu[mu<0] = 0
A = nim.get_sform()
mud = {'im':mu, 'affine':A}

# if pseudo-CT mu-map is provided use this:
##mud = nipet.img.jjl_mmrimg.jjl_mumap(datain, txLUT, axLUT, Cnt, t0=0, t1=3600, faff='', fpet='', fileprefix='umapSynthFullsize_frame65', fcomment='', store=True, petopt='nac')
'''
t0, t1: start and stop time for the PET time frame
faff: string for the path to affine transformation to be used for the mu-map
fpet: string for the path to PET image to which the mu-map will be coregistered
store: store the mu-map as NIfTI image
petopt: the PET image can be reconstructed from list-mode data without attenuation correction, with which the mu-map will be in coregister
'''

#  hardware mu-maps
# [1,2,4]: numbers for top and bottom head-coil (1,2) and patient table (4)
# N.B.:  mricrogl_lx versions after 7/26/2017 produce the following error:
#Chris Rorden's dcm2niiX version v1.0.20170724 GCC4.4.7 (64-bit Linux)
#Found 192 DICOM image(s)
#Convert 192 DICOM as /data/nil-bluearc/raichle/jjlee/Local/Pawel/FDG_V1/umap/converted_e2b (192x192x192x1)
#Conversion required 1.228201 seconds (0.160000 for core code).
#Err:Parameter -pad unknown.
#Usage:/usr/local/nifty_reg/bin/reg_resample -target <referenceImageName> -source <floatingImageName> [OPTIONS].
#See the help for more details (-h).

hmudic = nipet.img.mmrimg.hdw_mumap(datain, [1,4,5], Cnt, use_stored=True)
mumaps = [hmudic['im'], mud['im']]
#------------------------------------------------------
hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, t0=0, t1=3600, store=True, use_stored=True)
recon = nipet.prj.mmrprj.osemone(datain, mumaps, hst, txLUT, axLUT, Cnt, recmod=3, itr=10, fwhm=4.0, mask_radious=29, store_img=True, ret_sct=True)





#plt.matshow(recon.im[60,:,:])
#plt.matshow(recon.im[:,170,:])
#plt.matshow(recon.im[:,:,170])
#plt.show()
sys.exit()

# PVC
import amyroi_def as amyroi

imgdir = os.path.join(datain['corepath'], 'img')
imgtmp = os.path.join( imgdir, 'tmp')
nipet.mmraux.create_dir( imgdir )
nipet.mmraux.create_dir( imgtmp )

# get the PSF kernel for PVC
krnlPSF = nipet.img.prc.getPSFmeasured()
# trim PET and upsample.  The trimming is done based on the first file in the list below, i.e., recon with pCT
imdic = nipet.img.prc.trimPET( datain, Cnt, imgs=recon.im, scale=2, fcomment='trim_', int_order=1)
# file name for the trimmed image
fpet = os.path.join(imgtmp, 'trimed_upR.nii.gz')
# save image
nipet.img.mmrimg.array2nii( imdic['im'][0,::-1,::-1,:], imdic['affine'], fpet, descrip='')
imp = {'im':imdic['im'][0,:,:,:], 'fpet':fpet, 'affine':imdic['affine']} 
pvcdic = nipet.img.prc.pvc_iYang(datain, Cnt, imp, amyroi, krnlPSF)
