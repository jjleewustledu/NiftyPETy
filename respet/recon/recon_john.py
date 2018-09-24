#!/usr/bin/env python
"""Reconstruction of PET data using Python package NiftyPET."""
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2017, University College London"
# ---------------------------------------------------------------------------------


# import matplotlib.pyplot as plt
import sys

import nipet
# get all the constants and LUTs
Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()

#------------------------------------------------------
# GET ALL THE INPUT
folderin = '/data/WashU/HYGLY28/series_1'
datain = nipet.mmraux.explore_input(folderin, Cnt)
if 'T1nii' not in datain and 'T1bc' in datain:
	datain['T1nii'] = datain['T1bc']
#------------------------------------------------------

Cnt['VERBOSE']=True

#------------------------------------------------------
# get the object mu-map through resampling and store it in a numpy array for future uses
muodic = nipet.img.mmrimg.obj_mumap(datain, Cnt, store=True)
# if pseudo-CT mu-map is provided use this:
# mud = nipet.img.mmrimg.pct_mumap(datain, txLUT, axLUT, Cnt, t0=0, t1=0, faff='', fpet='', fcomment='', store=True, petopt='nac')
'''
t0, t1: start and stop time for the PET time frame
faff: string for the path to affine transformation to be used for the mu-map
fpet: string for the path to PET image to which the mu-map will be coregistered
store: store the mu-map as NIfTI image
petopt: the PET image can be reconstructed from list-mode data without attenuation correction, with which the mu-map will be in coregister
'''

#  hardware mu-maps
# [1,2,4]: numbers for top and bottom head-coil (1,2) and patient table (4)
hmudic = nipet.img.mmrimg.hdw_mumap(datain, [1,2,4], Cnt, use_stored=True)

muh = hmudic['im']
muo = muodic['im']
#------------------------------------------------------

from respet.pvc import roi_def as rois

imgs = nipet.img.prc.osemrec(datain, txLUT, axLUT, Cnt, muh, muo=muo, t0=0, t1=0, itr=4, fwhm=0., pvcroi=rois.pvcroi, use_stored=True, fcomment='', ret_all=True)



hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, store=True, use_stored=True)

recon = nipet.prj.mmrprj.osemone(datain, [muh, muo], hst, txLUT, axLUT, Cnt,
                    recmod=3, itr=4, fwhm=0.0, store_img=True, ret_sct=True)

sys.exit()
# PVC
import amyroi_def as amyroi
datain['T1lbl'] = '/data/WashU/026-010/t1_026-010/t1_026-010_NeuroMorph_Parcellation.nii.gz' 
datain['T1nii'] = '/data/WashU/026-010/t1_026-010/t1_026-010_NeuroMorph_BiasCorrected.nii.gz'

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
