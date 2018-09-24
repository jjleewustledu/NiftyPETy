#!/usr/bin/env python

folderin = '/data/nil-bluearc/raichle/jjlee/Local/Pawel/FDG_V1'
humanmu  = '/home/usr/jjlee/Local/Pawel/FDG_V1/umap'
t0_ = 3000
t1_ = 3600




# import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
import time
import scipy.ndimage as ndi
import time
import dicom as dcm
import respet.img

# get all the constants and LUTs
Cnt, txLUT, axLUT = nipet.mmraux.mmrinit()
Cnt['VERBOSE'] = True
Cnt['HMUDIR']  = humanmu

# get all the input
# need to have norm in *.{dcm,bf} or in *.ima; similarly for listmode; need folder with umap (192 DICOMs)
# optional:
#     norm files can be in a "norm" folder and listmode in an "LM" folder and umap in "humanumap";
#     useful for debugging
datain = nipet.mmraux.explore_input(folderin, Cnt)

# image registration/resampling apps; needed if already not defined in ~/.niftypet/resources.py
#Cnt['RESPATH']	 = 'reg_resample'
#Cnt['REGPATH']	 = 'reg_aladin'
#Cnt['DCM2NIIX'] = '/home/usr/jjlee/Local/mricrogl_lx/dcm2niix'

# get the object mu-map through resampling and store it in a numpy array for future uses
# mud = nipet.img.mmrimg.obj_mumap(datain, Cnt, store=True)
# if pseudo-CT mu-map is provided use this:
mud = img.jjl_mmrimg.jjl_mumap(datain, txLUT, axLUT, Cnt, t0=t0_, t1=t1_, faff='', fpet='', fileprefix='umapSynth_full_frame0', fcomment='', store=True, petopt='nac')

'''
t0, t1: start and stop time for the PET time frame in sec
faff: string for the path to affine transformation to be used for the mu-map
fpet: string for the path to PET image to which the mu-map will be coregistered
store: store the mu-map as NIfTI image
petopt: the PET image can be reconstructed from list-mode data without attenuation correction, with which the mu-map will be in coregister
'''

# hardware mu-maps
# [1,2,4]: numbers for top and bottom head-coil (1,2) and patient table (4)
# N.B.:  mricrogl_lx versions after 7/26/2017 produce the following error:
#Chris Rorden's dcm2niiX version v1.0.20170724 GCC4.4.7 (64-bit Linux)
#Found 192 DICOM image(s)
#Convert 192 DICOM as /data/nil-bluearc/raichle/jjlee/Local/Pawel/FDG_V1/umap/converted_e2b (192x192x192x1)
#Conversion required 1.228201 seconds (0.160000 for core code).
#Err:Parameter -pad unknown.
#Usage:/usr/local/nifty_reg/bin/reg_resample -target <referenceImageName> -source <floatingImageName> [OPTIONS].
#See the help for more details (-h).

hmudic = nipet.img.mmrimg.hdw_mumap(datain, [1,2,4], Cnt, use_stored=True)
mumaps = [hmudic['im'], mud['im']]

hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, t0=t0_, t1=t1_, store=True, use_stored=True)

recon = nipet.prj.mmrprj.osemone(datain, mumaps, hst, txLUT, axLUT, Cnt,
                    recmod=3, itr=5, fwhm=0.0, mask_radious=29, store_img=True, ret_sct=True)

matshow(recon.im[60,:,:])
#matshow(recon.im[:,170,:])
#matshow(recon.im[:,:,170])
sys.exit()




class Reconstruction:

    @staticmethod
    def godo(loc, t0, t1, fr, umapIdx):
        # type: (object, object, object, object, object) -> object
        import NiftyPETy
        np = NiftyPETy.NiftyPETy(loc)
        recon = np.reconTimeInterval(int(t0), int(t1), int(fr), int(umapIdx))
        return recon

    def reconAllTimes(self):
        return recon # recon.fpet := NIfTI filename

    def reconTimeInterval(self, t0, t1, fr, umapIdx):
        """@param preconditions of __init__.
           @param t0 is the interval start in sec.
           @param t1 is the interval end in sec.
           @param fr is frame number, 0 ... length(frames)-1.
           @param umapIdx is the index identifying which umap to use, 0 ... length(umaps)-1..
           @return folder self.tracerRawdataLocation/img containing, e.g.:
                   -rw-r--r--+  1 jjlee wheel   28873627 Nov 14 17:47 1_3_12_2.nii.gz"""
        import nipet
        self._t0 = t0
        self._t1 = t1
        self._frame = fr
        self._umapIdx = umapIdx

        # get constants and LUTs
        cnt, txLUT, axLUT = nipet.mmraux.mmrinit()
        cnt['VERBOSE'] = self.verbose
        cnt['HMUDIR'] = self.tracerRawdataLocation + self.umapFolder

        # get data input
        datain = nipet.mmraux.explore_input(
            self.tracerRawdataLocation, cnt)

        # specify image registration/resampling apps if not already not defined in ~/.niftypet/resources.py
        # cnt['RESPATH'] = 'reg_resample'
        # cnt['REGPATH'] = 'reg_aladin'
        # cnt['DCM2NIIX'] = '/home/usr/jjlee/Local/mricrogl_lx/dcm2niix'

        # get mu-map data and store in a numpy array
        mu = self.custom_mumap(
            datain, fileprefix=self.umapSynthFileprefix+str(self._umapIdx), fcomment='', store=True)

        # hardware mu-maps
        # [1,2,4]:  numbers for top and bottom head-coil (1,2) and patient table (4)
        ''' N.B.:  mricrogl_lx versions after 7/26/2017 produce the following error:
        Chris Rorden's dcm2niiX version v1.0.20170724 GCC4.4.7 (64-bit Linux)
        Found 192 DICOM image(s)
        Convert 192 DICOM as /data/nil-bluearc/raichle/jjlee/Local/Pawel/FDG_V1/umap/converted_e2b (192x192x192x1)
        Conversion required 1.228201 seconds (0.160000 for core code).
        Err:Parameter -pad unknown.
        Usage:/usr/local/nifty_reg/bin/reg_resample -target <referenceImageName> -source <floatingImageName> [OPTIONS].
        See the help for more details (-h). '''
        hmudic = nipet.img.mmrimg.hdw_mumap(
            datain, [1, 2, 4], cnt, use_stored=True)
        mumaps = [hmudic['im'], mu]
        hst = nipet.lm.mmrhist.hist(
            datain, txLUT, axLUT, cnt, t0=self._t0, t1=self._t1, store=True, use_stored=True)
        recon = nipet.prj.mmrprj.osemone(
            datain, mumaps, hst, txLUT, axLUT, cnt,
            recmod=3, itr=5, fwhm=0.0, store_img=True, fcomment=self.frameSuffix+str(self._frame), ret_sct=True)
        return recon # recon.fpet := NIfTI filename

    def mMR_mumap(self):

        return mu
