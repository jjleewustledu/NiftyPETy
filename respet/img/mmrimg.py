"""image functions for PET data reconstruction and processing."""

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2016, University College London"

#-------------------------------------------------------------------------------

import numpy as np
# import matplotlib.pyplot as plt
import math
import sys
import os
import scipy.ndimage as ndi

import nibabel as nib
import pydicom as dcm
import re
import glob

import nipet
import improc

import time

from subprocess import call



#=================================================================================================
# IMAGE ROUTINES
#=================================================================================================



def convert2e7(img, Cnt):
    '''Convert GPU optimsed image to E7 image dimension (127,344,344)'''

    margin = (Cnt['SO_IMX']-Cnt['SZ_IMX'])/2

    #permute the dims first
    imo = np.transpose(img, (2,0,1))

    nvz = img.shape[2]

    #get the x-axis filler and apply it
    filler = np.zeros((nvz, Cnt['SZ_IMY'], margin), dtype=np.float32)
    imo = np.concatenate((filler, imo, filler), axis=2)

    #get the y-axis filler and apply it
    filler = np.zeros((nvz, margin, Cnt['SO_IMX']), dtype=np.float32)
    imo = np.concatenate((filler, imo, filler), axis=1)
    return imo

def convert2dev(im, Cnt):
    #reorganise image for optimal GPU execution
    im_sqzd = np.zeros((Cnt['SZ_IMZ'], Cnt['SZ_IMY'], Cnt['SZ_IMX']), dtype=np.float32)
    vz0  = 2*Cnt['RNG_STRT']
    vz1_ = 2*Cnt['RNG_END']
    margin = (Cnt['SO_IMX']-Cnt['SZ_IMX'])/2
    margin_=-margin
    if margin==0: 
        margin = None
        margin_= None
    im_sqzd[vz0:vz1_, :, :] = im[:, margin:margin_, margin:margin_]
    im_sqzd = np.transpose(im_sqzd, (1, 2, 0))
    return im_sqzd

def cropxy(im, imsize, datain, Cnt, store_pth=''):
    '''
    crop image transaxially to the size in tuple imsize.  
    returns the image and the affine matrix.
    '''
    if not imsize[0]%2==0 and not imsize[1]%2==0:
        print 'e> image size has to be an even number!'
        return None

    # cropping indexes
    i0 = (Cnt['SO_IMX']-imsize[0])/2
    i1 = (Cnt['SO_IMY']+imsize[1])/2

    # bed position
    vbed, hbed = nipet.mmraux.vh_bedpos(datain, Cnt)

    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(   .5*Cnt['SO_IMX']   *Cnt['SO_VXX'] )         - 10*Cnt['SO_VXX']*i0
    B[1,3] = 10*( (-.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY'] )         + 10*Cnt['SO_VXY']*(Cnt['SO_IMY']-i1)
    B[2,3] = 10*( (-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] + hbed)

    cim = im[:, i0:i1, i0:i1]

    if store_pth!='':
        respet.img.mmrimg.array2nii(cim[::-1, ::-1, :], B, store_pth, descrip='cropped')
        if Cnt['VERBOSE']:  print 'i> saved cropped image to:', store_pth

    return cim, B

#================================================================================
def getnii(fim):
    '''Get NIfTI file in E7 dimensions'''
    nim = nib.load(fim)
    imr = nim.get_data()
    imr[np.isnan(imr)]=0
    #flip y-axis and z-axis
    imr  = imr[:,::-1,::-1]
    imr  = np.transpose(imr, (2, 1, 0))
    return imr

def getnii_affine(fim):
    '''Get NIfTI file in E7 dimensions'''
    nim = nib.load(fim)
    A = nim.get_sform()
    return A

def getniiDescr(fim):
    '''
    Extracts the custom description header field to dictionary
    '''
    nim = nib.load(fim)
    hdr = nim.header
    rcnlst = hdr['descrip'].item().split(';')
    rcndic = {}
    
    if rcnlst[0]=='':
        # print 'w> no description in the NIfTI header'
        return rcndic
    
    for ci in range(len(rcnlst)):
        tmp = rcnlst[ci].split('=')
        rcndic[tmp[0]] = tmp[1]
    return rcndic

def array2nii(im, A, fnii, descrip=''):
    # print 'A = ', A
    im = np.transpose(im, (2, 1, 0))
    nii = nib.Nifti1Image(im, A)
    hdr = nii.header
    hdr.set_sform(None, code='scanner')
    hdr['cal_max'] = np.max(im)
    hdr['cal_min'] = np.min(im)
    hdr['descrip'] = descrip
    nib.save(nii, fnii)

def array4D2nii(im, A, fnii, descrip=''):
    # print 'A = ', A
    im = np.transpose(im, (3, 2, 1, 0))
    nii = nib.Nifti1Image(im, A)
    hdr = nii.header
    hdr.set_sform(None, code='scanner')
    hdr['cal_max'] = np.max(im)
    hdr['cal_min'] = np.min(im)
    hdr['descrip'] = descrip
    nib.save(nii, fnii)


def savenii(im, fpth, A, Cnt):
    '''save image data to NIfTI file'''

    if im.shape[0]==Cnt['SZ_IMX'] and im.shape[1]==Cnt['SZ_IMY'] and im.shape[2]==Cnt['SZ_IMZ']:
        print 'i> image in the GPU processing size of (320,320,128) will be converted to the original size.'
        im = convert2e7(im, Cnt)
      
    imn = np.transpose(im, (2, 1, 0))
    #flip y-axis and z-axis
    imn = imn[:,::-1,::-1]
    
    #make new nifti image
    nii = nib.Nifti1Image(imn, A)
    hdr = nii.header
    hdr.set_sform(None, code='scanner')
    hdr['cal_max'] = np.max(imn)
    hdr['cal_min'] = np.min(imn)
    #save to nifti image file
    nib.save(nii, fpth)

def orientnii(datain):
    '''Get the orientation from NIfTI sform.  Not fully functional yet.'''
    strorient = ['L-R', 'S-I', 'A-P']
    niiorient = []
    niixyz = np.zeros(3,dtype=np.int8)

    if os.path.isfile(datain['pCT']):
        nim = nib.load(datain['pCT'])
        pct = nim.get_data()
        A = nim.get_sform()
        for i in range(3):
            niixyz[i] = np.argmax(abs(A[i,:-1]))
            niiorient.append( strorient[ niixyz[i] ] )
        print niiorient
#================================================================================

def getmu_off(mu, Cnt, Offst=np.array([0., 0., 0.])):
    #number of voxels
    nvx = mu.shape[0]
    #change the shape to 3D
    mu.shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    #-------------------------------------------------------------------------
    # CORRECT THE MU-MAP for GANTRY OFFSET
    #-------------------------------------------------------------------------
    Cim = {
        'VXSOx':0.208626,
        'VXSOy':0.208626,
        'VXSOz':0.203125,
        'VXNOx':344,
        'VXNOy':344,
        'VXNOz':127,

        'VXSRx':0.208626,
        'VXSRy':0.208626,
        'VXSRz':0.203125,
        'VXNRx':344,
        'VXNRy':344,
        'VXNRz':127
    }
    #original image offset
    Cim['OFFOx'] = -0.5*Cim['VXNOx']*Cim['VXSOx']
    Cim['OFFOy'] = -0.5*Cim['VXNOy']*Cim['VXSOy']
    Cim['OFFOz'] = -0.5*Cim['VXNOz']*Cim['VXSOz']
    #resampled image offset
    Cim['OFFRx'] = -0.5*Cim['VXNRx']*Cim['VXSRx']
    Cim['OFFRy'] = -0.5*Cim['VXNRy']*Cim['VXSRy']
    Cim['OFFRz'] = -0.5*Cim['VXNRz']*Cim['VXSRz']
    #transformation matrix
    A = np.array(
        [[ 1., 0., 0.,  Offst[0] ],
        [  0., 1., 0.,  Offst[1] ],
        [  0., 0., 1.,  Offst[2] ],
        [  0., 0., 0.,  1. ]], dtype=np.float32
        )
    #apply the gantry offset to the mu-map
    mur = improc.resample(mu, A, Cim)
    return mur

def getinterfile_off(fmu, Cnt, Offst=np.array([0., 0., 0.])):
    "Return the floating point mu-map in an array from Interfile, accounting for image offset (does slow interpolation)."

    #read the image file
    f = open(fmu, 'rb')
    mu = np.fromfile(f, np.float32)
    f.close()
    
    # save_im(mur, Cnt, os.path.dirname(fmu) + '/mur.nii')
    #-------------------------------------------------------------------------
    mur = respet.img.mmrimg.getmu_off(mu, Cnt)
    #create GPU version of the mu-map
    murs = convert2dev(mur, Cnt)
    #get the basic stats
    mumax = np.max(mur)
    mumin = np.min(mur)
    #number of voxels greater than 10% of max image value
    n10mx = np.sum(mur>0.1*mumax)
    #return image dictionary with the image itself and some other stats
    mu_dct = {'im':mur,
              'ims':murs,
              'max':mumax,
              'min':mumin,
              'nvx':nvx,
              'n10mx':n10mx}
    return mu_dct

#================================================================================
def getinterfile(fim, Cnt):
    '''Return the floating point image file in an array from an Interfile file.'''
    
    #read the image file
    f = open(fim, 'rb')
    im = np.fromfile(f, np.float32)
    f.close()

    #number of voxels
    nvx = im.shape[0]
    #change the shape to 3D
    im.shape = (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX'])

    #get the basic stats
    immax = np.max(im)
    immin = np.min(im)

    #number of voxels greater than 10% of max image value
    n10mx = np.sum(im>0.1*immax)

    #reorganise the image for optimal gpu execution
    im_sqzd = convert2dev(im, Cnt)

    #return image dictionary with the image itself and some other stats
    im_dct = {'im':im,
              'ims':im_sqzd,
              'max':immax,
              'min':immin,
              'nvx':nvx,
              'n10mx':n10mx}

    return im_dct
#================================================================================


#=================================================================================
#define uniform cylinder
def get_cylinder(Cnt, rad=25, xo=0, yo=0, unival=1, gpu_dim=False):
    '''Outputs image with a uniform cylinder of intensity = unival, radius = rad, and transaxial centre (xo, yo)'''
    imdsk = np.zeros((1, Cnt['SO_IMX'], Cnt['SO_IMY']), dtype=np.float32)
    for t in np.arange(0, math.pi, math.pi/(2*360)):
        x = xo+rad*math.cos(t)
        y = yo+rad*math.sin(t)
        yf = np.arange(-y+2*yo, y, Cnt['SO_VXY']/2)
        v = np.int32(.5*Cnt['SO_IMX'] - np.ceil(yf/Cnt['SO_VXY']))
        u = np.int32(.5*Cnt['SO_IMY'] + np.floor(x/Cnt['SO_VXY']))
        imdsk[0,v,u] = unival
    imdsk = np.repeat(imdsk, Cnt['NSEG0'], axis=0)
    if gpu_dim: imdsk = respet.img.mmrimg.convert2dev(imdsk, Cnt)
    return imdsk


#================================================================================
# Get DICOM with affine transformation (matrix)
def dcm2im(fpth):
    '''get the DICOM files from <fpth> into image with the affine transformation.'''

    SZ0 = len([d for d in os.listdir(fpth) if d.endswith(".dcm")])
    if SZ0<1:
        print 'e> no DICOM images in the specified path.'
        sys.exit()
        #

    #patient position
    dhdr = dcm.read_file(os.path.join(fpth, os.listdir(fpth)[0]))
    if [0x018, 0x5100] in dhdr:
        ornt = dhdr[0x18,0x5100].value
    else:
        ornt = 'unkonwn'

    # image position 
    P = np.zeros((SZ0,3), dtype=np.float64)
    #image orientation
    Orn = np.zeros((SZ0,6), dtype=np.float64)
    #xy resolution
    R = np.zeros((SZ0,2), dtype=np.float64)
    #slice thickness
    S = np.zeros((SZ0,1), dtype=np.float64)
    #slope and intercept
    SI = np.ones((SZ0,2), dtype=np.float64)
    SI[:,1] = 0

    #image data as an list of array for now
    IM = []

    c = 0;
    for d in os.listdir(fpth):
        if d.endswith(".dcm"):
            dhdr = dcm.read_file(fpth+'/'+d)
            P[c,:] = np.array([float(f) for f in dhdr[0x20,0x32].value])
            Orn[c,:] = np.array([float(f) for f in dhdr[0x20,0x37].value])
            R[c,:] = np.array([float(f) for f in dhdr[0x28,0x30].value])
            S[c,:] = float(dhdr[0x18,0x50].value)
            if [0x28,0x1053] in dhdr and [0x28,0x1052] in dhdr:
                SI[c,0] = float(dhdr[0x28,0x1053].value)
                SI[c,1] = float(dhdr[0x28,0x1052].value)
            IM.append(dhdr.pixel_array)
            c += 1


    #check if orientation/resolution is the same for all slices
    if np.sum(Orn-Orn[0,:]) > 1e-6:
        print 'e> varying orientation for slices'
    else:
        Orn = Orn[0,:]
    if np.sum(R-R[0,:]) > 1e-6:
        print 'e> varying resolution for slices'
    else:
        R = R[0,:]

    # Patient Position
    patpos = dhdr[0x18,0x5100].value
    # Rows and Columns
    SZ2 = dhdr[0x28,0x10].value
    SZ1 = dhdr[0x28,0x11].value
    # image resolution
    SZ_VX2 = R[0]
    SZ_VX1 = R[1]

    #now sort the images along k-dimension
    k = np.argmin(abs(Orn[:3]+Orn[3:]))
    #sorted indeces
    si = np.argsort(P[:,k])
    Pos = np.zeros(P.shape, dtype=np.float64)
    im = np.zeros((SZ0, SZ1, SZ2 ), dtype=np.float32)

    #check if the dementions are in agreement (the pixel array could be transposed...)
    if IM[0].shape[0]==SZ1:
        for i in range(SZ0):
            im[i,:,:] = IM[si[i]]*SI[si[i],0] + SI[si[i],1]
            Pos[i,:] = P[si[i]]
    else:
        for i in range(SZ0):
            im[i,:,:] = IM[si[i]].T * SI[si[i],0] + SI[si[i],1]
            Pos[i,:] = P[si[i]]

    # proper slice thickness
    Zz = (P[si[-1],2] - P[si[0],2])/(SZ0-1)
    Zy = (P[si[-1],1] - P[si[0],1])/(SZ0-1)
    Zx = (P[si[-1],0] - P[si[0],0])/(SZ0-1)
    

    # dictionary for affine and image size for the image
    A = {
        'AFFINE':np.array([[SZ_VX2*Orn[0], SZ_VX1*Orn[3], Zx, Pos[0,0]],
                           [SZ_VX2*Orn[1], SZ_VX1*Orn[4], Zy, Pos[0,1]],
                           [SZ_VX2*Orn[2], SZ_VX1*Orn[5], Zz, Pos[0,2]],
                           [0., 0., 0., 1.]]),
        'SHAPE':(SZ0, SZ1, SZ2)
    }

    #the returned image is already scaled according to the dcm header
    return im, A, ornt


def mr2petAffine(datain, Cnt, fpet, fcomment='', rmsk=True, rfwhm=1.5, rthrsh=0.05, pi=50, pv=50, smof=0, smor=0):
    # --- MR T1w
    if os.path.isfile(datain['T1nii']):
        ft1w = datain['T1nii']
    elif os.path.isfile(datain['T1bc']):
        ft1w = datain['T1bc']
    elif os.path.isdir(datain['MRT1W']):
        # create file name for the converted NIfTI image
        fnii = 'converted'
        call( [ Cnt['DCM2NIIX'], '-f', fnii, datain['T1nii'] ] )
        ft1nii = glob.glob( os.path.join(datain['T1nii'], '*converted*.nii*') )
        ft1w = ft1nii[0]
    else:
        print 'e> disaster: no T1w image!'
        sys.exit()

    if rmsk:
        fimdir = os.path.join(os.path.join(datain['corepath'],'img'), 'tmp')
        nipet.mmraux.create_dir(fimdir)
        fmsk = os.path.join(fimdir, 'rmask.nii.gz')
        smoim = ndi.filters.gaussian_filter(respet.img.mmrimg.getnii(fpet),
                                            nipet.mmraux.fwhm2sig(rfwhm), mode='mirror')
        thrsh = rthrsh*smoim.max()
        immsk = np.int8(smoim>thrsh)
        for iz in range(immsk.shape[0]):
            for iy in range(immsk.shape[1]):
                ix0 = np.argmax(immsk[iz,iy,:]>0)
                ix1 = immsk.shape[2] - np.argmax(immsk[iz,iy,::-1]>0)
                if (ix1-ix0) > immsk.shape[2]-10: continue
                immsk[iz,iy,ix0:ix1] = 1
        respet.img.mmrimg.array2nii(immsk[::-1, ::-1, :], respet.img.mmrimg.getnii_affine(fpet), fmsk)

    #create a folder for MR images registered to PET
    mrodir = os.path.join(os.path.dirname(ft1w),'mr2pet')
    nipet.mmraux.create_dir(mrodir)

    #output for the T1w in register with PET
    ft1out = os.path.join(mrodir, 'T1w_r_to_'+os.path.basename(fpet).split('.')[0]+'_'+fcomment+'.nii.gz')
    #text file for the affine transform T1w->PET
    faff   = os.path.join(mrodir, 'mr2pet_affine'+os.path.basename(fpet).split('.')[0]+'_'+fcomment+'.txt')  
    #call the registration routine
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [Cnt['REGPATH'],
             '-ref', fpet,
             '-flo', ft1w,
             '-rigOnly', '-speeeeed',
             '-aff', faff,
             '-pi', str(pi),
             '-pv', str(pv),
             '-smooF', str(smof),
             '-smooR', str(smor),
             '-res', ft1out]
        if rmsk: 
            cmd.append('-rmask')
            cmd.append(fmsk)
        if not Cnt['VERBOSE']: cmd.append('-voff')
        print cmd
        call(cmd)
    else:
        print 'e> path to registration executable is incorrect!'
        sys.exit()
        
    return faff


def nii_res(imo, imref, imflo, aff, Cnt):
    fout = os.path.dirname( imo )
    #get the affine transformations and store to text file for resampling
    faff = os.path.join(fout, 'affine.txt')
    np.savetxt(faff, aff, fmt='%1.8f')
    # the converted nii image resample to the reference size
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [ Cnt['RESPATH'],
                    '-ref', imref,
                    '-flo', imflo,
                    '-trans', faff,
                    '-res', imo,
                    '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
    else:
        print 'e> path to resampling executable is incorrect!'
        sys.exit()

def nii_ugzip(fim):
    import gzip
    with gzip.open(fim, 'rb') as f:
        s = f.read()
    # Now store the uncompressed data
    fout = fim[:-3]
    # store uncompressed file data from 's' variable
    with open(fout, 'wb') as f:
        f.write(s)
    return fout

def nii_gzip(fim):
    import gzip
    with open(fim, 'rb') as f:
        d = f.read()
    # Now store the uncompressed data
    fout = fim+'.gz'
    # store compressed file data from 'd' variable
    with gzip.open(fout, 'wb') as f:
        f.write(d)
    return fout


#---- SPM ----
def spm_resample(imref, imflo, m, intrp=1, dirout='', r_prefix='r_', del_ref_uncmpr=False, del_flo_uncmpr=False, del_out_uncmpr=False):
    import matlab.engine
    from pkg_resources import resource_filename
    # start matlab engine
    eng = matlab.engine.start_matlab()
    # add path to SPM matlab file
    spmpth = resource_filename(__name__, 'spm')
    eng.addpath(spmpth, nargout=0)

    # decompress if necessary 
    if imref[-3:]=='.gz':
        imrefu = respet.img.mmrimg.nii_ugzip(imref)
    else:
        imrefu = imref
    if imflo[-3:]=='.gz': 
        imflou = respet.img.mmrimg.nii_ugzip(imflo)
    else:
        imflou = imflo

    # run the matlab SPM resampling
    mm = eng.spm_resample(imrefu, imflou, matlab.single(m.tolist()), intrp, r_prefix)

    # delete the uncomressed
    if del_ref_uncmpr:  os.remove(imrefu)
    if del_flo_uncmpr:  os.remove(imflou)

    # compress the output
    split = os.path.split(imflou)
    fim = os.path.join(split[0], r_prefix+split[1])
    respet.img.mmrimg.nii_gzip(fim)
    if del_out_uncmpr: os.remove(fim)

    if dirout!='':
        # move to the output dir
        fout = os.path.join(dirout, r_prefix+split[1]+'.gz')
        os.rename(fim+'.gz', fout)
    else:
        fout = fim+'.gz'

    return fout


def spm_coreg(imref, imflo, del_uncmpr=False):
    import matlab.engine
    from pkg_resources import resource_filename
    # start matlab engine
    eng = matlab.engine.start_matlab()
    # add path to SPM matlab file
    spmpth = resource_filename(__name__, 'spm')
    eng.addpath(spmpth, nargout=0)

    # decompress if necessary 
    if imref[-3:]=='.gz':
        imrefu = respet.img.mmrimg.nii_ugzip(imref)
    else:
        imrefu = imref
    if imflo[-3:]=='.gz': 
        imflou = respet.img.mmrimg.nii_ugzip(imflo)
    else:
        imflou = imflo

    # run the matlab SPM coregistration
    M = eng.spm_mr2pet(imrefu, imflou)
    # get the affine matrix
    m = np.array(M._data.tolist())
    m = m.reshape(4,4).T
    # delete the uncompressed files
    if del_uncmpr:
        if imref[-3:]=='.gz': os.remove(imrefu)
        if imflo[-3:]=='.gz': os.remove(imflou)

    return m

#---- FSL ----
def fsl_coreg(imref, imflo, faff, costfun='normmi', dof=6):

    cmd = [ 'fsl5.0-flirt',
            '-cost', costfun,
            '-dof', str(dof),
            '-omat', faff,
            '-in', imflo,
            '-ref', imref]
    call(cmd)

    # convert hex parameters to dec
    aff = np.loadtxt(faff)
    # faffd = faff[:-4]+'d.mat'
    np.savetxt(faff, aff)
    return aff


def fsl_res(imout, imref, imflo, faff, interp=1):

    if interp==1:
        interpolation = 'trilinear'
    elif interp==0:
        interpolation = 'nearestneighbour'

    cmd = [ 'fsl5.0-flirt',
            '-in', imflo,
            '-ref', imref,
            '-out', imout,
            '-applyxfm', '-init', faff,
            '-interp', interpolation]
    call(cmd)

#================================================================================
def hu2mu(im):
    '''HU units to 511keV PET mu-values'''

    # convert nans to -1024 for the HU values only
    im[np.isnan(im)] = -1024
    # constants
    muwater  = 0.096
    mubone   = 0.172
    rhowater = 0.158
    rhobone  = 0.326
    uim = np.zeros(im.shape, dtype=np.float32)
    uim[im<=0] = muwater * ( 1+im[im<=0]*1e-3 )
    uim[im> 0] = muwater * ( 1+im[im> 0]*1e-3 * rhowater/muwater*(mubone-muwater)/(rhobone-rhowater) )
    return uim


# =====================================================================================
# object/patient mu-map resampling to nifti
# better use dcm3nii
def mudcm2nii(datain, Cnt):
    '''DICOM mu-map to NIfTI'''
    mu, pos, ornt = dcm2im(datain['mumapDCM']) #nipet.img.mmrimg.
    mu *= 0.0001
    A = pos['AFFINE']
    A[0,0] *= -1
    A[0,3] *= -1
    A[1,3] += A[1,1]
    array2nii(mu[:,::-1,:], A, os.path.join(os.path.dirname(datain['mumapDCM']),'mu.nii.gz'))

    #------get necessary data for creating a blank reference image (to which resample)-----
    # gantry offset
    goff, tpo = nipet.mmraux.lm_pos(datain, Cnt)
    ihdr, csainfo = nipet.mmraux.hdr_lm(datain)
    #start horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(.5*Cnt['SO_IMX']*Cnt['SO_VXX']      + goff[0])
    B[1,3] = 10*((-.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY'] - goff[1])
    B[2,3] = 10*((-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] - goff[2] + hbedpos)
    im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
    array2nii(im, B, os.path.join(os.path.dirname(datain['mumapDCM']),'muref.nii.gz'))
    # -------------------------------------------------------------------------------------
    fmu = os.path.join(os.path.dirname(datain['mumapDCM']),'mu_r.nii.gz')
    if os.path.isfile( Cnt['RESPATH'] ):
        call( [ Cnt['RESPATH'],
                    '-ref', os.path.join(os.path.dirname(datain['mumapDCM']),'muref.nii.gz'),
                    '-flo', os.path.join(os.path.dirname(datain['mumapDCM']),'mu.nii.gz'),
                    '-res', fmu,
                    '-pad', '0'] )
    else:
        print 'e> path to resampling executable is incorrect!'
        sys.exit()

    return fmu

# =====================================================================================
def obj_mumap(datain, Cnt, store=False, comment=''):
    '''Get the object mu-map from DICOM images'''

    # get NIfTI from DICOM mu-maps slices
    # fmu = mudcm2nii(datain, Cnt)

    #------get necessary data for creating a blank reference image (to which resample)-----
    # gantry offset
    goff, tpo = nipet.mmraux.lm_pos(datain, Cnt)
    ihdr, csainfo = nipet.mmraux.hdr_lm(datain, Cnt)
    #start horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    # output folder
    fout = os.path.join( datain['corepath'], 'mumap_obj' )
    nipet.mmraux.create_dir(fout)

    # ref file name
    fmuref = os.path.join(fout, 'muref.nii.gz')

    # create a reference empty mu-map image
    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(.5*Cnt['SO_IMX']*Cnt['SO_VXX']      + goff[0])
    B[1,3] = 10*((-.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY'] - goff[1])
    B[2,3] = 10*((-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] - goff[2] + hbedpos)
    im = np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)
    array2nii(im, B, fmuref)
    # -------------------------------------------------------------------------------------

    # check if the object dicom files for MR-based mu-map exists
    if not os.path.isdir(datain['mumapDCM']):
        print 'e> DICOM forlder for the mu-map does not exist.'
        return None

    fnii = 'converted'
    # convert the DICOM mu-map images to nii
    call( [ Cnt['DCM2NIIX'], '-f', fnii, datain['mumapDCM'] ] )
    #files for the T1w, pick one:
    fmunii = glob.glob( os.path.join(datain['mumapDCM'], '*converted*.nii*') )
    fmunii = fmunii[0]

    # the converted nii image resample to the reference size
    fmu = os.path.join(os.path.dirname(datain['mumapDCM']), comment+'mumap_tmp.nii.gz')
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [ Cnt['RESPATH'],
                    '-ref', fmuref,
                    '-flo', fmunii,
                    '-res', fmu,
                    '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
    else:
        print 'e> path to resampling executable is incorrect!'
        sys.exit()


    nim = nib.load(fmu)
    # get the affine transform
    A = nim.get_sform()
    mu = nim.get_data()
    mu = np.transpose(mu[:,::-1,::-1], (2, 1, 0))
    # convert to mu-values
    mu = np.float32(mu)/1e4
    mu[mu<0] = 0

    # del the temporary file for mumap
    os.remove(fmu)
    os.remove(fmunii)
    
    #return image dictionary with the image itself and some other stats
    mu_dct = {'im':mu,
              'affine':A}

    # store the mu-map if requested (by default no)
    if store:
        # to numpy array
        fnp = os.path.join(fout, 'mumapUTE.npy' )
        np.save(fnp, (mu, A))
        
        # with this file name
        fout = os.path.join(fout, comment+'mumap_fromDCM.nii.gz')
        respet.img.mmrimg.array2nii(mu[::-1, ::-1, :], A, fout)
        mu_dct['fmu'] = fout

    return mu_dct


#=================================================================================
# PSEUDO CT MU-MAP
#---------------------------------------------------------------------------------
def pct_mumap(datain, txLUT, axLUT, Cnt, hst=[], t0=0, t1=0, faff='', fpet='', fcomment='', store=False, petopt='nac'):
    '''
    GET THE MU-MAP from pCT IMAGE (which is in T1w space)
    * the mu-map will be registered to PET which will be reconstructed for time frame t0-t1
    * it f0 and t1 are not given the whole LM dataset will be reconstructed 
    * the reconstructed PET can be attenuation and scatter corrected or NOT using petopt
    '''
    # ----------------------------------
    # get hardware mu-map
    if os.path.isfile(datain['hmumap']):
        muh, _ = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    else:
        hmudic = respet.img.mmrimg.hdw_mumap(datain, [1, 2, 4], Cnt)
        muh = hmudic['im']

    if datain['MRT1W#']==0 and not os.path.isfile(datain['T1nii']) and not os.path.isfile(datain['T1bc']):
        print 'e> no MR T1w images!'
        sys.exit()
    # ----------------------------------

    # histogram the list data if needed
    if not hst and not os.path.isfile(faff):
        hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt, t0=t0, t1=t1)

    if not os.path.isfile(faff):
        # first recon pet to get the T1 aligned to it
        if petopt=='qnt':
            # ---------------------------------------------
            # OPTION 1 (quantitative recon with all corrections using MR-based mu-map)
            # get UTE object mu-map (may not be in register with the PET data)
            mudic = respet.img.mmrimg.obj_mumap(datain, Cnt)
            muo = mudic['im']
            # reconstruct PET image with UTE mu-map to which co-register T1w
            recout = nipet.prj.mmrprj.osemone(datain, [muh, muo], hst, txLUT, axLUT, Cnt, recmod=3, itr=4, fwhm=0., fcomment=fcomment+'_QNT-UTE', store_img=True)
            fpet = recout.fpet
            # do the affine
            faff = respet.img.mmrimg.mr2petAffine(datain, Cnt, fpet, fcomment=fcomment)
        elif petopt=='nac':
            # ---------------------------------------------
            # OPTION 2 (recon without any corrections for scatter and attenuation)
            # reconstruct PET image with UTE mu-map to which co-register T1w
            muo = np.zeros(muh.shape, dtype=muh.dtype)
            recout = nipet.prj.mmrprj.osemone(datain, [muh, muo], hst, txLUT, axLUT, Cnt, recmod=1, itr=3, fwhm=0., fcomment=fcomment+'_NAC', store_img=True)
            fpet = recout.fpet
            # do the affine
            faff = respet.img.mmrimg.mr2petAffine(datain, Cnt, fpet, fcomment=fcomment)

    
    
    # pCT file name
    fpct = os.path.join(os.path.dirname(datain['pCT']), 'pCT_r'+fcomment+'.nii.gz')
    #call the resampling routine to get the pCT in place
    if os.path.isfile( Cnt['RESPATH'] ):
        cmd = [Cnt['RESPATH'],
            '-ref', fpet,
            '-flo', datain['pCT'],
            '-trans', faff,
            '-res', fpct,
            '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
    else:
        print 'e> path to resampling executable is incorrect!'
        sys.exit()


    # get the NIfTI of the pCT
    nim = nib.load(fpct)
    A   = nim.get_sform()
    pct = np.float32( nim.get_data() )
    pct = pct[:,::-1,::-1]
    pct = np.transpose(pct, (2, 1, 0))
    #convert the HU units to mu-values
    mu = respet.img.mmrimg.hu2mu(pct)
    #get rid of negatives
    mu[mu<0] = 0
    
    #return image dictionary with the image itself and some other stats
    mu_dct = {'im':mu,
              'affine':A}

    if store:
        # now save to NIfTI in this folder
        fout = os.path.join( datain['corepath'], 'mumap_obj' )
        nipet.mmraux.create_dir(fout)

        fnp = os.path.join(fout, 'mumapCT.npy')
        np.save(fnp, (mu, A))

        # with this file name
        fout = os.path.join(fout, fcomment+'mumap_frompCT.nii.gz')
        respet.img.mmrimg.array2nii(mu[::-1, ::-1, :], A, fout)
        mu_dct['fmu'] = fout

    return mu_dct


#*********************************************************************************
#GET HARDWARE MU-MAPS with positions and offsets
#---------------------------------------------------------------------------------
def hdr_mu(datain, Cnt):
    '''Get the headers from DICOM data file'''
    #get one of the DICOM files of the mu-map
    dcmf = os.path.join(datain['mumapDCM'], os.listdir(datain['mumapDCM'])[0]) 
    if os.path.isfile( dcmf ):
        dhdr = dcm.read_file( dcmf )
    else:
        print 'e> DICOM mMR mu-maps not found!'
        return None
    # CSA Series Header Info
    if [0x29,0x1020] in dhdr:
        csahdr = dhdr[0x29,0x1020].value
        if Cnt['VERBOSE']: print 'i> got CSA mu-map info.'
    return csahdr, dhdr

def hmu_shape(hdr):
    #regular expression to find the shape
    p = re.compile(r'(?<=:=)\s*\d{1,4}')
    # x: dim [1]
    i0 = hdr.find('matrix size[1]')
    i1 = i0+hdr[i0:].find('\n')
    u = int(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('matrix size[2]')
    i1 = i0+hdr[i0:].find('\n')
    v = int(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('matrix size[3]')
    i1 = i0+hdr[i0:].find('\n')
    w = int(p.findall(hdr[i0:i1])[0])
    return w,v,u

def hmu_voxsize(hdr):
    #regular expression to find the shape
    p = re.compile(r'(?<=:=)\s*\d{1,2}[.]\d{1,10}')
    # x: dim [1]
    i0 = hdr.find('scaling factor (mm/pixel) [1]')
    i1 = i0+hdr[i0:].find('\n')
    vx = float(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('scaling factor (mm/pixel) [2]')
    i1 = i0+hdr[i0:].find('\n')
    vy = float(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('scaling factor (mm/pixel) [3]')
    i1 = i0+hdr[i0:].find('\n')
    vz = float(p.findall(hdr[i0:i1])[0])
    return np.array([0.1*vz, 0.1*vy, 0.1*vx])

def hmu_origin(hdr):
    #regular expression to find the origin
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{1,10}')
    # x: dim [1]
    i0 = hdr.find('$umap origin (pixels) [1]')
    i1 = i0+hdr[i0:].find('\n')
    x = float(p.findall(hdr[i0:i1])[0])
    # x: dim [2]
    i0 = hdr.find('$umap origin (pixels) [2]')
    i1 = i0+hdr[i0:].find('\n')
    y = float(p.findall(hdr[i0:i1])[0])
    # x: dim [3]
    i0 = hdr.find('$umap origin (pixels) [3]')
    i1 = i0+hdr[i0:].find('\n')
    z = -float(p.findall(hdr[i0:i1])[0])
    return np.array([z, y, x])

def hmu_offset(hdr):
    #regular expression to find the origin
    p = re.compile(r'(?<=:=)\s*\d{1,5}[.]\d{1,10}')
    if hdr.find('$origin offset')>0:
        # x: dim [1]
        i0 = hdr.find('$origin offset (mm) [1]')
        i1 = i0+hdr[i0:].find('\n')
        x = float(p.findall(hdr[i0:i1])[0])
        # x: dim [2]
        i0 = hdr.find('$origin offset (mm) [2]')
        i1 = i0+hdr[i0:].find('\n')
        y = float(p.findall(hdr[i0:i1])[0])
        # x: dim [3]
        i0 = hdr.find('$origin offset (mm) [3]')
        i1 = i0+hdr[i0:].find('\n')
        z = -float(p.findall(hdr[i0:i1])[0])
        return np.array([0.1*z, 0.1*y, 0.1*x])
    else:
        return np.array([0.0, 0.0, 0.0])

def rd_hmu(fh):
    #--read hdr file--
    f = open(fh, 'r')
    hdr = f.read()
    f.close()
    #-----------------
    #regular expression to find the file name
    p = re.compile(r'(?<=:=)\s*\w*[.]\w*')
    i0 = hdr.find('!name of data file')
    i1 = i0+hdr[i0:].find('\n')
    fbin = p.findall(hdr[i0:i1])[0]
    #--read img file--
    f = open(os.path.join(os.path.dirname(fh), fbin.strip()), 'rb')
    im = np.fromfile(f, np.float32)
    f.close()
    #-----------------
    return  hdr, im 


def get_hmupos(datain, parts, Cnt):

    # check if registration executable exists
    if not os.path.isfile(Cnt['RESPATH']):
        print 'e> no registration executable found!'
        sys.exit()

    #----- get positions from the DICOM list-mode file -----
    ihdr, csainfo = nipet.mmraux.hdr_lm(datain, Cnt)

    #table position origin
    fi = csainfo.find('TablePositionOrigin')
    tpostr = csainfo[fi:fi+200]
    tpo = re.sub(r'[^a-zA-Z0-9\-\.]', '', tpostr).split('M')
    tpozyx = np.array([float(tpo[-1]), float(tpo[-2]), float(tpo[-3])]) / 10
    if Cnt['VERBOSE']: print 'i> table position (z,y,x) (cm):', tpozyx
    #--------------------------------------------------------


    #------- get positions from the DICOM mu-map file -------
    csamu, dhdr = respet.img.mmrimg.hdr_mu(datain, Cnt)
    tmp = re.search('GantryTableHomeOffset(?!_)', csamu)
    gtostr = csamu[ tmp.start():tmp.start()+300 ]
    gto = re.sub(r'[^a-zA-Z0-9\-\.]', '', gtostr).split('M')
    # get the first three numbers
    zyx = np.zeros(3, dtype=np.float32)
    c = 0
    for i in range(len(gto)):
        if re.search(r'[\d]', gto[i])!=None and c<3:
            zyx[c] = np.float32(re.sub(r'[^0-9\-\.]', '', gto[i]))
            c+=1
    #gantry table offset
    gtozyx = zyx[::-1]/10
    if Cnt['VERBOSE']: print 'i> gantry table offset (z,y,x) (cm):', gtozyx
    #old way:only worked for syngo MR B20P
    # fi = csamu.find('GantryTableHomeOffset') 
    # gtostr =csamu[fi:fi+300]
    # if dhdr[0x0018, 0x1020].value == 'syngo MR B20P':
    #     gto = re.sub(r'[^a-zA-Z0-9\-\.]', '', gtostr).split('M')
    #     # get the first three numbers
    #     zyx = np.zeros(3, dtype=np.float32)
    #     c = 0
    #     for i in range(len(gto)):
    #         if re.search(r'[\d]', gto[i])!=None and c<3:
    #             zyx[c] = np.float32(re.sub(r'[^0-9\-\.]', '', gto[i]))
    #             c+=1
    #     #gantry table offset
    #     gtozyx = zyx[::-1]/10
    #     if Cnt['VERBOSE']: print 'i> gantry table offset (z,y,x) (cm):', gtozyx
    # # older scanner version
    # elif dhdr[0x0018, 0x1020].value == 'syngo MR B18P':
    #     zyx = np.zeros(3, dtype=np.float32)
    #     for k in range(3):
    #         tmp = re.search(r'\{\s*[\-0-9.]*\s*\}', gtostr)
    #         i0 = tmp.start()
    #         i1 = tmp.end()
    #         if gtostr[i0+1:i1-1]!=' ':  zyx[k] = np.float32(gtostr[i0+1:i1-1])
    #         gtostr = gtostr[i1:]
    #     #gantry table offset
    #     gtozyx = zyx[::-1]/10
    #     if Cnt['VERBOSE']: print 'i> gantry table offset (z,y,x) (cm):', gtozyx
    #--------------------------------------------------------

    # create the folder for hardware mu-maps
    dirhmu = os.path.join( datain['corepath'], 'mumap_hdw')
    if not os.path.isdir(dirhmu):
        os.makedirs(dirhmu)

    # get the reference nii image
    fref = os.path.join(dirhmu, 'hmuref.nii.gz')

    #start horizontal bed position
    p = re.compile(r'start horizontal bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    hbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    #start vertical bed position
    p = re.compile(r'start vertical bed position.*\d{1,3}\.*\d*')
    m = p.search(ihdr)
    fi = ihdr[m.start():m.end()].find('=')
    vbedpos = 0.1*float(ihdr[m.start()+fi+1:m.end()])

    if Cnt['VERBOSE']: print 'i> creating reference nii image for resampling'
    B = np.diag(np.array([-10*Cnt['SO_VXX'], 10*Cnt['SO_VXY'], 10*Cnt['SO_VXZ'], 1]))
    B[0,3] = 10*(.5*Cnt['SO_IMX'])*Cnt['SO_VXX']
    B[1,3] = 10*( -.5*Cnt['SO_IMY']+1)*Cnt['SO_VXY']
    B[2,3] = 10*((-.5*Cnt['SO_IMZ']+1)*Cnt['SO_VXZ'] + hbedpos )
    array2nii(  np.zeros((Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32), B, fref)

    #define a dictionary of all positions/offsets of hardware mu-maps
    hmupos = [None]*5
    hmupos[0] = {
        'TabPosOrg' :   tpozyx, #from DICOM of LM file
        'GanTabOff' :   gtozyx, #from DICOM of mMR mu-map file
        'HBedPos'   :   hbedpos, #from Interfile of LM file [cm]
        'VBedPos'   :   vbedpos, #from Interfile of LM file [cm]
        'niipath'   :   fref
        }
        

    #--------------------------------------------------------------------------
    # iteratively go through the mu-maps and add them as needed
    for i in parts:
        
        fh = os.path.join(Cnt['HMUDIR'], Cnt['HMULIST'][i-1])
        # get the interfile header and binary data 
        hdr, im = rd_hmu(fh)
        #get shape, origin, offset and voxel size
        s = hmu_shape(hdr)
        im.shape = s
        # get the origin, offset and voxel size for the mu-map interfile data
        org = hmu_origin(hdr)
        off = hmu_offset(hdr)
        vs = hmu_voxsize(hdr)
        # corner voxel position for the interfile image data
        vpos = (-org*vs + off + gtozyx - tpozyx)

        #add to the dictionary
        hmupos[i] = {
            'vpos'    :   vpos,
            'shape'   :   s,   #from interfile
            'iorg'    :   org, #from interfile
            'ioff'    :   off, #from interfile
            'ivs'     :   vs,  #from interfile
            'img'     :   im, #from interfile
            'niipath' :   os.path.join(dirhmu, Cnt['HMULIST'][i-1].split('.')[0]+'.nii.gz')
        }

        #save to NIfTI
        if Cnt['VERBOSE']: print 'i> creating mu-map for:', Cnt['HMULIST'][i-1]
        A = np.diag(np.append(10*vs[::-1], 1))
        A[0,0] *= -1
        A[0,3] =  10*(-vpos[2])
        A[1,3] = -10*((s[1]-1)*vs[1] + vpos[1])
        A[2,3] = -10*((s[0]-1)*vs[0] - vpos[0])
        array2nii(im[::-1,::-1,:], A, hmupos[i]['niipath'])

        # resample using nify.reg
        fout = os.path.join(    os.path.dirname (hmupos[0]['niipath']),
                                os.path.basename(hmupos[i]['niipath']).split('.')[0]+'_r.nii.gz' )
        
        cmd = [ Cnt['RESPATH'],
                '-ref', hmupos[0]['niipath'],
                '-flo', hmupos[i]['niipath'],
                '-res', fout,
                '-pad', '0']
        if not Cnt['VERBOSE']: cmd.append('-voff')
        call(cmd)
                
    return hmupos


#-------------------------------------------------------------------------------------
def hdw_mumap(datain, hparts, Cnt, use_stored=False):

    # if requested to use the stored hardware mu_map get it from the path in datain
    if os.path.isfile(datain['hmumap']) and use_stored:
        hmu, A = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    # otherwise generate it from the parts through resampling the high resolution CT images
    else:
        hmupos = get_hmupos(datain, hparts, Cnt)

        #just to get the dims, get the ref image
        nimo = nib.load(hmupos[0]['niipath'])
        A = nimo.get_sform()
        imo = np.float32( nimo.get_data() )
        imo[:] = 0

        for i in hparts:
            fin  = os.path.join(    os.path.dirname (hmupos[0]['niipath']),
                                    os.path.basename(hmupos[i]['niipath']).split('.')[0]+'_r.nii.gz' )

            nim = nib.load(fin)
            mu = nim.get_data()
            mu[mu<0] = 0
            
            imo += mu

        hdr = nimo.header
        hdr['cal_max'] = np.max(imo)
        hdr['cal_min'] = np.min(imo)
        fout  = os.path.join(os.path.dirname (hmupos[0]['niipath']), 'hardware_umap.nii.gz' )
        nib.save(nimo, fout)

        hmu = np.transpose(imo[:,::-1,::-1], (2, 1, 0))

        # save the objects to numpy arrays
        fnp = os.path.join( datain['corepath'], 'mumap_hdw' )
        fnp = os.path.join(fnp, 'hmumap.npy')
        np.save(fnp, (hmu, A))
        #update the datain dictionary (assuming it is mutable)
        datain = nipet.mmraux.explore_input(datain['corepath'], Cnt)

    #return image dictionary with the image itself and some other stats
    hmu_dct = {'im':hmu,
               'affine':A}


    return hmu_dct


def rmumaps(datain, Cnt, t0=0, t1=0, use_stored=False):
    '''
    get the mu-maps for hardware and object and trim it axially for reduced rings case
    '''

    fcomment = '(R)'

    # get hardware mu-map
    if os.path.isfile(datain['hmumap']) and use_stored:
        muh, _ = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    else:
        hmudic = respet.img.mmrimg.hdw_mumap(datain, [1, 2, 4], Cnt)
        muh = hmudic['im']

    # get pCT mu-map if stored in numpy file and then exit, otherwise do all the processing
    if os.path.isfile(datain['mumapCT']) and use_stored:
        mup, _ = np.load(datain['mumapCT'])
        muh = muh[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        mup = mup[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        return [muh, mup]

    # get UTE object mu-map (may be not in register with the PET data)
    if os.path.isfile(datain['mumapUTE']) and use_stored:
        muo, _ = np.load(datain['mumapUTE'])
    else:
        mudic = respet.img.mmrimg.obj_mumap(datain, Cnt, store=True)
        muo = mudic['im']

    if os.path.isfile(datain['pCT']):
        # reconstruct PET image with default settings to be used to alight pCT mu-map
        Cnt_, txLUT_, axLUT_ = nipet.mmraux.mmrinit()
        # histogram for reconstruction with UTE mu-map
        hst = nipet.lm.mmrhist.hist(datain, txLUT_, axLUT_, Cnt_, t0=t0, t1=t1)
        # reconstruct PET image with UTE mu-map to which co-register T1w
        recute = nipet.prj.mmrprj.osemone( datain, [muh, muo], hst, txLUT_, axLUT_, Cnt_, 
                                            recmod=3, itr=4, fwhm=0., store_img=True, fcomment=fcomment+'_QNT-UTE')
        # --- MR T1w
        if os.path.isfile(datain['T1nii']):
            ft1w = datain['T1nii']
        elif os.path.isfile(datain['T1bc']):
            ft1w = datain['T1bc']
        elif os.path.isdir(datain['MRT1W']):
            # create file name for the converted NIfTI image
            fnii = 'converted'
            call( [ Cnt['DCM2NIIX'], '-f', fnii, datain['T1nii'] ] )
            ft1nii = glob.glob( os.path.join(datain['T1nii'], '*converted*.nii*') )
            ft1w = ft1nii[0]
        else:
            print 'e> disaster: no T1w image!'
            sys.exit()

        #output for the T1w in register with PET
        ft1out = os.path.join(os.path.dirname(ft1w), 'T1w_r'+'.nii.gz')
        #text file fo rthe affine transform T1w->PET
        faff   = os.path.join(os.path.dirname(ft1w), fcomment+'mr2pet_affine'+'.txt')  #time.strftime('%d%b%y_%H.%M',time.gmtime())
        #call the registration routine
        if os.path.isfile( Cnt['REGPATH'] ):
            cmd = [Cnt['REGPATH'],
                 '-ref', recute.fpet,
                 '-flo', ft1w,
                 '-rigOnly', '-speeeeed',
                 '-aff', faff,
                 '-res', ft1out]
            if not Cnt['VERBOSE']: cmd.append('-voff')
            call(cmd)
        else:
            print 'e> path to registration executable is incorrect!'
            sys.exit()

        #get the pCT mu-map with the above faff
        pmudic = respet.img.mmrimg.pct_mumap(datain, txLUT, axLUT, Cnt, faff=faff, fpet=recute.fpet, fcomment=fcomment)
        mup = pmudic['im']

        muh = muh[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        mup = mup[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        return [muh, mup]
    else:
        muh = muh[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        muo = muo[2*Cnt['RNG_STRT'] : 2*Cnt['RNG_END'], :, :]
        return [muh, muo]

#------------------------------------------------------
def get_mumaps(datain, Cnt, hst={}, hparts=[1,2,4], t0=0, t1=0, use_stored = False, fcomment=''):
    '''
    GET ALL THE MU-MAPS, hardware and object/patient
    tstrat, t1 are the time points of frame to which the mu-map will be registered
    hparts are the parts of the hardware mu-maps being included.  for more check Cnt dictionary
    '''

    # time it
    stime = time.time()

    # get default constatnts and LUTs 
    Cnt_, txLUT, axLUT = nipet.mmraux.mmrinit()
    Cnt_['VERBOSE'] = Cnt['VERBOSE']

    # hardware first
    if os.path.isfile(datain['hmumap']):
        hmu, hA = np.load(datain['hmumap'])
        if Cnt['VERBOSE']: print 'i> loaded hardware mu-map from file:', datain['hmumap']
    else:
        # get the hardware my-map through resampling and store it in a numpy array
        hmud = respet.img.mmrimg.hdw_mumap(datain, hparts, Cnt_)
        hmu = hmud['im']
        hA = hmud['affine']

    #now the object/patient
    r = re.compile(r'CT') #regular expression to check if pCT-mumaps are sotred
    flg_mu = True #flag: assumes object mu-map found
    if os.path.isfile(datain['mumap']) and use_stored_mu:
        if (r.search(datain['mumap']) is None) and os.path.isfile(datain['pCT']):
            if Cnt['VERBOSE']: print 'i> will replace the numpy object/patient mumap with mumapCT'
            #you can pick the start time and end time for the list mode data acquistion to which the mu-map will be aligned
            if not hst:
                hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt_, t0=t0, t1=t1, use_stored=False)
            elif hst['psino'].shape[0]!=Cnt_['NSN11']:
                hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt_, t0=t0, t1=t1, use_stored=False)
            umap = respet.img.mmrimg.pct_mumap(datain, txLUT, axLUT, Cnt_, hst=hst, fcomment=fcomment)
            omu = umap['im']
        else:
            omu, A = np.load(datain['mumap'])
        
    elif os.path.isfile(datain['pCT']):
        #you can pick the start time and end time for the list mode data acquisition to which the mu-map will be aligned
        if not hst:
            hst = nipet.lm.mmrhist.hist(datain, txLUT, axLUT, Cnt_, t0=t0, t1=t1)
        umap = respet.img.mmrimg.pct_mumap(datain, txLUT, axLUT, Cnt_, hst=hst, fcomment=fcomment)
        omu = umap['im']

    elif os.path.isdir(datain['mumapDCM']):
        # get the object mu-map through resampling and store it in a numpy array for future uses
        umap = respet.img.mmrimg.obj_mumap(datain, Cnt_)
        omu = umap['im']
    else:
        print 'w> no object mu-map found!'
        flg_mu = False


    if flg_mu:
        # combined mu-map at full size, object mu-map full size, and object mu-map GPU size
        mumaps = (hmu, omu)
    else:
        # get the mumaps just for the bed
        mumaps =    ()

    if Cnt['VERBOSE']: print 'i> mu-map done in:', (time.time() - stime), 'seconds.'

    return mumaps
    #------------------------------------------------------
