import numpy as np
import os

class Reconstruction:
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2018"

    span = 11
    bootstrap = 2
    recmod = 3
    itr = 5
    fwhm = 4.0
    maskRadius = 29
    hmuSelection = [1,2,4] # selects from ~/.niftypet/resources.py:  hrdwr_mu
    tracerRawdataLocation = ''
    umapFolder = 'umap'
    umapSynthFileprefix = 'umapSynth_full_frame'
    frameSuffix = '_frame'
    verbose = True

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

    def createNAC(self):
        import nipet
        self._constants['VERBOSE'] = self.verbose
        hst = nipet.lm.mmrhist.hist(self._datain,
                                    self._txLUT, self._axLUT, self._constants,
                                    t0=self._t0, t1=self._t1,
                                    store=True, use_stored=True)
        mumaps = [self.muHardware(), self.muZero()]
        nac = nipet.prj.mmrprj.osemone(self._datain, mumaps, hst,
                                       self._txLUT, self._axLUT, self._constants,
                                       recmod = self.recmod,
                                       itr    = self.itr,
                                       fwhm   = self.fwhm,
                                       mask_radious = self.maskRadius,
                                       store_img=True, ret_sct=True)
        return nac

    def createDynamicNAC(self):
        import nipet
        self._constants['VERBOSE'] = self.verbose
        times  = self.getTimes()
        nac = []
        mumaps = [self.muHardware(), self.muZero()]
        for it in np.arange(1, times.shape[0]):
            hst = nipet.lm.mmrhist.hist(self._datain,
                                        self._txLUT, self._axLUT, self._constants,
                                        t0=times[it-1], t1=times[it],
                                        store=True, use_stored=True)
            nac[it-1] = nipet.prj.mmrprj.osemone(self._datain, mumaps, hst,
                                                self._txLUT, self._axLUT, self._constants,
                                                recmod = self.recmod,
                                                itr    = self.itr,
                                                fwhm   = self.fwhm,
                                                mask_radious = self.maskRadius,
                                                store_img=True, ret_sct=True)
        return nac

    def getTaus(self):
        """
        :return:  1x65 array of frame durations:  10 x 30-sec-frames + 55 x 60-sec-frames
        :rtype:  numpy.int_
        """
        return np.int_([30,30,30,30,30,30,30,30,30,30,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60])

    def getTimes(self):
        """
        :return:  up to 1x66 array of times including 0 and np.cumsum(taus); max(times) <= self.getTimeMax
        :rtype:  numpy.int_
        """
        taus = self.getTaus()
        t = np.hstack((np.int_(0), np.cumsum(taus)))
        return t[t <= self.getTimeMax()]

    def getTimeMax(self):
        """
        :return:  max time available from listmode data in sec.
        """
        from nipet.lm import mmr_lmproc
        nele, ttags, tpos = mmr_lmproc.lminfo(self._datain['lm_bf'])
        return (ttags[1]-ttags[0]+999)/1000 # sec

    def muHardware(self):
        """
        :return:  hardware mu-map image provided by nipet.img.mmrimg.hdw_mumap.
        See also self.hmuSelection.
        """
        import nipet
        hmudic = nipet.img.mmrimg.hdw_mumap(self._datain, self.hmuSelection, self._constants)
        return hmudic['im']

    def muZero(self):
        """
        :return:  mu-map image of zeroes sized according to self._constants['SO_IM*']
        """
        return np.zeros((self._constants['SO_IMZ'], self._constants['SO_IMY'], self._constants['SO_IMX']), dtype=np.float32)

    # def custom_mumap(self, datain, fileprefix='', fcomment='', store=False):
    #     """is a derivative of nipet.img.mmrimg.pct_mumap from Pawel Markiewicz' NiftyPETx"""
    #     import numpy
    #     import os
    #     import nibabel
    #
    #     # get the NIfTI of the custom umap
    #     nim = nibabel.load(
    #         os.path.join(self.tracerRawdataLocation, fileprefix + fcomment + '.nii.gz'))
    #     cmu = numpy.float32(nim.get_data())
    #     cmu = cmu[::-1, ::-1, ::-1]
    #     cmu = numpy.transpose(cmu, (2, 1, 0))
    #     mu = cmu
    #     mu[mu < 0] = 0
    #
    #     return mu

    def __init__(self, loc):
        """@param loc specifies the location of tracer rawdata.
           @param self.tracerRawdataLocation contains Siemens sinograms, e.g.:
                  -rwxr-xr-x+  1 jjlee wheel   16814660 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.2016090913012239062507614.bf
                  -rwxr-xr-x+  1 jjlee wheel     141444 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.2016090913012239062507614.dcm
                  -rwxr-xr-x+  1 jjlee wheel  247141283 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000050.bf
                  -rwxr-xr-x+  1 jjlee wheel     151868 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000050.dcm
                  -rw-r--r--+  1 jjlee wheel    3081280 Nov 14 14:53 umapSynth_full_frame0.nii.gz
           as well as folders:
                  norm, containing, e.g.:
                        -rwxr-xr-x+  1 jjlee wheel 323404 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000048.bf
                        -rwxr-xr-x+  1 jjlee wheel 143938 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000048.dcm
                  mumap_hdw, containing:
                          -rw-r--r--+  1 jjlee wheel  20648135 Oct 31 14:40 hardware_umap.nii.gz
                          -rw-r--r--+  1 jjlee wheel  60115163 Oct 31 14:40 hmumap.npy
                          -rw-r--r--+  1 jjlee wheel    262479 Oct 31 14:37 hmuref.nii.gz
                          -rw-r--r--+  1 jjlee wheel 158313150 Oct 31 14:38 umap_HNMCL_10606489.nii.gz
                          -rw-r--r--+  1 jjlee wheel   6883547 Oct 31 14:38 umap_HNMCL_10606489_r.nii.gz
                          -rw-r--r--+  1 jjlee wheel 254435655 Oct 31 14:38 umap_HNMCU_10606489.nii.gz
                          -rw-r--r--+  1 jjlee wheel  10634278 Oct 31 14:38 umap_HNMCU_10606489_r.nii.gz
                          -rw-r--r--+  1 jjlee wheel 479244202 Oct 31 14:40 umap_PT_2291734.nii.gz
                          -rw-r--r--+  1 jjlee wheel  13624762 Oct 31 14:40 umap_PT_2291734_r.nii.gz
                  LM, containing, e.g.:
                          -rwxr-xr-x+  1 jjlee wheel 6817490860 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.bf
                          -rwxr-xr-x+  1 jjlee wheel     145290 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.dcm"""

        self.tracerRawdataLocation = loc
        self._organizeRawdataLocation()
        self._mmrinit()

    ### PRIVATE

    _constants = {}
    _txLUT = {}
    _axLUT = {}
    _datain = {}
    _frame = 0
    _umapIdx = 0
    _t0 = 0
    _t1 = 0

    def _organizeRawdataLocation(self):
        import glob
        import dicom
        try:
            fns = glob.glob(os.path.join(self.tracerRawdataLocation, '*.dcm'))
            for fn in fns:
                ds = dicom.read_file(fn)
                if ds.ImageType[2] == 'PET_NORM':
                    self._moveToNamedLocation(fn, 'norm')
                if ds.ImageType[2] == 'PET_LISTMODE':
                    self._moveToNamedLocation(fn, 'LM')
        except OSError:
            os.listdir(self.tracerRawdataLocation)
            raise

    def _moveToNamedLocation(self, dcm, name):
        import shutil
        import errno
        namedLoc = os.path.join(self.tracerRawdataLocation, name)
        if not os.path.exists(namedLoc):
            os.makedirs(namedLoc)
        try:
            bf = os.path.splitext(dcm)[0]+'.bf'
            shutil.move(bf,  os.path.join(namedLoc, os.path.basename(bf)))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            shutil.move(dcm, os.path.join(namedLoc, os.path.basename(dcm)))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def _mmrinit(self):
        import nipet
        c,tx,ax = nipet.mmraux.mmrinit()
        self._constants = c
        self._constants['SPN'] = self.span
        self._constants['BTP'] = self.bootstrap
        self._txLUT = tx
        self._axLUT = ax
        d = nipet.mmraux.explore_input(self.tracerRawdataLocation, self._constants)
        self._datain = d

    def _tracer(self):
        import re
        return re.split('_', os.path.basename(self.tracerRawdataLocation))[0]

