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
    use_stored = True
    verbose = True

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

    def createStaticNAC(self):
        times = np.int_([0,0])
        return self.createDynamic(times, self.muTiny())

    def createDynamicNAC(self):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        print(times)
        return self.createDynamic(times, self.muTiny())

    def createStaticUTE(self, fcomment=''):
        times = np.int_([0,0])
        return self.createDynamic(times, self.muUTE(), fcomment)

    def createDynamicUTE(self, fcomment=''):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        print(times)
        return self.createDynamic(times, self.muUTE(), fcomment)

    def createDynamic(self, times, muo, fcomment='_createDynamic'):
        """
        :param times:  np.int_
        :param muo:    mu-map of imaged object
        :return:       dictionary from nipet.prj.mmrprj.osemone
        """
        import nipet
        self._constants['VERBOSE'] = self.verbose
        mumaps = [self.muHardware(), muo]
        dyn = (times.shape[0]-1)*[None]
        for it in np.arange(1, times.shape[0]):
            hst = nipet.lm.mmrhist.hist(self._datain,
                                        self._txLUT, self._axLUT, self._constants,
                                        t0=times[it-1], t1=times[it],
                                        store=True, use_stored=True)
            dyn[it-1] = nipet.prj.mmrprj.osemone(self._datain, mumaps, hst,
                                                 self._txLUT, self._axLUT, self._constants,
                                                 recmod = self.recmod,
                                                 itr    = self.itr,
                                                 fwhm   = self.fwhm,
                                                 mask_radious = self.maskRadius,
                                                 store_img=False, ret_sct=True, fcomment='_time'+str(it-1))
        self.save(dyn, mumaps, hst, fcomment)
        return dyn

    def save(self, dyn, mumaps, hst, fcomment=''):
        """
        :param dyn:       dictionary from nipet.prj.mmrprj.osemone
        :param mumaps:    dictionary of mu-maps from imaged object, hardware
        :param hst:       dictionary from nipet.lm.mmrhist.hist
        :param fcomment:  string to append to canonical filename
        """
        import nipet
        fout = self._createFilename(fcomment)
        im = self._gatherOsemoneList(dyn)
        if self._constants['VERBOSE']:
            print('i> saving '+str(len(im.shape))+'D image to: ', fout)

        A = self.getAffine()
        muo,muh = mumaps  # object and hardware mu-maps
        desc = self._createDescrip(hst, muh, muo)
        if len(im.shape) == 3:
            nipet.img.mmrimg.array2nii(im[::-1,::-1,:],   A, fout, descrip=desc)
        elif len(im.shape) == 4:
            nipet.img.mmrimg.array4D2nii(im[::-1,::-1,:,:], A, fout, descrip=desc)

    def _gatherOsemoneList(self, olist):
        """
        :param olist:  list
        :return:       numpy.array
        """
        im = [olist[0].im]
        for i in range(1, len(olist)):
            im = np.append(im, [olist[i].im], axis=0)
        return np.float_(im)

    def getAffine(self):
        import nipet
        cnt = self._constants
        vbed, hbed = nipet.mmraux.vh_bedpos(self._datain, cnt)  # bed positions

        # affine transformations for NIfTI
        A      = np.diag(np.array([-10*cnt['SO_VXX'], 10*cnt['SO_VXY'], 10*cnt['SO_VXZ'], 1]))
        A[0,3] = 10*(  0.5*cnt['SO_IMX']     *cnt['SO_VXX'])
        A[1,3] = 10*((-0.5*cnt['SO_IMY'] + 1)*cnt['SO_VXY'])
        A[2,3] = 10*((-0.5*cnt['SO_IMZ'] + 1)*cnt['SO_VXZ'] + hbed)
        return A

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
        t = t[t <= self.getTimeMax()]
        return np.int_(t)

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
        hmudic = nipet.img.mmrimg.hdw_mumap(self._datain, self.hmuSelection, self._constants, use_stored=self.use_stored)
        return hmudic['im']

    def muTiny(self):
        """
        :return:  mu-map image of mu == 0.01 sized according to self._constants['SO_IM*']
        """
        return 0.01*np.ones((self._constants['SO_IMZ'], self._constants['SO_IMY'], self._constants['SO_IMX']), dtype=np.float32)

    def muUTE(self):
        """
        :return:  mu-map image from Siemens UTE
        :rtype:  numpy.array
        """
        import nipet
        ute = nipet.img.mmrimg.obj_mumap(self._datain, self._constants, store=True)
        im  = ute['im']
        return im

    def muCarney(self, fileprefix='', fcomment=''):
        """is a derivative of nipet.img.mmrimg.pct_mumap from Pawel Markiewicz' NiftyPETx"""
        import nibabel

        # get the NIfTI of the custom umap
        nim = nibabel.load(
            os.path.join(self.tracerRawdataLocation, fileprefix + fcomment + '.nii.gz'))
        cmu = np.float32(nim.get_data())
        cmu = cmu[::-1, ::-1, ::-1]
        cmu = np.transpose(cmu, (2, 1, 0))
        mu = cmu
        mu[mu < 0] = 0
        return mu

    def __init__(self, loc):
        """:param:  loc specifies the location of tracer rawdata.
           :param:  self.tracerRawdataLocation contains Siemens sinograms, e.g.:
                  -rwxr-xr-x+  1 jjlee wheel   16814660 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.2016090913012239062507614.bf
                  -rwxr-xr-x+  1 jjlee wheel     141444 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.2016090913012239062507614.dcm
                  -rwxr-xr-x+  1 jjlee wheel  247141283 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000050.bf
                  -rwxr-xr-x+  1 jjlee wheel     151868 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000050.dcm
                  -rw-r--r--+  1 jjlee wheel    3081280 Nov 14 14:53 umapSynth_full_frame0.nii.gz
           :param:  self.tracerRawdataLocation also contains folders:
                  norm, containing, e.g.:
                        -rwxr-xr-x+  1 jjlee wheel 323404 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000048.bf
                        -rwxr-xr-x+  1 jjlee wheel 143938 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000048.dcm
                  LM, containing, e.g.:
                          -rwxr-xr-x+  1 jjlee wheel 6817490860 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.bf
                          -rwxr-xr-x+  1 jjlee wheel     145290 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.dcm"""

        os.chdir(loc)
        self.tracerRawdataLocation = loc
        self._organizeRawdataLocation()
        self._mmrinit()

    ########## PRIVATE ##########

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
        print("########## respet.recon.reconstruction.Reconstruction._mmrinit ##########")
        print(self._constants)
        print(self._datain)

    def _tracer(self):
        import re
        return re.split('_', os.path.basename(self.tracerRawdataLocation))[0]

    def _createDescrip(self, hst, muh, muo):
        """
        :param hst:  from nipet.lm.mmrhist.hist
        :param muh:  from mumaps dictionary
        :param muo:  from mumaps dictionary
        :return:     description text for NIfTI
        if only bed present, attnum := 0.5
        """
        import nipet
        cnt    = self._constants
        attnum = (1 * (np.sum(muh) > 0.5) + 1 * (np.sum(muo) > 0.5)) / 2.
        ncmp,_ = nipet.mmrnorm.get_components(self._datain, cnt)
        rilut  = self._riLUT()
        qf     = ncmp['qf'] / rilut[cnt['ISOTOPE']]['BF'] / float(hst['dur'])
        desc   = 'alg=osem' + \
                 ';sub=14' + \
                 ';att='   + str(attnum * (self.recmod > 0)) + \
                 ';sct='   + str(1 * (self.recmod > 1)) + \
                 ';spn='   + str(cnt['SPN']) + \
                 ';itr='   + str(self.itr)   + \
                 ';fwhm='  + str(self.fwhm)  + \
                 ';t0='    + str(hst['t0'])  + \
                 ';t1='    + str(hst['t1'])  + \
                 ';dur='   + str(hst['dur']) + \
                 ';qf='    + str(qf)
        return desc

    def _createFilename(self, fcomment):
        import nipet
        fn = os.path.join(self._datain['corepath'], 'img')
        nipet.mmraux.create_dir(fn)
        fn = os.path.join(fn, os.path.basename(self._datain['lm_dcm'])[:8] + fcomment + '.nii.gz')
        return fn

    def _riLUT(self):
        """
        :return:  radioisotope look-up table
        """
        return {'Ge68':{'BF':0.891, 'thalf':270.9516*24*60*60},
                'Ga68':{'BF':0.891, 'thalf':67.71*60},
                 'F18':{'BF':0.967, 'thalf':109.77120*60},
                 'C11':{'BF':0.998, 'thalf':20.38*60}}

