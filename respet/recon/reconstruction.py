import numpy as np
import os
import respet.recon

UMAP_SYNTH_FILEPREFIX = 'umapSynthFull'

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
    use_stored = True
    verbose = True

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

    def createStaticNAC(self, fcomment=''):
        self.recmod = 1
        self.itr = 3
        self.fwhm = 0
        return self.createStatic(self.muNAC(), fcomment)

    def createStaticUTE(self, fcomment=''):
        return self.createStatic(self.muUTE(), fcomment)

    def createStaticCarney(self, fcomment=''):
        return self.createStatic(self.muCarney(frames=[0]), fcomment)

    def createDynamicNAC(self, fcomment='_createDynamicNAC'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        print(times)
        self.recmod = 1
        self.itr = 3
        self.fwhm = 0
        return self.createDynamic(times, self.muNAC(), fcomment)

    def createDynamicTiny(self, fcomment='_createDynamicTiny'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicTiny ##########")
        print(times)
        self.recmod = 1
        self.itr = 3
        self.fwhm = 0
        return self.createDynamic(times, self.muTiny(), fcomment)

    def createDynamicUTE(self, fcomment='_createDynamicUTE'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        print(times)
        return self.createDynamic(times, self.muUTE(), fcomment)

    def createDynamic2Carney(self, fcomment='_createDynamicCarney'):
        times = self.getTimes()
        times2 = self.getTimes2()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamic2Carney ##########")
        print(times)
        print(times2)
        return self.createDynamic2(times, times2, self.muCarney(), fcomment)

    def createStatic(self, muo, fcomment='_createStatic'):
        """
        :param muo:       mu-map of imaged object
        :param fcomment;  string for naming subspace
        :return:          result from nipet.prj.mmrprj.osemone
        :rtype:           dictionary
        """
        import nipet
        self._constants['VERBOSE'] = self.verbose
        mumaps = [muo, self.muHardware()]
        hst = nipet.lm.mmrhist.hist(self._datain,
                                    self._txLUT, self._axLUT, self._constants,
                                    t0=0, t1=0,
                                    store=True, use_stored=True)
        sta = nipet.prj.mmrprj.osemone(self._datain, mumaps, hst,
                                       self._txLUT, self._axLUT, self._constants,
                                       recmod = self.recmod,
                                       itr    = self.itr,
                                       fwhm   = self.fwhm,
                                       mask_radious = self.maskRadius,
                                       store_img=False, fcomment=fcomment)
        self.saveStatic(sta, mumaps, hst, fcomment)
        return sta

    def saveStatic(self, sta, mumaps, hst, fcomment=''):
        """
        :param sta:       dictionary from nipet.prj.mmrprj.osemone
        :param mumaps:    dictionary of mu-maps from imaged object, hardware
        :param hst:       dictionary from nipet.lm.mmrhist.hist
        :param fcomment:  string to append to canonical filename
        """
        import nipet
        fout = self._createFilename(fcomment)
        im = sta.im
        if self._constants['VERBOSE']:
            print('i> saving 3D image to: ', fout)

        A = self.getAffine()
        muo,muh = mumaps  # object and hardware mu-maps
        desc = self._createDescrip(hst, muh, muo)
        assert len(im.shape) == 3, "Reconstruction.saveStatic.im.shape == " + str(len(im.shape))
        nipet.img.mmrimg.array2nii(im[::-1,::-1,:], A, fout, descrip=desc)

    def createDynamic(self, times, muo, fcomment='_createDynamic'):
        """
        :param times:  np.int_; [0,0] produces a single time-frame
        :param muo:    3D or 4D mu-map of imaged object
        :return:       last result from nipet.prj.mmrprj.osemone
        :rtype:        dictionary
        """
        import nipet
        self._constants['VERBOSE'] = self.verbose
        for it in np.arange(1, times.shape[0]):
            hst = nipet.lm.mmrhist.hist(self._datain,
                                        self._txLUT, self._axLUT, self._constants,
                                        t0=times[it-1], t1=times[it],
                                        store=True, use_stored=True)
            dynFrame = nipet.prj.mmrprj.osemone(self._datain,
                                                self.getMumaps(muo, it-1),
                                                hst,
                                                self._txLUT, self._axLUT, self._constants,
                                                recmod = self.recmod,
                                                itr    = self.itr,
                                                fwhm   = self.fwhm,
                                                mask_radious = self.maskRadius,
                                                store_img=True,
                                                ret_sct=True,
                                                fcomment=fcomment + '_time' + str(it - 1))
        return dynFrame

    def createDynamic2(self, times, times2, muo, fcomment='_createDynamic2'):
        """
        :param times:   np.int_; [0,0] produces a single time-frame
        :param times2:  np.int_
        :param muo:     3D or 4D mu-map of imaged object
        :return:        last result from nipet.prj.mmrprj.osemone
        :rtype:         dictionary
        """
        import nipet
        self._constants['VERBOSE'] = self.verbose
        it = 1                                     # mu-map frame
        for it2 in np.arange(1, times2.shape[0]):  # hist frame
            hst = nipet.lm.mmrhist.hist(self._datain,
                                        self._txLUT, self._axLUT, self._constants,
                                        t0=times2[it2-1], t1=times2[it2],
                                        store=True, use_stored=True)
            if times2[it2-1] >= times[it]:
                it = it + 1
            dynFrame = nipet.prj.mmrprj.osemone(self._datain,
                                                self.getMumaps(muo, it-1),
                                                hst,
                                                self._txLUT, self._axLUT, self._constants,
                                                recmod = self.recmod,
                                                itr    = self.itr,
                                                fwhm   = self.fwhm,
                                                mask_radious = self.maskRadius,
                                                store_img=True,
                                                ret_sct=True,
                                                fcomment=fcomment + '_time' + str(it2 - 1))
        return dynFrame

    def createDynamicInMemory(self, times, muo, fcomment='_createDynamic'):
        """
        within unittest environment, may use ~60 GB memory for 60 min FDG recon with MRAC
        :param times:  np.int_; [0,0] produces a single time-frame
        :param muo:    3D or 4D mu-map of imaged object
        :return:       result from nipet.prj.mmrprj.osemone
        :rtype:        dictionary
        """
        import nipet
        self._constants['VERBOSE'] = self.verbose
        dyn = (times.shape[0]-1)*[None]
        for it in np.arange(1, times.shape[0]):
            hst = nipet.lm.mmrhist.hist(self._datain,
                                        self._txLUT, self._axLUT, self._constants,
                                        t0=times[it-1], t1=times[it],
                                        store=True, use_stored=True)
            dyn[it-1] = nipet.prj.mmrprj.osemone(self._datain,
                                                 self.getMumaps(muo, it-1),
                                                 hst,
                                                 self._txLUT, self._axLUT, self._constants,
                                                 recmod = self.recmod,
                                                 itr    = self.itr,
                                                 fwhm   = self.fwhm,
                                                 mask_radious = self.maskRadius,
                                                 store_img=False,
                                                 ret_sct=True,
                                                 fcomment=fcomment + '_time' + str(it - 1))
        self.saveDynamic(dyn, self.getMumaps(muo), hst, fcomment)
        return dyn

    def saveDynamic(self, dyn, mumaps, hst, fcomment=''):
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
            nipet.img.mmrimg.array2nii(im[::-1,::-1,:],     A, fout, descrip=desc)
        elif len(im.shape) == 4:
            nipet.img.mmrimg.array4D2nii(im[:,::-1,::-1,:], A, fout, descrip=desc)

    def checkTimeAliasingUTE(self, fcomment='_checkTimeAliasingUTE'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeAliasingUTE ##########")
        print(times[0:2])
        return self.createDynamicInMemory(times[0:3], self.muUTE(), fcomment)

    def checkTimeAliasingCarney(self, fcomment='_checkTimeAliasingCarney'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeAliasingCarney ##########")
        print(times[0:2])
        return self.createDynamic(times[0:3], self.muCarney(frames=[0,1]), fcomment)

    def checkTimeHierarchiesCarney(self, fcomment='_checkTimeHierarchiesCarney'):
        times = self.getTimes()
        times2 = self.getTimes(self.getTaus2())
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeHierarchiesCarney ##########")
        print(times)
        return self.createDynamic2(times[0:3], times2[0:7], self.muCarney(frames=[0,1]), fcomment)

    def getAffine(self):
        """
        :return:  affine transformations for NIfTI
        :rtype:   list 2D numeric
        """
        import nipet
        cnt = self._constants
        vbed, hbed = nipet.mmraux.vh_bedpos(self._datain, cnt)  # bed positions

        A      = np.diag(np.array([-10*cnt['SO_VXX'], 10*cnt['SO_VXY'], 10*cnt['SO_VXZ'], 1]))
        A[0,3] = 10*(  0.5*cnt['SO_IMX']     *cnt['SO_VXX'])
        A[1,3] = 10*((-0.5*cnt['SO_IMY'] + 1)*cnt['SO_VXY'])
        A[2,3] = 10*((-0.5*cnt['SO_IMZ'] + 1)*cnt['SO_VXZ'] + hbed)
        return A

    def getMumaps(self, muo, it = 0):
        """
        :param muo:  numpy.array of len == 3 or len == 4
        :param it:   list, default is empty
        :return:     list of numpy.array := [mu-hardware, mu-object]
        """
        if muo.ndim == 4:
            return [np.squeeze(muo[it,:,:,:]), self.muHardware()]
        else:
            return [muo, self.muHardware()]

    @staticmethod
    def getTaus():
        """
        :return:  1x65 array of frame durations:  10 x 30-sec-frames + 55 x 60-sec-frames
        :rtype:  numpy.int_
        """
        return np.int_([30,30,30,30,30,30,30,30,30,30,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60])

    @staticmethod
    def getTaus2():
        """
        :return:  1x65 array of frame durations:  12 x 10-sec frames + 6 x 30-sec-frames + 55 x 60-sec-frames
        :rtype:  numpy.int_
        """
        return np.int_([10,10,10,10,10,10,10,10,10,10,10,10,30,30,30,30,30,30,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60])

    def getTimes(self, taus=None):
        """
        :return:  up to 1x66 array of times including 0 and np.cumsum(taus); max(times) <= self.getTimeMax
        :rtype:  numpy.int_
        """
        if taus is None:
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

    def muCarney(self, fileprefix=UMAP_SYNTH_FILEPREFIX, fcomment='', frames=None):
        """
        get NIfTI of the custom umap; see also img.mmrimg.obj_mumap lines 751-758
        :param fileprefix:  string for fileprefix of 4D image-object
        :param fcomment:  string to append to fileprefix
        :param frames:  frame indices to select from mu;  default selects all frames
        :return:  np.float32
        """
        import nibabel
        nim = nibabel.load(os.path.join(self.tracerRawdataLocation, fileprefix + fcomment + '.nii.gz'))
        mu = np.float32(nim.get_data())
        if frames is None:
            mu = np.transpose(mu[:,::-1,::-1,:], (3, 2, 1, 0))
        else:
            mu = np.transpose(mu[:,::-1,::-1,frames], (3, 2, 1, 0))
        mu = np.squeeze(mu)
        mu[mu < 0] = 0
        return mu

    def muNAC(self):
        """
        :return:  mu-map image of mu == 0.01 sized according to self.muHardware()
        """
        muh = self.muHardware()
        return np.zeros(muh.shape, dtype=muh.dtype)

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



    _constants = {}
    _txLUT = {}
    _axLUT = {}
    _datain = {}
    _frame = 0
    _umapIdx = 0
    _t0 = 0
    _t1 = 0

    def _gatherOsemoneList(self, olist):
        """
        :param olist:  list
        :return:       numpy.array with times concatenated along axis=0 (c-style)
        """
        im = [olist[0].im]
        for i in range(1, len(olist)):
            im = np.append(im, [olist[i].im], axis=0)
        return np.float_(im)

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

