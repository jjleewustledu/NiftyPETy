import numpy as np
import sys, os
import respet.recon
#from niftypet import nimpa
#from pylab import *



class Reconstruction(object):
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2018"

    umapSynthFileprefix = ''
    span = 11
    bootstrap = 0
    datain = {}
    recmod = 3
    itr = 4
    fwhm = 4.3/2.08626 # number of voxels;  https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
    mMRparams = {}
    hmuSelection = [1,4,5] # selects from ~/.niftypet/resources.py:  hrdwr_mu
    outfolder = 'output'
    umap4dfp='umapSynth.4dfp'
    umapFolder = 'umap'
    use_stored_hdw_mumap = True
    use_stored_hist = False
    verbose = True

    @property
    def tracerRawdataLocation(self):
        if not self._ac:
            return self._tracerRawdataLocation+'-NAC'
        else:
            return self._tracerRawdataLocation+'-AC'

    def tracerRawdataLocationWith(self, ac=False):
        if not ac:
            return self._tracerRawdataLocation+'-NAC'
        else:
            return self._tracerRawdataLocation+'-AC'

    @property
    def outpath(self):
        return os.path.join(self.datain['corepath'], self.outfolder)

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

    @staticmethod
    def printd(d):
        for keys,values in d.items():
            print(keys)
            print(values)



    def createStaticNAC(self, fcomment='_createStaticNAC'):
        self.recmod = 0
        self.bootstrap = 0
        _hst = self.checkHistogramming(fcomment)
        return self.createStatic(self.muNAC(), hst=_hst, fcomment=fcomment)

    def createStaticUTE(self, fcomment='_createStaticUTE'):
        self.recmod = 3
        self.bootstrap = 0
        self.checkUmaps(self.muUTE(), fcomment)
        _hst = self.checkHistogramming(fcomment)
        return self.createStatic(self.muUTE(), hst=_hst, fcomment=fcomment)

    def createStaticCarney(self, fcomment='_createStaticCarney'):
        self.recmod = 3
        self.bootstrap = 0
        self.checkUmaps(self.muCarney(frames=[0]), fcomment)
        _hst = self.checkHistogramming(fcomment)
        return self.createStatic(self.muCarney(frames=[0]), hst=_hst, fcomment=fcomment)

    def createDynamicNAC(self, fcomment='_createDynamicNAC'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        print(times)
        self.recmod = 0
        self.bootstrap = 0
        self.checkUmaps(self.muHardware(), fcomment)
        self.checkHistogramming(fcomment)
        return self.createDynamic(times, self.muNAC(), fcomment)

    def createDynamicTiny(self, fcomment='_createDynamicTiny'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicTiny ##########")
        print(times)
        self.recmod = 1
        self.bootstrap = 0
        self.itr = 3
        self.checkUmaps(self.muTiny(), fcomment)
        self.checkHistogramming(fcomment)
        return self.createDynamic(times, self.muTiny(), fcomment)

    def createDynamicUTE(self, fcomment='_createDynamicUTE'):
        times = self.getTimes()
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        print(times)
        self.checkUmaps(self.muUTE(), fcomment)
        self.checkHistogramming(fcomment)
        return self.createDynamic(times, self.muUTE(), fcomment)

    def createDynamic2Carney(self, fcomment='_createDynamic2Carney'):
        times = self.getTimes()
        times2 = self.getTimes(self.getTaus2())
        print("########## respet.recon.reconstruction.Reconstruction.createDynamic2Carney ##########")
        print(times)
        print(times2)
        self.checkUmaps(self.muCarney(frames=[0]), fcomment)
        self.checkHistogramming(fcomment)
        return self.createDynamic2(times, times2, fcomment)

    def createStatic(self, muo, hst=None, fcomment='_createStatic'):
        """
        :param muo:       mu-map of imaged object
        :param fcomment;  string for naming subspace
        :return:          result from nipet.mmrchain
        :rtype:           dictionary
        """
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        hst = nipet.mmrhist(self.datain, self.mMRparams)
        sta = nipet.mmrchain(self.datain, self.mMRparams,
                             mu_h      = self.muHardware(),
                             mu_o      = muo,
                             itr       = self.itr,
                             fwhm      = self.fwhm,
                             recmod    = self.recmod,
                             outpath   = self.outpath,
                             store_img = False,
                             fcomment  = fcomment)
        self.saveStatic(sta, [muo, self.muHardware()], hst, fcomment)
        return sta

    def saveStatic(self, sta, mumaps, hst, fcomment=''):
        """
        :param sta:       dictionary from nipet.mmrchain
        :param mumaps:    dictionary of mu-maps from imaged object, hardware
        :param hst:       dictionary from nipet.mmrhist
        :param fcomment:  string to append to canonical filename
        """
        fout = self._createFilename(fcomment)
        im = sta['im']
        if self.mMRparams['Cnt']['VERBOSE']:
            print('i> saving 3D image to: ', fout)

        A = self.getAffine()
        muo,muh = mumaps  # object and hardware mu-maps
        if hst is None:
            hst = self.checkHistogramming()
        desc = self._createDescrip(hst, muh, muo)
        assert len(im.shape) == 3, "Reconstruction.saveStatic.im.shape == " + str(len(im.shape))
        self._array2nii(im[::-1,::-1,:], A, fout, descrip=desc)

    def createDynamic(self, times, muo, fcomment='_createDynamic'):
        """
        :param times:  np.int_; [0,0] produces a single time-frame
        :param muo:    3D or 4D mu-map of imaged object
        :return:       last result from nipet.mmrchain
        :rtype:        dictionary
        """
        global dynFrame
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        for it in np.arange(1, times.shape[0]):
            try:
                dynFrame = nipet.mmrchain(self.datain, self.mMRparams,
                                          frames    = ['fluid', [times[it-1], times[it]]],
                                          mu_h      = self.muHardware(),
                                          mu_o      = muo,
                                          itr       = self.itr,
                                          fwhm      = self.fwhm,
                                          recmod    = self.recmod,
                                          outpath   = self.outpath,
                                          store_img = True,
                                          fcomment  = fcomment + '_time' + str(it-1))
            except UnboundLocalError as e:
                print('==========================================================================')
                print('w> nipet.img.pipe will fail by attempting to use recimg before assignment;')
                print('w> skip reconstruction of time frame '+str(it - 1)+'.')
                print('==========================================================================')
                self.replaceFrameInSitu(times[it-1], times[it], fcomment, it-1)

        assert isinstance(dynFrame, dict)
        return dynFrame

    def createDynamic2(self, times, times2, fcomment='_createDynamic2'):
        """
        :param times:   np.int_ for mu-map frames; [0,0] produces a single time-frame
        :param times2:  np.int_ for emission frames
        :param muo:     3D or 4D mu-map of imaged object
        :return:        last result from nipet.mmrchain
        :rtype:         dictionary
        """
        global dynFrame
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        it = 1                                     # mu-map frame
        for it2 in np.arange(1, times2.shape[0]):  # hist frame
            if times2[it2-1] >= times[it]:
                it = it + 1
            try:
                dynFrame = nipet.mmrchain(self.datain, self.mMRparams,
                                          frames    = ['fluid', [times2[it2-1], times2[it2]]],
                                          mu_h      = self.muHardware(),
                                          mu_o      = self.muCarney(frames=(it-1)),
                                          itr       = self.itr,
                                          fwhm      = self.fwhm,
                                          recmod    = self.recmod,
                                          outpath   = self.outpath,
                                          store_img = True,
                                          fcomment  = fcomment + '_time' + str(it2 - 1))
            except UnboundLocalError as e:
                print('==========================================================================')
                print('w> nipet.img.pipe will fail by attempting to use recimg before assignment;')
                print('w> skip reconstruction of time frame '+str(it2 - 1)+'.')
                print('==========================================================================')
                self.replaceFrameInSitu(times2[it2-1], times2[it2], fcomment, it2-1)

        assert isinstance(dynFrame, dict)
        return dynFrame

    def replaceFrameInSitu(self, t0, t1, tag, fr):
        from shutil import copyfile
        copyfile(
            os.path.join(os.getenv('PPG_SUBJECTS_DIR'), 'zeros_frame.nii.gz'),
            self.nipetFrameFilename(t0, t1, tag, fr))

    def nipetFrameFilename(self, t0, t1, tag, fr):
        #       a_itr-4_t-10-20sec_createDynamic2Carney_time1.nii.gz
        return os.path.join(self.outpath, 'PET', 'single-frame', 'a_itr-'+str(self.itr)+'_t-'+str(t0)+'-'+str(t1)+'sec'+tag+'_time'+str(fr)+'.nii.gz')

    def createDynamicInMemory(self, times, muo, hst=None, fcomment='_createDynamicInMemory'):
        """
        within unittest environment, may use ~60 GB memory for 60 min FDG recon with MRAC
        :param times:  np.int_; [0,0] produces a single time-frame
        :param muo:    3D or 4D mu-map of imaged object
        :return:       result from nipet.mmrchain
        :rtype:        dictionary
        """
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        dyn = (times.shape[0]-1)*[None]
        for it in np.arange(1, times.shape[0]):
            dyn[it-1] = nipet.mmrchain(self.datain, self.mMRparams,
                                       frames    = ['fluid', [times[it-1], times[it]]],
                                       mu_h      = self.muHardware(),
                                       mu_o      = muo,
                                       itr       = self.itr,
                                       fwhm      = self.fwhm,
                                       recmod    = self.recmod,
                                       outpath   = self.outpath,
                                       store_img = False,
                                       fcomment  = fcomment + '_time' + str(it - 1))
        self.saveDynamicInMemory(dyn, [muo, self.muHardware()], hst, fcomment)
        return dyn

    def saveDynamicInMemory(self, dyn, mumaps, hst, fcomment=''):
        """
        :param dyn:       dictionary from nipet.mmrchain
        :param mumaps:    dictionary of mu-maps from imaged object, hardware
        :param hst:       dictionary from nipet.mmrhist
        :param fcomment:  string to append to canonical filename
        """
        fout = self._createFilename(fcomment)
        im = self._gatherOsemoneList(dyn)
        if self.mMRparams['Cnt']['VERBOSE']:
            print('i> saving '+str(len(im.shape))+'D image to: ', fout)

        A = self.getAffine()
        muo,muh = mumaps  # object and hardware mu-maps
        if hst is None:
            hst = self.checkHistogramming()
        desc = self._createDescrip(hst, muh, muo)
        if len(im.shape) == 3:
            self._array2nii(im[::-1,::-1,:],     A, fout, descrip=desc)
        elif len(im.shape) == 4:
            self._array4D2nii(im[:,::-1,::-1,:], A, fout, descrip=desc)

    def createUmapSynthFullBlurred(self):
        """
        :return:  os.path.join(tracerRawdataLocation, umapSynthFileprefix+'.nii.gz') with 4.3 mm fwhm blur
        """
        from subprocess import call
        pwd0 = os.getcwd()
        os.chdir(self.tracerRawdataLocationWith(ac=False))
        call('/data/nil-bluearc/raichle/lin64-tools/nifti_4dfp -n ' +
             os.path.join(self.tracerRawdataLocationWith(ac=False), self.umap4dfp) + '.ifh umap_.nii',
             shell=True, executable='/bin/bash')
        call('/bin/gzip umap_.nii', shell=True, executable='/bin/bash')
        call('/usr/local/fsl/bin/fslroi umap_ umap__ -86 344 -86 344 0 -1',
             shell=True, executable='/bin/bash')
        call('/usr/local/fsl/bin/fslmaths umap__ -s 1.826 ' + self.umapSynthFileprefix,
             shell=True, executable='/bin/bash')
        os.remove('umap_.nii.gz')
        os.remove('umap__.nii.gz')
        os.chdir(pwd0)

    def checkHistogramming(self, fcomment=''):
        from niftypet import nipet
        from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig

        hst = nipet.mmrhist(self.datain, self.mMRparams)
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)

        # sinogram index (<127 for direct sinograms, >=127 for oblique sinograms)
        si = 64

        # prompt sinogram
        figure()
        matshow(hst['psino'][si, :, :], cmap='inferno')
        colorbar()
        xlabel('bins')
        ylabel('angles')
        savefig(os.path.join(self.outpath, fcomment+'_promptsino.pdf'))

        # delayed sinogram
        figure()
        matshow(hst['dsino'][si, :, :], cmap='inferno')
        colorbar()
        xlabel('bins')
        ylabel('angles')
        savefig(os.path.join(self.outpath, fcomment+'_delayedsino.pdf'))

        # head curve for prompt and delayed events
        figure()
        plot(hst['phc'], label='prompt TAC')
        plot(hst['dhc'], label='delayed TAC')
        #show()
        legend()
        grid('on')
        xlabel('time/s')
        ylabel('specific activity / (Bq/mL)')
        savefig(os.path.join(self.outpath, fcomment+'_tacs.pdf'))

        # center of mass
        figure()
        plot(hst['cmass'])
        #show()
        grid('on')
        xlabel('time / s')
        ylabel('center of mas of radiodistribution')
        savefig(os.path.join(self.outpath, fcomment+'_cmass.pdf'))

        return hst

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

    def checkUmaps(self, muo, fcomment=''):
        muh = self.muHardware()
        iz  = 64
        ix  = 172

        # plot axial image with a colour bar
        matshow(muh['im'][iz, :, :] + muo['im'][iz, :, :], cmap='bone')
        colorbar()
        savefig(os.path.join(self.outpath, fcomment+'_tumaps.pdf'))

        # plot sagittal image with a colour bar
        matshow(muh['im'][:, :, ix] + muo['im'][:, :, ix], cmap='bone')
        colorbar()
        savefig(os.path.join(self.outpath, fcomment+'_sumaps.pdf'))

        # plot coronal image with a colour bar
        matshow(muh['im'][:, ix, :] + muo['im'][:, ix, :], cmap='bone')
        colorbar()
        savefig(os.path.join(self.outpath, fcomment+'_cumaps.pdf'))

    def getAffine(self):
        """
        :return:  affine transformations for NIfTI
        :rtype:   list 2D numeric
        """
        from niftypet import nipet
        cnt = self.mMRparams['Cnt']
        vbed, hbed = nipet.mmraux.vh_bedpos(self.datain, cnt)  # bed positions

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
        :return:  1x73 array of frame durations:  12 x 10-sec frames + 6 x 30-sec-frames + 55 x 60-sec-frames
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
        return np.int_(t) # TODO return np.trunc(t)

    def getTimeMax(self):
        """
        :return:  max time available from listmode data in sec.
        """
        from niftypet.nipet.lm import mmr_lmproc #CUDA
        nele, ttags, tpos = mmr_lmproc.lminfo(self.datain['lm_bf'])
        return (ttags[1]-ttags[0]+999)/1000 # sec

    def migrateCndaDownloads(self, cndaDownload):
        return None

    def muHardware(self):
        """
        :return:  dictionary for hardware mu-map image provided by nipet.hdw_mumap.  Keys:  'im', ...
        See also self.hmuSelection.
        """
        from niftypet import nipet
        if self.use_stored_hdw_mumap:
            self.datain['hmumap'] = os.path.join(
                os.getenv('HOME'), 'Local', 'JSRecon12e11p', 'hardwareumaps', 'hmumap.npy')
        return nipet.hdw_mumap(
            self.datain, self.hmuSelection, self.mMRparams, outpath=self.outpath, use_stored=self.use_stored_hdw_mumap)

    def muCarney(self, fileprefix=None, imtype='object mu-map', fcomment='', frames=None):
        """
        get NIfTI of the custom umap; see also nipet.mmrimg.obtain_image
        :param fileprefix:  string for fileprefix of 4D image-object; default := self.umapSynthFileprefix
        :param imgtype:  string; cf. obtain_image
        :param fcomment:  string to append to fileprefix
        :param frames:  frame indices to select from _im;  default selects all frames
        :return:  np.float32
        """
        from niftypet import nimpa

        if fileprefix is None:
            fileprefix = self.umapSynthFileprefix
        fqfn = os.path.join(self.tracerRawdataLocationWith(ac=True), fileprefix + fcomment + '.nii.gz')
        nimpa_dct = nimpa.getnii(fqfn, output='all')
        _im = nimpa_dct['im']
        if not frames is None:
            _im = _im[frames,:,:,:]
        _im = np.squeeze(_im)
        _im[_im < 0] = 0

        output = {}
        output['im'] = _im
        output['affine'] = nimpa_dct['affine']
        output['exists'] = True
        output['fim'] = fqfn
        Cnt = self.mMRparams['Cnt']
        if Cnt['VERBOSE']:
            print 'i> using ' + imtype + ' from NIfTI file.'
        if Cnt and output['im'].shape != (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            print 'e> provided ' + imtype + ' via file has inconsistent dimensions compared to Cnt.'
            raise ValueError('Wrong dimensions of the mu-map')
        return output

        # PREVIOUSLY:
        # import nibabel
        # if fileprefix is None:
        #     fileprefix = self.umapSynthFileprefix
        # fqfn = os.path.join(self.tracerRawdataLocation, fileprefix + fcomment + '.nii.gz')
        # nim  = nibabel.load(fqfn)
        # _im  = np.float32(nim.get_data())
        # if frames is None:
        #     if np.ndim(_im) == 3:
        #         _im = np.transpose(_im[:,::-1,::-1], (2, 1, 0))
        #     elif np.ndim(_im) == 4:
        #         _im = np.transpose(_im[:,::-1,::-1,:], (3, 2, 1, 0))
        #     else:
        #         raise ValueError('unsupported np.ndim(Reconstruction.muCarney._im)->' + str(np.ndim(_im)))
        # else:
        #     _im = np.transpose(_im[:,::-1,::-1,frames], (3, 2, 1, 0))
        # _im = np.squeeze(_im)
        # _im[_im < 0] = 0

    def muNAC(self):
        """
        :return:  [] which NIPET 1.1 uses for no attenuation correction
        """
        return []

    def muTiny(self):
        """
        :return:  mu-map image of mu == 0.01 sized according to self.mMRparams['Cnt']['SO_IM*']
        """
        _cnt = self.mMRparams['Cnt']
        _im  = 0.01*np.ones((_cnt['SO_IMZ'], _cnt['SO_IMY'], _cnt['SO_IMX']), dtype=np.float32)
        return dict(im=_im)

    def muUTE(self):
        """
        :return:  mu-map image from Siemens UTE
        :rtype:  numpy.array
        """
        from niftypet import nipet
        return nipet.obj_mumap(self.datain, self.mMRparams, outpath=self.outpath, store=True)

    def organizeRawdataLocation(self, cndaDownload=None):
        import glob
        import pydicom
        import shutil
        import errno

        if not self._ac:
            if isinstance(cndaDownload, str):
                self.migrateCndaDownloads(cndaDownload)
            try:
                # NAC:  check umap; move norm and listmode to folders
                assert(os.path.isdir(os.path.join(self.tracerRawdataLocation, 'umap', '')))
                fns = glob.glob(os.path.join(self.tracerRawdataLocation, '*.dcm'))
                for fn in fns:
                    ds = pydicom.read_file(fn)
                    if ds.ImageType[2] == 'PET_NORM':
                        self._moveToNamedLocation(fn, 'norm')
                    if ds.ImageType[2] == 'PET_LISTMODE':
                        self._moveToNamedLocation(fn, 'LM')
            except OSError:
                os.listdir(self.tracerRawdataLocation)
                raise
        else:
            # AC:  move .bf, .dcm and umap to tracerRawdataLocation
            nac_norm = os.path.join(self.tracerRawdataLocationWith(ac=False), 'norm', '')
            ac_norm  = os.path.join(self.tracerRawdataLocationWith(ac=True),  'norm', '')
            nac_lm   = os.path.join(self.tracerRawdataLocationWith(ac=False), 'LM', '')
            ac_lm    = os.path.join(self.tracerRawdataLocationWith(ac=True),  'LM', '')
            nac_umap = os.path.join(self.tracerRawdataLocationWith(ac=False), 'umap', '')
            ac_umap  = os.path.join(self.tracerRawdataLocationWith(ac=True),  'umap', '')
            if not os.path.isdir(ac_norm):
                shutil.move(nac_norm, self.tracerRawdataLocation)
            if not os.path.isdir(ac_lm):
                shutil.move(nac_lm,   self.tracerRawdataLocation)
            if not os.path.isdir(ac_umap):
                shutil.move(nac_umap, self.tracerRawdataLocation)

    def __init__(self, loc=None, ac=False, umapSF='umapSynth', v=False, cndaDownload=None):
        """
        :param:  loc specifies the location of tracer rawdata.
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
                          -rwxr-xr-x+  1 jjlee wheel     145290 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.dcm:param:  cndaDownload is a path
        :param:  ac, attenuation correction, is bool
        :param:  umapSF is a fileprefix
        :param:  v, verbosity, is bool
        :param:  cndaDownload is a path
        """
        if loc is None:
            return
        self._tracerRawdataLocation = loc
        self._ac = ac
        if not os.path.exists(self.tracerRawdataLocation):
            os.makedirs(self.tracerRawdataLocation)
        os.chdir(self.tracerRawdataLocation)
        self.umapSynthFileprefix = umapSF
        self.verbose = v
        self.organizeRawdataLocation(cndaDownload)
        self._initializeNiftypet()

    # listing of instance variables:
    # _frame = 0
    # _ac = False
    # _umapIdx = 0
    # _t0 = 0
    # _t1 = 0
    # _tracerRawdataLocation = None



    def _array2nii(self, im, A, fnii, descrip=''):
        """
        Store the numpy array to a NIfTI file <fnii>
        """
        if im.ndim == 3:
            im = np.transpose(im, (2, 1, 0))
        elif im.ndim == 4:
            im = np.transpose(im, (3, 2, 1, 0))
        else:
            raise StandardError('unrecognised image dimensions')

        import nibabel as nib
        nii = nib.Nifti1Image(im, A)
        hdr = nii.header
        hdr.set_sform(None, code='scanner')
        hdr['cal_max'] = np.max(im)
        hdr['cal_min'] = np.min(im)
        hdr['descrip'] = descrip
        nib.save(nii, fnii)

    def _array4D2nii(self, im, A, fnii, descrip=''):
        # print 'A = ', A

        import nibabel as nib
        im = np.transpose(im, (3, 2, 1, 0))
        nii = nib.Nifti1Image(im, A)
        hdr = nii.header
        hdr.set_sform(None, code='scanner')
        hdr['cal_max'] = np.max(im)
        hdr['cal_min'] = np.min(im)
        hdr['descrip'] = descrip
        nib.save(nii, fnii)

    def _gatherOsemoneList(self, olist):
        """
        :param olist:  list of dictionaries
        :return:       numpy.array with times concatenated along axis=0 (c-style)
        """
        im = [olist[0]['im']]
        for i in range(1, len(olist)):
            im = np.append(im, [olist[i].im], axis=0)
        return np.float_(im)

    def _initializeNiftypet(self):
        from niftypet import nipet
        self.mMRparams = nipet.get_mmrparams()
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['SPN']     = self.span
        self.mMRparams['Cnt']['BTP']     = self.bootstrap
        self.datain = nipet.classify_input(self.tracerRawdataLocation, self.mMRparams)
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        if self.verbose:
            print("########## respet.recon.reconstruction.Reconstruction._initializeNiftypet ##########")
            print(self.datain)
            
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

    def _tracer(self):
        import re
        return re.split('_', os.path.basename(self.tracerRawdataLocation))[0]

    def _createDescrip(self, hst, muh, muo):
        """
        :param hst:  from nipet.mmrhist
        :param muh:  is mumaps dictionary
        :param muo:  is mumaps dictionary
        :return:     description text for NIfTI
        if only bed present, attnum := 0.5
        """
        from niftypet import nipet
        cnt    = self.mMRparams['Cnt']
        attnum = (1 * (np.sum(muh['im']) > 0.5) + 1 * (np.sum(muo['im']) > 0.5)) / 2.
        ncmp,_ = nipet.mmrnorm.get_components(self.datain, cnt)
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
        from niftypet import nipet
        nipet.mmraux.create_dir(self.outpath)
        pth = os.path.join(self.outpath, os.path.basename(self.datain['lm_dcm'])[:8] + fcomment + '.nii.gz')
        return pth

    def _riLUT(self):
        """
        :return:  radioisotope look-up table
        """
        return {'Ge68':{'BF':0.891, 'thalf':270.9516*24*60*60},
                'Ga68':{'BF':0.891, 'thalf':67.71*60},
                 'F18':{'BF':0.967, 'thalf':109.77120*60},
                 'C11':{'BF':0.998, 'thalf':20.38*60}}

