import numpy as np
import os

class Reconstruction(object):
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2018"

    DCYCRR = True
    DEVID = 0
    bootstrap = 0
    datain = {}
    fwhm = 4.3/2.08626 # number of voxels;  https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
    hmuSelection = [1,4,5] # selects from ~/.niftypet/resources.py:  hrdwr_mu
    itr = 4
    mMRparams = {}
    outfolder = 'output'
    recmod = 3
    span = 11
    tracerMemory = None
    umap4dfp='umapSynth.4dfp'
    umapFolder = 'umap'
    umapSynthFileprefix = ''
    use_stored_hdw_mumap = True
    use_stored_hist = False
    verbose = True



    @property
    def outpath(self):
        """
        :return e.g., '/work/HYGLY48/V1/OO1_V1-Converted-NAC/output':
        """
        return os.path.join(self.datain['corepath'], self.outfolder)

    @property
    def PETpath(self):
        """
        :return e.g., '/work/HYGLY48/V1/OO1_V1-Converted-NAC/output/PET':
        """
        return os.path.join(self.outpath, 'PET')

    @property
    def reconstruction_finished(self):
        return os.path.exists(self._filename_to_finish('_finished'))

    @property
    def reconstruction_started(self):
        return os.path.exists(self._filename_to_touch('_started'))

    @property
    def tracer(self):
        """
        :return e.g., 'OO1':
        """
        import re
        return re.split('_', os.path.basename(self.tracerRawdataLocation))[0]

    @property
    def tracerRawdataLocation(self):
        """
        :return e.g., '/work/HYGLY48/V1/OO1_V1-Converted-NAC':
        """
        return self._tracerRawdataLocation

    def tracerRawdataLocation_with(self, ac=False):
        from re import compile
        conv = compile('-Converted-')
        s = conv.search(self._tracerRawdataLocation)
        baseloc = self._tracerRawdataLocation[:s.end()-1]
        if not ac:
            return baseloc+'-NAC'
        else:
            return baseloc+'-AC'

    @property
    def visitStr(self):
        """
        e.g., for 'FDG_DT1234567789.000000-Converted-NAC' and 'FDG_V1-Converted-NAC'
        :return 'dt123456789' and 'v1':
        """
        import re
        v = re.split('_', os.path.basename(self.tracerRawdataLocation))[1]
        v = re.split('-Converted', v)[0]
        w = re.split('\.', v)[0]
        if not w:
            return v.lower()
        else:
            return w.upper()



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
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        self.recmod = 0
        self.bootstrap = 0
        self.checkUmaps(self.muHardware(), fcomment)
        self.checkHistogramming(fcomment)
        return self.createDynamic(self.getTaus(), self.muNAC(), fcomment)

    def createDynamicUTE(self, fcomment='_createDynamicUTE'):
        print("########## respet.recon.reconstruction.Reconstruction.createDynamicNAC ##########")
        self.checkUmaps(self.muUTE(), fcomment)
        self.checkHistogramming(fcomment)
        return self.createDynamic(self.getTaus(), self.muUTE(), fcomment)

    def createDynamic2Carney(self, fcomment='_createDynamic2Carney'):
        print("########## respet.recon.reconstruction.Reconstruction.createDynamic2Carney ##########")
        self.checkUmaps(self.muCarney(frames=[0]), fcomment)
        self.checkHistogramming(fcomment)
        taus = self.getTaus(self.json_filename_with(ac=False))
        wtime = self.getWTime(self.json_filename_with(ac=False))
        return self.createDynamic2(wtime, taus, self.getTaus2(), fcomment)

    def createStatic(self, muo, hst=None, fcomment='_createStatic'):
        """
        :param muo:       mu-map of imaged object
        :param fcomment;  string for naming subspace
        :return:          result from nipet.mmrchain
        :rtype:           dictionary
        """
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['DCYCRR'] = self.DCYCRR
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

    def createDynamic(self, taus, muo, fcomment='_createDynamic'):
        """
        :param taus: np.int_
        :param muo:  3D or 4D mu-map of imaged object
        :return:     last result from nipet.mmrchain
        :rtype:      dictionary
        """
        global dynFrame
        from numpy import isnan
        from niftypet import nipet
        from warnings import warn
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['DCYCRR'] = self.DCYCRR
        if self.reconstruction_started:
            return None
        self._do_touch_file('_started')
        dynFrame = None
        times = self.getTimes(taus)
        wtime = times[0]
        fit = None
        for it in np.arange(1, times.shape[0]):
            fit = it
            try:
                if self.frame_exists(times[it-1], times[it], fcomment, it):
                    continue
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
                if isnan(dynFrame['im']).any():
                    wtime = times[it]
            except (UnboundLocalError, IndexError) as e:
                warn(e.message)
                warn('Reconstruction.createDynamic:  nipet.img.pipe will fail by attempting to use recimg before assignment')
                if times[it] < times[-1]/2:
                    warn('Reconstruction.createDynamic:  calling requestFrameInSitu')
                    self.replaceFrameInSitu(times[it-1], times[it], fcomment, it-1)
                    wtime = times[it]
                else:
                    warn('Reconstruction.createDynamic:  break for it->' + str(it))
                    break

        self.save_json(taus[:fit], waittime=wtime)
        self._do_touch_file('_finished')
        return dynFrame

    def createDynamic2(self, wtime, taus, taus2, fcomment='_createDynamic2'):
        """
        :param wtime is determined by createDynamic:
        :param taus     np.int_ for mu-map frames:
        :param taus2    np.int_ for emission frames:
        :param muo      3D or 4D mu-map of imaged object:
        :return         last result from nipet.mmrchain:
        :rtype          dictionary:
        """
        global dynFrame
        from numpy import isnan
        from niftypet import nipet
        from warnings import warn
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['DCYCRR'] = self.DCYCRR
        if self.reconstruction_started:
            return None
        self._do_touch_file('_started')
        dynFrame = None
        times = self.getTimes(taus)
        times2 = self.getTimes(taus2) + wtime
        wtime2 = times2[0] - wtime
        fit2 = None
        it = 1                                     # mu-map frame
        for it2 in np.arange(1, times2.shape[0]):  # hist frame
            fit2 = it2
            try:
                while times[it] < times2[it2-1] and times[it] < times[-1]:
                    it += 1 # find the best mu-map
                if self.frame_exists(times2[it2-1], times2[it2], fcomment, it2):
                    continue
                if times2[it2-1] < min(times2[it2], times[-1]):
                    print('createDynamic2:  AC frame samples {}-{} s; NAC frame samples {}-{} s'.format(times2[it2-1], times2[it2], times[it-1], times[it]))
                    dynFrame = nipet.mmrchain(self.datain, self.mMRparams,
                                              frames    = ['fluid', [times2[it2-1], min(times2[it2], times[-1])]],
                                              mu_h      = self.muHardware(),
                                              mu_o      = self.muCarney(frames=(it-1)),
                                              itr       = self.itr,
                                              fwhm      = self.fwhm,
                                              recmod    = self.recmod,
                                              outpath   = self.outpath,
                                              store_img = True,
                                              fcomment  = fcomment + '_time' + str(it2-1))
                    if times2[it2] == times[-1]:
                        fit2 = it2
                        print('createDynamic2.fit2->' + str(fit2))
                    if isnan(dynFrame['im']).any():
                        wtime2 = times2[it2] - wtime
                        print('createDynamic2.wtime2->' + str(wtime2))
            except (UnboundLocalError, IndexError) as e:
                warn(e.message)
                warn('Reconstruction.createDynamic2:  nipet.img.pipe will fail by attempting to use recimg before assignment')
                if times2[it2] < times2[-1]/2:
                    warn('Reconstruction.createDynamic2:  calling requestFrameInSitu')
                    self.replaceFrameInSitu(times2[it2-1], times2[it2], fcomment, it2-1)
                    wtime2 = times2[it2] - wtime
                else:
                    warn('Reconstruction.createDynamic2:  break for it2->' + str(it2))
                    break

        self.save_json(taus2[:fit2], offsettime=wtime, waittime=wtime2)
        self._do_touch_file('_finished')
        return dynFrame

    def createDynamicInMemory(self, taus, muo, hst=None, fcomment='_createDynamicInMemory'):
        """
        within unittest environment, may use ~60 GB memory for 60 min FDG recon with MRAC
        :param taus:  np.int_
        :param muo:   3D or 4D mu-map of imaged object
        :return:      result from nipet.mmrchain
        :rtype:       dictionary
        """
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['DCYCRR'] = self.DCYCRR
        times = self.getTimes(taus)[0:3] # abbreviated for testing
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

    def createUmapSynthFullBlurred(self):
        """
        :return:  os.path.join(tracerRawdataLocation, umapSynthFileprefix+'.nii.gz') with 4.3 mm fwhm blur
        """
        from subprocess import call
        pwd0 = os.getcwd()
        os.chdir(self.tracerRawdataLocation_with(ac=False))
        call('/data/nil-bluearc/raichle/lin64-tools/nifti_4dfp -n ' +
             os.path.join(self.tracerRawdataLocation_with(ac=False), self.umap4dfp) + '.ifh umap_.nii',
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
        from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig, matshow, colorbar, legend, grid

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
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeAliasingUTE ##########")
        return self.createDynamicInMemory(self.getTaus(), self.muUTE(), fcomment)

    def checkTimeAliasingCarney(self, fcomment='_checkTimeAliasingCarney'):
        times = self.getTimes(self.getTaus())
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeAliasingCarney ##########")
        print(times[0:2])
        return self.createDynamic(self.getTaus()[0:2], self.muCarney(frames=[0,1]), fcomment)

    def checkTimeHierarchiesCarney(self, fcomment='_checkTimeHierarchiesCarney'):
        times = self.getTimes(self.getTaus())
        times2 = self.getTimes(self.getTaus2())
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeHierarchiesCarney ##########")
        print(times)
        return self.createDynamic2(self.getTaus()[0:2], self.getTaus2()[0:6], self.muCarney(frames=[0,1]), fcomment)

    def checkUmaps(self, muo, fcomment=''):
        from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig, matshow, colorbar, legend, grid

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

    def frame_exists(self, t0, tf, fcomment, it2):
        """
        e.g., a_itr-4_t-577-601sec_createDynamic2Carney_time57.nii.gz
        :param t0:
        :param tf:
        :param fcomment:
        :param it2:
        :return bool:
        """
        fn = "a_itr-" + str(self.itr) + "_t-" + str(t0) + "-" + str(tf) + "sec" + fcomment + "_time" + str(it2-1) + ".nii.gz"
        return os.path.exists(os.path.join(self.PETpath, 'single-frame', fn))

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

    def getInterfile(self, dcm):
        """
        :param dcm:
        :return lm_dict, a dictionary of interfile fields:
        """
        from interfile import Interfile
        try:
            lm_dict = Interfile.load(dcm)
        except (AttributeError, TypeError):
            raise AssertionError('dcm must be a filename')
        return lm_dict

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

    def getTaus(self, json_file=None):
        """
        :param:  json_file containing taus
        :return:  np array of frame durations
        :rtype:  numpy.int_
        """
        if json_file:
            taus,wtime = self.open_json(json_file)
            return taus
        if not self.tracerMemory:
            raise AssertionError('Reconstruction.getTaus:  no tracerMemory')
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            taus = np.int_([30,32,33,35,37,40,43,46,49,54,59,65,72,82,94,110,132,165,218,315,535,1354])
            # len == 22, dur == 3600
        elif self.tracerMemory.lower() == 'oxygen-water' or self.tracerMemory.lower() == 'carbon' or self.tracerMemory.lower() == 'oxygen':
            taus = np.int_([10,11,11,12,13,14,15,16,18,20,22,25,29,34,41,52,70,187])
            # len == 18, dur == 600
        else:
            raise AssertionError('Reconstruction.getTaus does not support tracerMemory->' + self.tracerMemory)
        return taus

    def getTaus2(self, json_file=None):
        """
        :param:  json_file containing taus
        :return:  np array of frame durations, waiting time
        :rtype:  numpy.int_
        """
        if json_file:
            taus,wtime = self.open_json(json_file)
            return taus
        if not self.tracerMemory:
            raise AssertionError('Reconstruction.getTaus2:  no tracerMemory')
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            taus = np.int_([10,10,10,11,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,15,15,15,16,16,17,17,18,18,19,19,20,21,21,22,23,24,25,26,27,28,30,31,33,35,37,39,42,45,49,53,58,64,71,80,92,107,128,159,208,295,485,1097])
            # len == 62, dur == 3887
        elif self.tracerMemory.lower() == 'oxygen-water' or self.tracerMemory.lower() == 'carbon' or self.tracerMemory.lower() == 'oxygen':
            taus = np.int_([5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,9,9,9,10,11,11,12,13,14,15,16,18,20,22,25,29,34,41,52,69,103])
            # len == 36, dur == 636
        else:
            raise AssertionError('Reconstruction.getTaus2 does not support tracerMemory->' + self.tracerMemory)
        return taus

    def getTimes(self, taus=None):
        """
        :return:  up to 1x66 array of times including 0 and np.cumsum(taus); max(times) <= self.getTimeMax
        :rtype:  numpy.int_
        """
        if not isinstance(taus, np.ndarray):
            raise AssertionError('Reconstruction.getTimes.taus is missing')
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

    def getWTime(self, json_file=None):
        """
        :param:  json_file containing taus
        :return:  waiting time
        :rtype:  numpy.int_
        """
        if json_file:
            taus,wtime = self.open_json(json_file)
            return wtime
        return 0

    def json_filename(self):
        return os.path.join(self.PETpath,
                            self.tracer + '_' + self.visitStr + '.json')

    def json_filename_with(self, ac=False):
        return os.path.join(self.tracerRawdataLocation_with(ac), 'output', 'PET',
                            self.tracer + '_' + self.visitStr + '.json')

    def open_json(self, json_file=None):
        """
        https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        :param json_file, a str:
        :return taus, a np array of frame durations, including waiting frames but only frames in the listmode archive:
        :return wtime, the waiting time in the early scan in sec:
        """
        import codecs, json
        if not json_file:
            raise AssertionError('Reconstruction.open_json.json_file is missing')
        t = codecs.open(json_file, 'r', encoding='utf-8').read()
        jt = json.loads(t)
        taus = np.array(jt['taus'])
        wtime = int(float(jt['waiting time']))
        return taus, wtime

    def save_json(self, taus=None, offsettime=0, waittime=0):
        """
        https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        :param taus, an np array of frame durations, including waiting frames but only frames in the listmode archive:
        :param offsettime, the duration between study time and the first saved frame:
        :param waittime, the witing time in the early scan; typically nan <- 0:
        :return json_file, a canonical json filename for ancillary data including timings:
        """
        import codecs, json
        if not isinstance(taus, np.ndarray):
            raise AssertionError('Reconstruction.save_json.taus is missing')
        jdict = {
            "study date": self.lm_studydate(),
            "acquisition time": self.lm_acquisitiontime(),
            "offset time": offsettime,
            "waiting time": waittime,
            "taus": taus.tolist(),
            "image duration": self.lm_imageduration()
        }
        json_file = self.json_filename()
        j = codecs.open(json_file, 'w', encoding='utf-8')  # overwrites existing
        json.dump(jdict, j)
        return json_file

    def lm_dcm(self):
        from glob import glob
        dcms = glob(os.path.join(self.tracerRawdataLocation, 'LM', '*.dcm'))
        for d in dcms:
            if os.path.exists(d):
                return d
        raise AssertionError("Reconstruction.lm_dcm could not open LM *.dcm")

    def lm_dcmread(self):
        """
        :return dcm_datset is a pydicom.dataset.FileDataset containing properties for DICOM fields:
        """
        from pydicom import dcmread
        try:
            dcm_datset = dcmread(self.lm_dcm())
        except (AttributeError, TypeError):
            raise AssertionError('dcm must be a filename')
        return dcm_datset

    def lm_imageduration(self):
        lm_dict = self.getInterfile(self.lm_dcm())
        return lm_dict['image duration']['value'] # sec

    def lm_studydate(self):
        """
        provides best estimate of date of listmode collection
        :param dcm filename:
        :return:
        """
        d = self.lm_dcmread()
        return d.StudyDate # YYYYMMDD after http://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

    def lm_acquisitiontime(self):
        """
        provides best estimate of start time (GMT) of listmode collection
        :param dcm filename:
        :return:
        """
        d = self.lm_dcmread()
        return d.AcquisitionTime # hhmmss.ffffff after http://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

    def lm_tracer(self):
        import re
        from warnings import warn
        if self.tracerMemory:
            return self.tracerMemory
        try:
            with open(self.lm_dcm(), 'r') as fid:
                fcontent = fid.read()
                p = re.compile('(?<=Radiopharmaceutical:)[A-Za-z\-]+')
                m = re.search(p, fcontent)
                self.tracerMemory = m.group(0)
                return self.tracerMemory
        except IOError as e:
            warn(e.message)
        raise AssertionError("Reconstruction.lm_tracer could not open LM *.dcm")

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
                os.getenv('HARDWAREUMAPS'), 'hmumap.npy')
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
        fqfn = os.path.join(self.tracerRawdataLocation_with(ac=True), fileprefix + fcomment + '.nii.gz')
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

    def muUTE(self):
        """
        :return:  mu-map image from Siemens UTE
        :rtype:  numpy.array
        """
        from niftypet import nipet
        return nipet.obj_mumap(self.datain, self.mMRparams, outpath=self.outpath, store=True)

    def nipetFrameFilename(self, t0, t1, tag, fr):
        #       a_itr-4_t-10-20sec_createDynamic2Carney_time1.nii.gz
        return os.path.join(self.outpath, 'PET', 'single-frame',
                            'a_itr-'+str(self.itr)+'_t-'+str(t0)+'-'+str(t1)+'sec'+tag+'_time'+str(fr)+'.nii.gz')

    def organizeNormAndListmode(self):
        import glob
        import pydicom

        try:
            # check umap; move norm and listmode to folders
            u = os.path.join(self.tracerRawdataLocation, 'umap', '')
            if not os.path.isdir(u):
                os.makedirs(u)
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

    def organizeRawdataLocation(self, cndaDownload=None):
        import shutil

        if self.tracerRawdataLocation.find('Twilite') > 0:
            self.organizeNormAndListmode()
            return
        if not self.ac:
            if cndaDownload:
                self.migrateCndaDownloads(cndaDownload)
            self.organizeNormAndListmode()
            return

        # AC:  move .bf, .dcm and umap to tracerRawdataLocation
        nac_norm = os.path.join(self.tracerRawdataLocation_with(ac=False), 'norm', '')
        ac_norm  = os.path.join(self.tracerRawdataLocation_with(ac=True), 'norm', '')
        nac_lm   = os.path.join(self.tracerRawdataLocation_with(ac=False), 'LM', '')
        ac_lm    = os.path.join(self.tracerRawdataLocation_with(ac=True), 'LM', '')
        nac_umap = os.path.join(self.tracerRawdataLocation_with(ac=False), 'umap', '')
        ac_umap  = os.path.join(self.tracerRawdataLocation_with(ac=True), 'umap', '')
        if not os.path.isdir(ac_norm):
            shutil.move(nac_norm, self.tracerRawdataLocation)
        if not os.path.isdir(ac_lm):
            shutil.move(nac_lm,   self.tracerRawdataLocation)
        if not os.path.isdir(ac_umap):
            shutil.move(nac_umap, self.tracerRawdataLocation)
        return

    @staticmethod
    def printd(d):
        for keys, values in d.items():
            print(keys)
            print(values)

    def replaceFrameInSitu(self, t0, t1, tag, fr):
        from shutil import copyfile
        copyfile(
            os.path.join(os.getenv('SUBJECTS_DIR'), 'zeros_frame.nii.gz'),
            self.nipetFrameFilename(t0, t1, tag, fr))

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

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



    # CLASS PRIVATE PROPERTIES & METHODS

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

    def _createDescrip(self, hst, muh, muo):
        """
        :param hst:  from nipet.mmrhist
        :param muh:  is mumaps list array
        :param muo:  is mumaps list array
        :return:     description text for NIfTI
        if only bed present, attnum := 0.5
        """
        from niftypet import nipet
        cnt    = self.mMRparams['Cnt']
        attnum = (1 * (np.sum(muh) > 0.5) + 1 * (np.sum(muo) > 0.5)) / 2.
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

    def _do_touch_file(self, tags=None):
        from pathlib2 import Path
        if not tags:
            return None
        f = self._filename_to_touch(tags)
        hd, tl = os.path.split(f)
        if not os.path.exists(hd):
            os.makedirs(hd)
        Path(f).touch()
        return f

    def _filename_to_touch(self, tags=None):
        if not tags:
            return None
        return os.path.join(self.PETpath, 'reconstruction_Reconstruction%s.touch' % tags)

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
        self.mMRparams['Cnt']['DCYCRR']  = self.DCYCRR
        self.mMRparams['Cnt']['DEVID']   = self.DEVID
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

    def _parse_prefix(self, prefix):
        """
        checks that prefix is well-formed and set class properties accordingly
        :param prefix is the location of tracer rawdata, e.g., FDG_DT123456789.000000-Converted-NAC:
        :return class properties ac & tracerRawdataLocation are valid:
        """
        from re import compile
        if not prefix:
            raise AssertionError(
                'reconstruction.Reconstruction requires a prefix parameter, the location of tracer data')
        nac = compile('-Converted-NAC')
        ac = compile('-Converted-AC')
        if nac.search(prefix):
            self.ac = False
        elif ac.search(prefix):
            self.ac = True
        else:
            raise AssertionError(
                'reconstruction.Reconstruction expected prefix parameter to end in -AC or -NAC')
        self._tracerRawdataLocation = prefix
        if not os.path.exists(self.tracerRawdataLocation):
            print(os.listdir('/'))
            print(os.listdir('/SubjectsDir'))
            raise AssertionError(
                'reconstruction.Reconstruction could not find prefix->' + self.tracerRawdataLocation)
            #os.makedirs(self.tracerRawdataLocation)

    def _riLUT(self):
        """
        :return:  radioisotope look-up table
        """
        return {'Ge68':{'BF':0.891, 'thalf':270.9516*24*60*60},
                'Ga68':{'BF':0.891, 'thalf':67.71*60},
                 'F18':{'BF':0.967, 'thalf':109.77120*60},
                 'C11':{'BF':0.998, 'thalf':20.38*60}}

    def __init__(self, prefix=None, umapSF='umapSynth', v=True, cndaDownload=None, devid=0):
        """
        :param:  prefix specifies the location of tracer rawdata.
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
                          -rwxr-xr-x+  1 jjlee wheel     145290 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.dcm
        :param:  cndaDownload is a path
        :param:  ac, attenuation correction, is bool
        :param:  umapSF is a fileprefix
        :param:  v, verbosity, is bool
        :param:  cndaDownload is a path
        """
        from string import split
        from niftypet import nipet
        self._parse_prefix(prefix)
        os.chdir(self.tracerRawdataLocation)
        self.umapSynthFileprefix = umapSF
        self.verbose = v
        self.organizeRawdataLocation(cndaDownload)
        self.tracerMemory = self.lm_tracer()
        print('reconstruction.__init__:\n')
        nipet.gpuinfo(extended=True)
        print('\n')
        print('self.tracerRawdataLocation->' + self.tracerRawdataLocation)
        print('\n')
        print('CUDA_VISIBLE_DEVICES:  ' + str(os.getenv('CUDA_VISIBLE_DEVICES')))
        print('\n')
        ids = os.getenv('CUDA_VISIBLE_DEVICES')
        if not ids:
            self.DEVID = devid
        else:
            self.DEVID = int(split(ids, ',')[0])
        assert 0 == self.DEVID, 'reconstruction.__init__.DEVID->' + str(self.DEVID)
        self._initializeNiftypet()



if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='usage:  nvidia-docker run -it niftypetr-image:reconstruction_cuda10 [-h] -p /SubjectsDir/CCIR_00123/ses-456789/TRACER_DT12345678000000.000000-Converted-NAC')
    p.add_argument('-p', '--prefix',
                   metavar='/path/to/TRACER_DT12345678000000.000000-Converted-NAC',
                   required=True,
                   help='location containing tracer raw data')
    args = p.parse_args()
    print(os.listdir('/'))
    print(os.listdir('/SubjectsDir'))
    print(os.listdir(args.prefix))
    r = Reconstruction(prefix=args.prefix)
    if not r.ac:
        r.createDynamicNAC(fcomment='_createDynamicNAC')
    else:
        r.createDynamic2Carney(fcomment='_createDynamic2Carney')
