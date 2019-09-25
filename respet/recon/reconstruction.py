import numpy as np
import os
import logging, sys

# create and configure main logger;
# see also https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules/50715155#50715155

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

class Reconstruction(object):
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2018"

    # see also:   mmrrec.py osemone param mask_radious
    bootstrap = 0
    cached_hdw_mumap = None
    datain = {}
    DCYCRR = True
    DEVID = 0

    histogram_si = 63 # sinogram index (<127 for direct sinograms)
    hmuSelection = [1,4,5] # selects from ~/.niftypet/resources.py:  hrdwr_mu
    minTime = 0
    mMRparams = {}
    outfolder = 'output'
    phantom = False
    recmod = 3
    tracerMemory = None
    umap4dfp='umapSynth.4dfp'
    umapFolder = 'umap'
    umapSynthFileprefix = ''
    use_mirror_hdw_mumap = False
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



    def createStaticNAC(self, time0=None, timeF=None, fcomment='_createStaticNAC'):
        self.recmod = 0
        self.bootstrap = 0
        return self.createStatic(self.muNAC(), 0, time0, timeF, fcomment=fcomment,)

    def createStaticUTE(self, time0=None, timeF=None, fcomment='_createStaticUTE'):
        self.recmod = 3
        self.bootstrap = 0
        self.checkUmaps(self.muUTE(), fcomment)
        #wtime = self.getWTime(self.json_filename_with(ac=False))
        return self.createStatic(self.muUTE(), 0, time0, timeF, fcomment=fcomment)

    def createStaticCarney(self, time0=None, timeF=None, fcomment='_createStaticCarney'):
        print("########## respet.recon.reconstruction.Reconstruction.createStaticCarney ##########")
        self.checkUmaps(self.muCarney(frames=[0]), fcomment)
        self.checkHistogramming(fcomment)
        wtime = self.getWTime(self.json_filename_with(ac=False))
        return self.createStatic(self.muCarney(frames=[0]), wtime, time0, timeF, fcomment=fcomment)

    def createPhantom(self, time0=None, timeF=None, fcomment='_createPhantom'):
        print("########## respet.recon.reconstruction.Reconstruction.createPhantom ##########")
        self.phantom = True
        self.checkUmaps(self.muCarney(), fcomment)
        self.checkHistogramming(fcomment)
        return self.createStatic(self.muCarney(), 0, time0, timeF, fcomment=fcomment)

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
        offset = self.getWTime(self.json_filename_with(ac=False))
        return self.createDynamic2(max(offset, self.minTime), taus, self.getTaus2(), fcomment)

    def createStatic(self, muo, wtime=0, time0=None, timeF=None, fcomment='_createStatic'):
        """
        :param muo       mu-map of imaged object:
        :param wtime is determined by createDynamic:
        :param time0     int sec:
        :param timeF     int sec:
        :param fcomment  string for naming subspace:
        :return          result from nipet.mmrchain:
        :rtype           dictionary:
        """
        from niftypet import nipet
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['DCYCRR'] = self.DCYCRR

        if self.reconstruction_started:
            logging.debug('reconstruction.Reconstruction.createDynamics.reconstruction_started == True')
            return None # to avoid race-conditions in parallel computing contexts

        self._do_touch_file('_started')
        if not time0:
            time0 = self.getTime0()
        time0 = min(wtime+time0, self.getTimeMax())
        if not timeF:
            timeF = self.getTimeF()
        timeF = min(wtime+timeF, self.getTimeMax())
        sta = nipet.mmrchain(self.datain, self.mMRparams,
                             frames    = ['fluid', [time0, timeF]],
                             mu_h      = self.muHardware(),
                             mu_o      = muo,
                             itr       = self.getItr(),
                             fwhm      = self.getFwhm(),
                             recmod    = self.recmod,
                             outpath   = self.outpath,
                             store_img = True,
                             fcomment  = fcomment)
        self.save_json(np.int_([timeF-time0]), waittime=wtime)
        self._do_touch_file('_finished')
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
            logging.debug('reconstruction.Reconstruction.createDynamics.reconstruction_started == True')
            return None # to avoid race-conditions in parallel computing contexts

        self._do_touch_file('_started')
        dynFrame = None
        times,taus = self.getTimes(taus) # length(times) == length(taus) + 1; revise taus using NIPET metrics
        print("times->" + str(times))
        print("taus->" + str(taus))
        wtime = times[0] # time to wait for nans to clear
        it_fin = None # passed to save_json()
        for it in np.arange(1, times.shape[0]):
            try:
                if self.frame_exists(times[it-1], times[it], fcomment, it):
                    continue # and reuse existings reconstructions

                logging.info('createDynamic:  frame samples {}-{} s;'.format(times[it-1], times[it]))
                logging.debug('createDynamic.datain->')
                logging.debug(self.datain)
                logging.debug('createDynamic.mMRparams->')
                logging.debug(self.mMRparams)

                dynFrame = nipet.mmrchain(self.datain, self.mMRparams,
                                          frames    = ['fluid', [times[it-1], times[it]]],
                                          mu_h      = self.muHardware(),
                                          mu_o      = muo,
                                          itr       = self.getItr(),
                                          fwhm      = self.getFwhm(),
                                          recmod    = self.recmod,
                                          outpath   = self.outpath,
                                          store_img = True,
                                          fcomment  = fcomment + '_time' + str(it-1))
                it_fin = it
                if isnan(dynFrame['im']).any():
                    if times[it] < times[-1] / 2:
                        wtime = times[it]
            except (UnboundLocalError, IndexError) as e:
                warn(e.message)
                if times[it] < times[-1]/2:
                    self.replaceFrameInSitu(times[it-1], times[it], fcomment, it-1)
                    wtime = times[it]
                else:
                    warn('createDynamic:  break for it->' + str(it))
                    break

        self.save_json(taus[:it_fin], waittime=wtime)
        self._do_touch_file('_finished')
        return dynFrame

    def createDynamic2(self, offset, taus, taus2, fcomment='_createDynamic2'):
        """
        :param offset is determined externaly by createDynamic():
        :param taus   np.int_ for mu-map frames:
        :param taus2  np.int_ for emission frames:
        :param muo    3D or 4D mu-map of imaged object:
        :return       last result from nipet.mmrchain:
        :rtype        dictionary:
        """
        global dynFrame
        from numpy import isnan
        from niftypet import nipet
        from warnings import warn
        self.mMRparams['Cnt']['VERBOSE'] = self.verbose
        self.mMRparams['Cnt']['DCYCRR'] = self.DCYCRR

        if self.reconstruction_started:
            logging.debug('reconstruction.Reconstruction.createDynamics2.reconstruction_started == True')
            return None # to avoid race-conditions in parallel computing contexts

        self._do_touch_file('_started')
        dynFrame = None
        times,trash = self.getTimes(taus) # of umap alignments
        times2,taus2 = self.getTimes(taus2, offset=offset)
        print("times->" + str(times))
        print("trash->" + str(trash))
        print("times2->" + str(times2))
        print("taus2->" + str(taus2))
        wtime2 = times2[0]
        it2_fin = None
        it = 1                                     # right edge of mu-map frame
        for it2 in np.arange(1, times2.shape[0]):  # right edge of hist frame to be attenuation corrected
            try:
                while times[it] < times2[it2-1] and times[it] < times[-1]:
                    it += 1 # select the mu-map for the hist in:  [times2[it2-1], times2[it2]]

                if self.frame_exists(times2[it2-1], times2[it2], fcomment, it2):
                    continue # and reuse existings reconstructions

                logging.info('createDynamic2:  AC frame samples {}-{} s; NAC frame samples {}-{} s'.format(times2[it2-1], times2[it2], times[it-1], times[it]))
                logging.debug('reconstruction.Reconstruction.createDynamic2.datain->')
                logging.debug(self.datain)
                logging.debug('reconstruction.Reconstruction.createDynamic2.mMRparams->')
                logging.debug(self.mMRparams)

                dynFrame = nipet.mmrchain(self.datain, self.mMRparams,
                                          frames    = ['fluid', [times2[it2-1], times2[it2]]],
                                          mu_h      = self.muHardware(),
                                          mu_o      = self.muCarney(frames=(it-1)),
                                          itr       = self.getItr(),
                                          fwhm      = self.getFwhm(),
                                          recmod    = self.recmod,
                                          outpath   = self.outpath,
                                          store_img = True,
                                          fcomment  = fcomment + '_time' + str(it2-1))
                it2_fin = it2
                if isnan(dynFrame['im']).any():
                    if times2[it2] < times2[-1]:
                        wtime2 = times2[it2]
            except (UnboundLocalError, IndexError) as e:
                warn(e.message)
                if times2[it2] < times2[-1]:
                    self.replaceFrameInSitu(times2[it2-1], times2[it2], fcomment, it2-1)
                    wtime2 = times2[it2]
                else:
                    warn('Reconstruction.createDynamic2:  break for it2->' + str(it2))
                    break

        self.save_json(taus2[:it2_fin], offsettime=offset, waittime=(wtime2-offset))
        self._do_touch_file('_finished')
        return dynFrame

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
        si = self.histogram_si

        # prompt sinogram
        figure()
        matshow(hst['psino'][si, :, :], cmap='inferno')
        colorbar()
        xlabel('bins')
        ylabel('angles')
        savefig(os.path.join(self.outpath, fcomment+'_promptsino.pdf'))

        # prompt sinogram oblique
        figure()
        matshow(hst['psino'][si+128, :, :], cmap='inferno')
        colorbar()
        xlabel('bins')
        ylabel('angles')
        savefig(os.path.join(self.outpath, fcomment+'_promptsino_oblique.pdf'))

        # delayed sinogram
        figure()
        matshow(hst['dsino'][si, :, :], cmap='inferno')
        colorbar()
        xlabel('bins')
        ylabel('angles')
        savefig(os.path.join(self.outpath, fcomment+'_delayedsino.pdf'))

        # delayed sinogram oblique
        figure()
        matshow(hst['dsino'][si+128, :, :], cmap='inferno')
        colorbar()
        xlabel('bins')
        ylabel('angles')
        savefig(os.path.join(self.outpath, fcomment+'_delayedsino_oblique.pdf'))

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

    def checkScattering(self, fcomment=''):
        from niftypet import nipet
        from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, savefig, matshow, colorbar, legend, grid

        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)

        # scattering
        # I don't recommend using it for dynamic scans, but static only, as it drains the memory big time:
        recon = nipet.mmrchain(
            self.datain, self.mMRparams,
            mu_h=self.muHardware(),
            mu_o=self.muCarney(frames=1),
            itr=2,
            fwhm=0.0,
            outpath=self.outpath,
            fcomment='_scattering',
            ret_sinos=True,
            store_img=True)

        # Then you sum up all sinograms to see the average performace:
        ssn = np.sum(recon['sinos']['ssino'], axis=(0, 1))
        psn = np.sum(recon['sinos']['psino'], axis=(0, 1))
        rsn = np.sum(recon['sinos']['rsino'], axis=(0, 1))
        msk = np.sum(recon['sinos']['amask'], axis=(0, 1))

        # plotting the sinogram profiles for angle indexes 128 and 196:
        figure()
        ia = 128
        plot(psn[ia, :], label='prompts')
        plot(rsn[ia, :], label='randoms')
        plot(rsn[ia, :] + ssn[ia, :], label='scatter+randoms')
        plot(msk[ia, :], label='mask')
        legend()
        savefig(os.path.join(self.outpath, fcomment + '_scattering128.pdf'))
        figure()
        ia = 196
        plot(psn[ia, :], label='prompts')
        plot(rsn[ia, :], label='randoms')
        plot(rsn[ia, :] + ssn[ia, :], label='scatter+randoms')
        plot(msk[ia, :], label='mask')
        legend()
        savefig(os.path.join(self.outpath, fcomment + '_scattering196.pdf'))

        return ssn, psn, rsn, msk

    def checkTimeAliasingCarney(self, fcomment='_checkTimeAliasingCarney'):
        times,trash = self.getTimes(self.getTaus())
        print("########## respet.recon.reconstruction.Reconstruction.checkTimeAliasingCarney ##########")
        print(times[0:2])
        return self.createDynamic(self.getTaus()[0:2], self.muCarney(frames=[0,1]), fcomment)

    def checkTimeHierarchiesCarney(self, fcomment='_checkTimeHierarchiesCarney'):
        times,trash = self.getTimes(self.getTaus())
        times2,trash = self.getTimes(self.getTaus2())
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


    def emissionsScatterThresh(self):
        """ provisions Cnt['ETHRLD'] for use by:
         mmrrec.py lines 208, 272 228; mmrimg.py 209; sct_module.cu line 302 """
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            thresh = 0.05
        elif self.tracerMemory.lower() == 'oxygen-water':
            thresh = 0.05
        elif self.tracerMemory.lower() == 'oxygen' or self.tracerMemory.lower() == 'carbon':
            thresh = 0.05
        else:
            raise AssertionError('Reconstruction.emissionsScatterThresh does not support tracerMemory->' + self.tracerMemory)
        return thresh

    def frame_exists(self, t0, tf, fcomment, it2):
        """
        e.g., a_itr-4_t-577-601sec_createDynamic2Carney_time57.nii.gz
        :param t0:
        :param tf:
        :param fcomment:
        :param it2:
        :return bool:
        """
        fn = "a_itr-" + str(self.getItr()) + "_t-" + str(t0) + "-" + str(tf) + "sec" + fcomment + "_time" + str(it2-1) + ".nii.gz"
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

    def getFwhm(self):
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            fwhm = 4.3 / 2.08626  # number of voxels;  https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
        elif self.tracerMemory.lower() == 'oxygen-water':
            fwhm = 4.3 / 2.08626
        elif self.tracerMemory.lower() == 'carbon' or self.tracerMemory.lower() == 'oxygen':
            fwhm = 0
        else:
            raise AssertionError('Reconstruction.getFwhm does not support tracerMemory->' + self.tracerMemory)
        return fwhm

    def getInterfile(self, dcm):
        """
        :param dcm:
        :return lm_dict, a dictionary of interfile fields:
        """
        from interfile import Interfile
        from warnings import warn
        try:
            try:
                lm_dict = Interfile.load(dcm)
            except Interfile.ParsingError as e:
                warn(e.message)
                return None
        except (AttributeError, TypeError):
            raise AssertionError('dcm must be a filename')
        return lm_dict

    def getItr(self):
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            itr = 4
        elif self.tracerMemory.lower() == 'oxygen-water' or self.tracerMemory.lower() == 'carbon':
            itr = 4
        elif self.tracerMemory.lower() == 'oxygen':
            itr = 2
        else:
            raise AssertionError('Reconstruction.getItr does not support tracerMemory->' + self.tracerMemory)
        return itr

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

    def getSpan(self):
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            span = 11
        elif self.tracerMemory.lower() == 'oxygen-water' or self.tracerMemory.lower() == 'carbon':
            span = 11
        elif self.tracerMemory.lower() == 'oxygen':
            span = 11
        else:
            raise AssertionError('Reconstruction.getSpan does not support tracerMemory->' + self.tracerMemory)
        return span

    def getTaus(self, json_file=None):
        """ see also mfiles/t0_and_dt.m
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
            taus = np.int_([30,35,39,43,47,51,55,59,64,68,72,76,81,85,89,93,98,102,106,111,115,120,124,129,133,138,142,147,151,156,161,165,170,175,171])
            # len == 35, nudge = 4, dur == 3601
        elif self.tracerMemory.lower() == 'oxygen-water' or self.tracerMemory.lower() == 'carbon' or self.tracerMemory.lower() == 'oxygen':
            taus = np.int_([12,13,14,15,17,18,20,23,26,30,35,43,55,75,114,91])
            # len == 16, dur == 601
        else:
            raise AssertionError('Reconstruction.getTaus does not support tracerMemory->' + self.tracerMemory)
        return taus

    def getTaus2(self, json_file=None):
        """ see also mfiles/t0_and_dt.m
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
            taus = np.int_([10,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44,46,47,49,50,52,53,56,57,59,60,62,63,65,66,68,69,71,72,74,76,78,79,81,82,84,85,87,88,91,92,94,95,97,98,100,101,104,105,108])
            # len == 62, nudge = 1.5, dur == 3601
        elif self.tracerMemory.lower() == 'oxygen-water':
            taus = np.int_([3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,8,8,8,9,9,10,10,11,11,12,13,14,15,16,18,20,22,25,29,34,41,51,52])
            # len == 54, dur == 601
        elif self.tracerMemory.lower() == 'carbon':
            # taus = np.int_([5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,9,9,9,10,11,11,12,13,14,15,16,18,20,22,25,29,34,41,52,137])
            # len == 35, dur == 601
            taus = np.int_([3,3,3,3,3,3,3,3,5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,8,9,9,10,10,11,11,12,13,14,15,16,18,19,22,24,28,33,39,49,64,49])
            # len = 45, dur = 601
        elif self.tracerMemory.lower() == 'oxygen':
            taus = np.int_([2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,8,9,9,10,10,15])
            # len = 63, dur = 301
        else:
            raise AssertionError('Reconstruction.getTaus2 does not support tracerMemory->' + self.tracerMemory)
        return taus

    def getTime0(self):
        if self.phantom:
            return 0
        times,trash = self.getTimes(self.getTaus())
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            return times[-1] - 20*60
        elif self.tracerMemory.lower() == 'oxygen-water':
            return times[0]
        elif self.tracerMemory.lower() == 'carbon':
            return times[0] + 2*60
        elif self.tracerMemory.lower() == 'oxygen':
            return times[0]
        else:
            raise AssertionError('Reconstruction.getTime0 does not support tracerMemory->' + self.tracerMemory)

    def getTimeF(self):
        if self.phantom:
            return self.getTimeMax()
        times,trash = self.getTimes(self.getTaus())
        if self.tracerMemory.lower() == 'fluorodeoxyglucose':
            return times[-1]
        elif self.tracerMemory.lower() == 'oxygen-water':
            return times[0] + 60
        elif self.tracerMemory.lower() == 'carbon':
            return times[0] + 3*60
        elif self.tracerMemory.lower() == 'oxygen':
            return times[0] + 60
        else:
            raise AssertionError('Reconstruction.getTimeF does not support tracerMemory->' + self.tracerMemory)

    def getTimeMax(self):
        """
        :return:  max time available from listmode data in sec.
        """
        from niftypet.nipet.lm import mmr_lmproc #CUDA
        nele, ttags, tpos = mmr_lmproc.lminfo(self.datain['lm_bf'])
        return (ttags[1]-ttags[0]+999)/1000 # sec

    def getTimes(self, taus=None, offset=0):
        """
        :param:  offset is predetermined duration to exclude from times
        :return:  array of times including 0 and np.cumsum(taus); max(times) == getTimeMax()
        :return:  array of taus revised to be consistent with getTimeMax()
        :rtype:  numpy.int_
        """
        if not isinstance(taus, np.ndarray):
            raise AssertionError('Reconstruction.getTimes.taus is missing')
        tmax = self.getTimeMax()
        t = np.hstack((np.int_(0), np.cumsum(taus)))
        t = t + offset
        t = t[t < tmax]
        t = np.hstack((t, np.int_(tmax)))
        taus = t[1:] - t[:-1]
        return np.int_(t), np.int_(taus)

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
        logging.debug('reconstruction.Reconstruction.open_json.jt->')
        logging.debug(str(jt))
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
        logging.debug('reconstruction.Reconstruction.save_json.jdict->')
        logging.debug(str(jdict))
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
        if lm_dict:
            return lm_dict['image duration']['value'] # sec
        else:
            return self.getTaus().sum()

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

    def adjust_mirror_hdw_mumap(self, hmu_dct):
        import nibabel as nib

        if not self.tracerMemory.lower() == 'oxygen':
            return hmu_dct

        nim = nib.load(os.path.join(os.getenv('HARDWAREUMAPS'), 'adhoc_mu.nii.gz'))
        imo = nim.get_data()
        hmu = np.transpose(imo[:,::-1,::-1], (2, 1, 0))
        hmu_dct = {'im': hmu+hmu_dct['im'],
                   'fim': hmu_dct['fim'],
                   'affine': hmu_dct['affine']}
        return hmu_dct

    def muHardware(self):
        """
        :return:  dictionary for hardware mu-map image provided by nipet.hdw_mumap.  Keys:  'im', ...
        See also self.hmuSelection.
        """
        from niftypet import nipet
        if self.use_stored_hdw_mumap:
            self.datain['hmumap'] = os.path.join(
                os.getenv('HARDWAREUMAPS'), 'hmumap.npy')
        if not self.cached_hdw_mumap:
            self.cached_hdw_mumap = nipet.hdw_mumap(
                self.datain, self.hmuSelection, self.mMRparams, outpath=self.outpath, use_stored=self.use_stored_hdw_mumap)
        if self.use_mirror_hdw_mumap:
            self.cached_hdw_mumap = self.adjust_mirror_hdw_mumap(self.cached_hdw_mumap)
        logging.debug('reconstruction.Reconstruction.muHardware.datain[''hmumap'']->')
        logging.debug(self.datain['hmumap'])
        return self.cached_hdw_mumap

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
        logging.debug('reconstruction.Reconstruction.muCarney is')
        logging.debug('using ' + imtype + ' from NIfTI file.')
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
                            'a_itr-'+str(self.getItr())+'_t-'+str(t0)+'-'+str(t1)+'sec'+tag+'_time'+str(fr)+'.nii.gz')

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
        if self.phantom:
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

    def saveDynamicInMemory(self, dyn, mumaps, hst, fcomment=''):
        """
        :param dyn:       dictionary from nipet.mmrchain
        :param mumaps:    dictionary of mu-maps from imaged object, hardware
        :param hst:       dictionary from nipet.mmrhist
        :param fcomment:  string to append to canonical filename
        """
        fout = self._createFilename(fcomment)
        im = self._gatherOsemoneList(dyn)
        logging.info('reconstruction.Reconstruction.saveDynamicInMemory is')
        logging.info('saving ' + str(len(im.shape)) + 'D image to: ' + fout)

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
        logging.info('reconstruction.Reconstruction.saveStatic is')
        logging.info('saving 3D image to: ' + fout)

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
                 ';itr='   + str(self.getItr())   + \
                 ';fwhm='  + str(self.getFwhm())  + \
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
        self.mMRparams['Cnt']['SPN']     = self.getSpan()
        self.mMRparams['Cnt']['BTP']     = self.bootstrap
        self.mMRparams['Cnt']['DCYCRR']  = self.DCYCRR
        self.mMRparams['Cnt']['DEVID']   = self.DEVID
        self.mMRparams['Cnt']['ETHRLD']  = self.emissionsScatterThresh()
        self.datain = nipet.classify_input(self.tracerRawdataLocation, self.mMRparams)
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        logging.info("reconstruction.Reconstruction._initializeNiftypet.datain->")
        logging.info(self.datain)
            
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
                 'O15':{'BF':0.999, 'thalf':122.2416},
                 'C11':{'BF':0.998, 'thalf':20.38*60}}

    def __init__(self, prefix=None, umapSF='umapSynth', v=False, cndaDownload=None, devid=0, minTime=0, phantom=False, si=63):
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
        from niftypet.nipet.dinf import dev_info
        logging.info('reconstruction.Reconstruction.__init__')
        self._parse_prefix(prefix)
        logging.info('self.tracerRawdataLocation->' + self.tracerRawdataLocation)
        os.chdir(self.tracerRawdataLocation)
        logging.info('cwd->' + os.getcwd())
        self.umapSynthFileprefix = umapSF
        self.verbose = v
        self.phantom = phantom
        self.organizeRawdataLocation(cndaDownload)
        self.tracerMemory = self.lm_tracer()
        logging.info(str(dev_info(1)))
        self.DEVID = devid
        self._initializeNiftypet()
        self.minTime = minTime
        self.histogram_si = si



def main():
    import argparse, textwrap
    from niftypet.nipet.dinf import dev_info

    p = argparse.ArgumentParser(
        description='provides interfaces to https://github.com/pjmark/NIMPA.git, https://github.com/jjleewustledu/NIPET.git',
        usage=textwrap.dedent('''\
        
    python reconstruction.py -h
    nvidia-docker run -it \\
                  -v ${DOCKER_HOME}/hardwareumaps/:/hardwareumaps \\
                  -v ${SINGULARITY_HOME}/:/SubjectsDir \\
                  niftypetr-image:reconstruction:latest -h
    singularity exec \\
                --nv \\
                --bind $SINGULARITY_HOME/hardwareumaps:/hardwareumaps \\
                --bind $SINGULARITY_HOME:/SubjectsDir \\
                $SINGULARITY_HOME/niftypetr-image_reconstruction.sif \\
                "python" "/work/NiftyPETy/respet/recon/reconstruction.py" "-h" 
        '''),
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-m', '--method',
                   metavar='createDynamic|createStatic|createPhantom|info',
                   type=str,
                   default='createDynamic')
    p.add_argument('-p', '--prefix',
                   metavar='/path/to/experiment-NAC',
                   help='location containing tracer listmode and norm data',
                   type=str,
                   required=True)
    p.add_argument('-v', '--verbose',
                   metavar='true|false',
                   type=str,
                   default='false')
    p.add_argument('-g', '--gpu',
                   metavar='0',
                   help='device ID used by cudaSetDevice',
                   type=str,
                   default='0')
    p.add_argument('-t', '--minTime',
                   metavar='0',
                   help='min time for which emission reconstruction is performed',
                   type=str,
                   default='0')
    args = p.parse_args()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #os.environ["NVIDIA_VISIBLE_DEVICES"] = str(args.gpu)

    v = args.verbose.lower() == 'true'
    if args.method.lower() == 'createdynamic':
        print('main.args.method->createdynamic')
        r = Reconstruction(prefix=args.prefix, v=v, devid=int(args.gpu), minTime=int(args.minTime))
        if not r.ac:
            print('main.r.createDynamicNAC')
            r.createDynamicNAC(fcomment='_createDynamicNAC')
        else:
            print('main.r.createDynamic2Carney')
            r.createDynamic2Carney(fcomment='_createDynamic2Carney')
    elif args.method.lower() == 'createstatic':
        print('main.args.method->createstatic')
        r = Reconstruction(prefix=args.prefix, v=v, devid=int(args.gpu), minTime=int(args.minTime))
        if not r.ac:
            print('main.r.createStaticNAC')
            r.createStaticNAC(fcomment='_createStaticNAC')
        else:
            print('main.r.createStaticCarney')
            r.createStaticCarney(fcomment='_createStaticCarney')
    elif args.method.lower() == 'createphantom':
        print('main.args.method->createphantom')
        print('main.r.createPhantom')
        r = Reconstruction(prefix=args.prefix, v=v, devid=int(args.gpu), minTime=int(args.minTime), phantom=True)
        r.createPhantom(fcomment='_createPhantom')
    elif args.method.lower() == 'info':
        print('main.args.method->info')
        print(dev_info(1))
        print('\n')
        print(r.mMRparams)
        print('\n')
        print(r.datain)
        print('\n')

if __name__ == '__main__':
    main()
