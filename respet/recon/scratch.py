import os
import numpy as np

class Scratch:
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2018"

    '''
    tracerRawdataLocation = ''
    umapFolder = 'umap'
    umapSynthFileprefix = 'umapSynth_full_frame'
    frameSuffix = '_frame'
    verbose = True
    _frame = 0
    _umapIdx = 0
    _t0 = 0
    _t1 = 0
    '''

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

    def replaceFrameInSitu(self, t0, t1, tag, fr):
        from shutil import copyfile
        copyfile(
            os.path.join(os.getenv('SUBJECTS_DIR'), 'zero_frame.nii.gz'),
            self.nipetFrameFilename(t0, t1, fr))

    def nipetFrameFilename(self, t0, t1, tag, fr):
        #       a_itr-4_t-10-20sec_createDynamic2Carney_time1.nii.gz
        return 'a_itr-'+str(self.itr)+'_t-'+str(t0)+'-'+str(t1)+'sec'+tag+'_time'+str(fr)+'.nii.gz'


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
        assert isinstance(dynFrame, dict)
        return dynFrame