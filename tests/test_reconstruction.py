import unittest
import respet
import matplotlib.pyplot as plt
import os
import numpy as np

class TestReconstruction(unittest.TestCase):

    twiliteLoc = '/home2/jjlee/Local/Pawel/HYGLY23/V2/Twilite_V2-NiftyPETy'
    tracerLoc  = '/home2/jjlee/Local/Pawel/HYGLY23/V2/FDG_V2-NiftyPETy'
    testObj = []

    @classmethod
    def setUpClass(self):
        self.testObj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)
        self.testObj.itr = 5
        self.testObj.fwhm = 4.3/2.08626 # num. of voxels
        self.testObj.use_stored_hist = True

    #@classmethod
    #def tearDownClass(self):

    def _test_sampleStaticMethod(self):
        self.assertEqual(respet.recon.reconstruction.Reconstruction.sampleStaticMethod(), 0.1234)

    def test_locs(self):
        self.assertTrue(os.path.exists(self.twiliteLoc))
        self.assertTrue(os.path.exists(self.tracerLoc))

    def test_ctor(self):
        self.assertIsInstance(self.testObj, respet.recon.reconstruction.Reconstruction)
        from numpy import array
        from numpy.testing import assert_array_equal
        c = self.testObj._constants
        self.assertEqual(c['HMULIST'], ['umap_HNMCL_10606489.v.hdr', 'umap_HOMCU_10606489.v.hdr', 'umap_SPMC_10606491.v.hdr', 'umap_PT_2291734.v.hdr', 'umap_HNMCU_10606489.v.hdr', 'umap_BR4CH_10185525.v.hdr'])
        self.assertEqual(c['NSRNG'], 8)
        self.assertEqual(c['NSN11'], 837)
        self.assertEqual(c['NRNG'], 64)
        self.assertEqual(c['NBCKT'], 224)
        self.assertEqual(c['SCTSCLMU'], [0.49606299212598426, 0.5, 0.5])
        self.assertEqual(c['ISOTOPE'], 'F18')
        self.assertEqual(c['SPN'], 1)
        assert_array_equal(c['SCTRNG'], array([ 0, 10, 19, 28, 35, 44, 53, 63], dtype='int16'))
        self.assertEqual(c['NSN64'], 4096)
        self.assertEqual(c['CWND'], 5.85938e-09)
        self.assertEqual(c['SCTSCLEM'], [0.33858267716535434, 0.3313953488372093, 0.3313953488372093])
        self.assertEqual(c['BTP'], 2)
        assert_array_equal(c['IMSIZE'], array([127, 344, 344]))
        #self.assertDictEqual(self.testObj._datain, {'em_nocrr': '', 'lm_bf': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/LM/1.3.12.2.1107.5.2.38.51010.30000017120616470612500000022.bf', 'mumapDCM#': 128, 'mumapUTE': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/mumap_obj/mumapUTE.npy', 'mumapCT': '', 'lm_dcm': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/LM/1.3.12.2.1107.5.2.38.51010.30000017120616470612500000022.dcm', 'MRT2W': '', 'pCT': '', 'nrm_dcm': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/norm/1.3.12.2.1107.5.2.38.51010.30000017120616470612500000021.dcm', 'T1nii': '', 'corepath': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy', 'lm_ima': '', 'MRT2WN': 0, 'sinos': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/LM/sinos_s11_n1_frm(0-0).npy', 'MRT1W#': 0, 'nrm_ima': '', 'nrm_bf': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/norm/1.3.12.2.1107.5.2.38.51010.30000017120616470612500000021.bf', 'hmumap': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/mumap_hdw/hmumap.npy', 'T1lbl': '', 'MRT1W': '', 'mumapDCM': '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy/umap', 'em_crr': '', 'T1bc': ''})

    def test_time_diff_norm_acq(self):
        from niftypet import nipet
        cnt, txLUG, axLUT = nipet.mmraux.mmrinit()
        datain = nipet.mmraux.explore_input(self.twiliteLoc, cnt)
        nipet.mmraux.time_diff_norm_acq(datain)
        # normal result is silent

    def test_getTimeMax(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        self.assertEqual(obj.getTimeMax(), 3601)

    def _test_getTimes(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        print(obj.getTimes())

    def test_createTwiliteStaticNAC(self):
        sta = self.testObj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTwiliteStaticUTE(self):
        sta = self.testObj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTwiliteStaticCarney(self):
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc, umapSF='umapSynth_b43_on_createStaticNAC')
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerStaticNAC(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerStaticUTE(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerStaticCarney(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.itr = 3
        obj.fwhm = 0
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerNAC(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        dyn = obj.createDynamicNAC(fcomment='_createDynamicNAC')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        #plt.show()

    def _test_createTracerUTE(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        dyn = obj.createDynamicUTE(fcomment='_createDynamicUTE')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def _test_checkTimeAliasingUTE(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        dyn = obj.checkTimeAliasingUTE(fcomment='_checkTimeAliasingUTE')
        plt.matshow(dyn[0].im[60,:,:])
        plt.matshow(dyn[1].im[60,:,:])
        plt.show()

    def _test_checkTimeAliasingCarney(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        dyn = obj.checkTimeAliasingCarney(fcomment='_checkTimeAliasingCarney')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def _test_checkTimeHierarchiesCarney(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        dyn = obj.checkTimeHierarchiesCarney(fcomment='_checkTimeHierarchiesCarney')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def _test_createTracerCarney(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        dyn = obj.createDynamic2Carney(fcomment='_createDynamic2Carney')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def _test_createUmapSynthFullBlurred_tracer(self):
        obj = respet.recon.reconstruction.Reconstruction(
            self.tracerLoc,
            '/data/nil-bluearc/raichle/PPGdata/jjlee2/HYGLY23/V2/FDG_V2-NAC')
        if not os.path.isfile(obj.umapSynthFileprefix + '.nii.gz'):
            obj.createUmapSynthFullBlurred()
        else:
            print(obj.umapSynthFileprefix + '.nii.gz already exists')

    # def test_custom_mumap(self):
    #     mu = self.testObj.custom_mumap([],
    #                                    fileprefix=os.path.join(self.twiliteLoc, 'mumap_obj', 'mumap_fromDCM.nii.gz'))
    #     mu0 = np.load(os.path.join(self.twiliteLoc, 'mumap_obj', 'mumapUTE.npy'))
    #     self.assertAlmostEqual(mu, mu0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestReconstruction)
unittest.TextTestRunner(verbosity=2).run(suite)
