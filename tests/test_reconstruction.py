import unittest
import respet
import matplotlib.pyplot as plt
import os
import numpy as np

class TestReconstruction(unittest.TestCase):

    twiliteLoc = '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy'
    tracerLoc  = '/home2/jjlee/Local/Pawel/HYGLY36/V3/FDG_V3-NiftyPETy'
    testObj = []

    #def setUp(self):
        #os.chdir(self.twiliteLoc)
        #self.testObj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)

    #def tearDown(self):

    def _test_sampleStaticMethod(self):
        self.assertEqual(respet.recon.reconstruction.Reconstruction.sampleStaticMethod(), 0.1234)

    def _test_locs(self):
        self.assertTrue(os.path.exists(self.twiliteLoc))
        self.assertTrue(os.path.exists(self.tracerLoc))

    def _test_ctor(self):
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)
        self.assertIsInstance(obj, respet.recon.reconstruction.Reconstruction)
        print(obj._constants)
        print(obj._datain)

    def _test_time_diff_norm_acq(self):
        import nipet
        cnt, txLUG, axLUT = nipet.mmraux.mmrinit()
        datain = nipet.mmraux.explore_input(self.twiliteLoc, cnt)
        nipet.mmraux.time_diff_norm_acq(datain)
        # normal result is silent

    def _test_getTimeMax(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        print(obj.getTimeMax())

    def _test_getTimes(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        print(obj.getTimes())

    def _test_createTwiliteStaticNAC(self):
        os.chdir(self.twiliteLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)
        obj.verbose = False
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTwiliteStaticUTE(self):
        os.chdir(self.twiliteLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTwiliteStaticCarney(self):
        os.chdir(self.twiliteLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc, umapSF='umapSynth_b43_on_createStaticNAC')
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerStaticNAC(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerStaticUTE(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerStaticCarney(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta.im[60,:,:])
        plt.matshow(sta.im[:,170,:])
        plt.matshow(sta.im[:,:,170])
        plt.show()

    def _test_createTracerNAC(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        dyn = obj.createDynamicNAC(fcomment='_createDynamicNAC')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        #plt.show()

    def _test_createTracerUTE(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        dyn = obj.createDynamicUTE(fcomment='_createDynamicUTE')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def _test_checkTimeAliasingUTE(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        dyn = obj.checkTimeAliasingUTE(fcomment='_checkTimeAliasingUTE')
        plt.matshow(dyn[0].im[60,:,:])
        plt.matshow(dyn[1].im[60,:,:])
        plt.show()

    def _test_checkTimeAliasingCarney(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        dyn = obj.checkTimeAliasingCarney(fcomment='_checkTimeAliasingCarney')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def test_checkTimeHierarchiesCarney(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        dyn = obj.checkTimeHierarchiesCarney(fcomment='_checkTimeHierarchiesCarney')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    def _test_createTracerCarney(self):
        os.chdir(self.tracerLoc)
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        obj.verbose = False
        dyn = obj.createDynamic2Carney(fcomment='_createDynamic2Carney')
        plt.matshow(dyn.im[60,:,:])
        plt.matshow(dyn.im[:,170,:])
        plt.matshow(dyn.im[:,:,170])
        plt.show()

    # def test_custom_mumap(self):
    #     mu = self.testObj.custom_mumap([],
    #                                    fileprefix=os.path.join(self.twiliteLoc, 'mumap_obj', 'mumap_fromDCM.nii.gz'))
    #     mu0 = np.load(os.path.join(self.twiliteLoc, 'mumap_obj', 'mumapUTE.npy'))
    #     self.assertAlmostEqual(mu, mu0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestReconstruction)
unittest.TextTestRunner(verbosity=2).run(suite)
