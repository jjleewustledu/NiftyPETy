import unittest
import respet
import matplotlib.pyplot as plt
import os
import numpy as np

class TestReconstruction(unittest.TestCase):

    twiliteLoc = '/home2/jjlee/Local/Pawel/HYGLY36/V3/Twilite_V3-NiftyPETy'
    tracerLoc  = '/home2/jjlee/Local/Pawel/HYGLY36/V3/FDG_V3-NiftyPETy'
    testObj = []

    def setUp(self):
        os.chdir(self.twiliteLoc)
        #self.testObj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)

    #def tearDown(self):

    def test_sampleStaticMethod(self):
        self.assertEqual(respet.recon.reconstruction.Reconstruction.sampleStaticMethod(), 0.1234)

    def test_locs(self):
        self.assertTrue(os.path.exists(self.twiliteLoc))
        self.assertTrue(os.path.exists(self.tracerLoc))

    def __test_ctor(self):
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)
        self.assertIsInstance(obj, respet.recon.reconstruction.Reconstruction)
        print(obj._constants)
        print(obj._datain)

    def __test_time_diff_norm_acq(self):
        import nipet
        cnt, txLUG, axLUT = nipet.mmraux.mmrinit()
        datain = nipet.mmraux.explore_input(self.twiliteLoc, cnt)
        nipet.mmraux.time_diff_norm_acq(datain)
        # normal result is silent

    def __test_getTimeMax(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        print(obj.getTimeMax())

    def __test_getTimes(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        print(obj.getTimes())

    def __test_createTwiliteNAC(self):
        obj = respet.recon.reconstruction.Reconstruction(self.twiliteLoc)
        nac = obj.createNAC()
        plt.matshow(nac.im[60,:,:])
        plt.matshow(nac.im[:,170,:])
        plt.matshow(nac.im[:,170,:])

    def __test_createTracerNAC(self):
        obj = respet.recon.reconstruction.Reconstruction(self.tracerLoc)
        nac = obj.createDynamicNAC()
        plt.matshow(nac[19].im[60,:,:])
        plt.matshow(nac[19].im[:,170,:])
        plt.matshow(nac[19].im[:,170,:])

    # def test_custom_mumap(self):
    #     mu = self.testObj.custom_mumap([],
    #                                    fileprefix=os.path.join(self.twiliteLoc, 'mumap_obj', 'mumap_fromDCM.nii.gz'))
    #     mu0 = np.load(os.path.join(self.twiliteLoc, 'mumap_obj', 'mumapUTE.npy'))
    #     self.assertAlmostEqual(mu, mu0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestReconstruction)
unittest.TextTestRunner(verbosity=2).run(suite)

