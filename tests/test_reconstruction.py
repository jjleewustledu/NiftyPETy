import unittest
from respet.recon.reconstruction import Reconstruction
import os
import matplotlib.pyplot as plt

class TestReconstruction(unittest.TestCase):

    twiliteLoc    = '/home2/jjlee/Local/Pawel/NP995_24/V1/Twilite_V1-Converted'
    tracerLoc     = '/home2/jjlee/Local/Pawel/HYGLY50/V1/FDG_V1-Converted' # 20, 21, 22
    testObj = []

    # def setUp(self):
    #     self.testObj = Reconstruction(self.tracerLoc)
    #     self.testObj.itr = 4
    #     self.testObj.fwhm = 4.3/2.08626 # num. of voxels
    #     self.testObj.use_stored_hist = True

    #def tearDown(self):

    def _test_sampleStaticMethod(self):
        self.assertEqual(Reconstruction.sampleStaticMethod(), 0.1234)

    def test_installation(self):
        from niftypet import nipet
        nipet.gpuinfo(extended=True)

    def test_ctor(self):
        self.assertIsInstance(self.testObj, Reconstruction)
        from numpy import array
        from numpy.testing import assert_array_equal
        c = self.testObj.mMRparams['Cnt']
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
        self.assertEqual(c['BTP'], 0)
        assert_array_equal(c['IMSIZE'], array([127, 344, 344]))

    def test_data(self):
        self.testObj.printd(self.testObj.mMRparams['Cnt'])
        self.testObj.printd(self.testObj.datain)

    def test_locs(self):
        self.assertTrue(os.path.exists(self.twiliteLoc))
        self.assertTrue(os.path.exists(self.tracerLoc))



class TestTwilite(TestReconstruction):

    def _test_createTwiliteStaticNAC(self):
        obj = Reconstruction(self.twiliteLoc, v = True)
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def _test_createTwiliteStaticUTE(self):
        obj = Reconstruction(self.twiliteLoc, ac=True, v = True)
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def _test_createTwiliteStaticCarney(self):
        obj = Reconstruction(self.twiliteLoc, ac=True, v = True)
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()



class TestNAC(TestReconstruction):

    def _test_createTracerStaticNAC(self):
        obj = Reconstruction(self.tracerLoc, ac = False, v = True)
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def test_createTracerNAC(self):
        mids = ['HYGLY23/V1/FDG_V1', 'NP995_24/V1/FDG_V1', 'NP995_19/V2/FDG_V2', 'HYGLY48/V1/FDG_V1', 'HYGLY50/V1/FDG_V1', 'HYGLY47/V1/FDG_V1' ]
        m = mids[0]
        loc = '/home2/jjlee/Local/Pawel/'+m+'-Converted'
        obj = Reconstruction(loc, ac = False, v = True)
        dyn = obj.createDynamicNAC(fcomment='_createDynamicNAC')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()



class TestUTE(TestReconstruction):

    def test_createTracerStaticUTE(self):
        obj = Reconstruction(self.tracerLoc, ac=True, v=True)
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def test_createTracerUTE(self):
        obj = Reconstruction(self.tracerLoc, ac=True, v=True)
        dyn = obj.createDynamicUTE(fcomment='_createDynamicUTE')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()



class TestCarney(TestReconstruction):

    def test_createTracerStaticCarney(self):
        obj = Reconstruction(self.tracerLoc, ac=True, v=True)
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def test_createTracerCarney(self):
        mids = ['HYGLY23/V1/FDG_V1', 'HYGLY23/V2/FDG_V2']
        for m in mids:
            loc = '/home2/jjlee/Local/Pawel/'+m+'-Converted'
            obj = Reconstruction(loc, ac=True, v=True)
            obj.createDynamic2Carney(fcomment='_createDynamic2Carney')



class TestOtherUmaps(TestReconstruction):

    def _test_createUmapSynthFullBlurred_tracer(self):
        obj = Reconstruction(
            self.tracerLoc,
            '/data/nil-bluearc/raichle/PPGdata/jjlee2/HYGLY23/V2/FDG_V2-NAC')
        if not os.path.isfile(obj.umapSynthFileprefix + '.nii.gz'):
            obj.createUmapSynthFullBlurred()
        else:
            print(obj.umapSynthFileprefix + '.nii.gz already exists')



class TestTimes(TestReconstruction):

    def test_getTimeMax(self):
        obj = Reconstruction(self.tracerLoc, ac=True)
        self.assertEqual(obj.getTimeMax(), 3601)

    def test_getTimes(self):
        obj = Reconstruction(self.tracerLoc, ac=True)
        print(obj.getTimes())

    def _test_checkTimeAliasingUTE(self):
        obj = Reconstruction(self.tracerLoc, ac=True, v=True)
        dyn = obj.checkTimeAliasingUTE(fcomment='_checkTimeAliasingUTE')
        plt.matshow(dyn[0]['im'][60,:,:])
        plt.matshow(dyn[1]['im'][60,:,:])
        plt.show()

    def _test_checkTimeAliasingCarney(self):
        obj = Reconstruction(self.tracerLoc, ac=True)
        dyn = obj.checkTimeAliasingCarney(fcomment='_checkTimeAliasingCarney')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()

    def _test_checkTimeHierarchiesCarney(self):
        obj = Reconstruction(self.tracerLoc, ac=True)
        dyn = obj.checkTimeHierarchiesCarney(fcomment='_checkTimeHierarchiesCarney')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()



class TestMatlab(TestReconstruction):

    def test_MCRtoys(self):
        import MCRtoys

    def test_constructResolved(self):
        import constructResolved



# N.B.:  duplicates unittest actions within pycharm
#suite = unittest.TestLoader().loadTestsFromTestCase(TestReconstruction)
#unittest.TextTestRunner(verbosity=2).run(suite)
