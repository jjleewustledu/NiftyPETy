import unittest
from respet.recon.reconstruction import Reconstruction
import os

class TestReconstruction(unittest.TestCase):

    twiliteLoc    = '/home2/jjlee/Docker/NiftyPETd/HYGLY30/V2/Twilite_V2-Converted'
    tracerLoc     = '/home2/jjlee/Docker/NiftyPETd/HYGLY30/V2/FDG_V2-Converted' # 20, 21, 22

    #def setUp(self):

    #def tearDown(self):

    def theTestObj(self):
        obj = Reconstruction(self.tracerLoc)
        obj.itr = 4
        obj.fwhm = 4.3 / 2.08626 # num. of voxels
        obj.use_stored_hist = True
        return obj

    def test_sampleStaticMethod(self):
        self.assertEqual(Reconstruction.sampleStaticMethod(), 0.1234)

    def test_installation(self):
        from niftypet import nipet
        nipet.gpuinfo(extended=True)
        print('\n')

    def test_ctor(self):
        self.assertIsInstance(self.theTestObj, Reconstruction)
        from numpy import array
        from numpy.testing import assert_array_equal
        c = self.theTestObj.mMRparams['Cnt']
        self.assertEqual(c['HMULIST'], ['umap_HNMCL_10606489.v.hdr', 'umap_HNMCU_10606489.v.hdr', 'umap_SPMC_10606491.v.hdr', 'umap_PT_2291734.v.hdr', 'umap_HOMCU_10606489.v.hdr', 'umap_BR4CH_10185525.v.hdr'])
        self.assertEqual(c['NSRNG'], 8)
        self.assertEqual(c['NSN11'], 837)
        self.assertEqual(c['NRNG'], 64)
        self.assertEqual(c['NBCKT'], 224)
        self.assertEqual(c['SCTSCLMU'], [0.49606299212598426, 0.5, 0.5])
        self.assertEqual(c['ISOTOPE'], 'F18')
        self.assertEqual(c['SPN'], 11)
        assert_array_equal(c['SCTRNG'], array([ 0, 10, 19, 28, 35, 44, 53, 63], dtype='int16'))
        self.assertEqual(c['NSN64'], 4096)
        self.assertEqual(c['CWND'], 5.85938e-09)
        self.assertEqual(c['SCTSCLEM'], [0.33858267716535434, 0.3313953488372093, 0.3313953488372093])
        self.assertEqual(c['BTP'], 0)
        self.assertTrue( c['DCYCRR'])
        assert_array_equal(c['IMSIZE'], array([127, 344, 344]))

    def test_data(self):
        self.theTestObj.printd(self.theTestObj.mMRparams['Cnt'])
        self.theTestObj.printd(self.theTestObj.datain)

    def test_locs(self):
        self.assertTrue(os.path.exists(self.twiliteLoc+'-AC'))
        self.assertTrue(os.path.exists(self.tracerLoc+'-NAC'))



class TestTwilite(TestReconstruction):

    def _test_createTwiliteStaticNAC(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.twiliteLoc, v = True)
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def _test_createTwiliteStaticUTE(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.twiliteLoc, ac=True, v = True)
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def _test_createTwiliteStaticCarney(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.twiliteLoc, ac=True, v = True)
        sta = obj.createStaticCarney(fcomment='_createStaticCarney')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()



class TestNAC(TestReconstruction):

    def test_createTracerStaticNAC(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.tracerLoc, ac = False, v = True)
        sta = obj.createStaticNAC(fcomment='_createStaticNAC')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def test_createTracerNAC(self):
        import matplotlib.pyplot as plt
        mids = ['HYGLY30/V2/Twilite_V2' ]
        loc = '/home2/jjlee/Docker/NiftyPETd/'+mids[0]+'-Converted'
        obj = Reconstruction(loc, ac = False, v = True)
        dyn = obj.createDynamicNAC(fcomment='_createDynamicNAC')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()



class TestUTE(TestReconstruction):

    def test_createTracerStaticUTE(self):
        import matplotlib.pyplot as plt
        mids = ['HYGLY30/V2/Twilite_V2' ]
        loc = '/home2/jjlee/Docker/NiftyPETd/'+mids[0]+'-Converted'
        obj = Reconstruction(loc, ac=True, v=True)
        sta = obj.createStaticUTE(fcomment='_createStaticUTE')
        plt.matshow(sta['im'][60,:,:])
        plt.matshow(sta['im'][:,170,:])
        plt.matshow(sta['im'][:,:,170])
        plt.show()

    def test_createTracerUTE(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.tracerLoc, ac=True, v=True)
        dyn = obj.createDynamicUTE(fcomment='_createDynamicUTE')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()



class TestCarney(TestReconstruction):

    def test_createTracerStaticCarney(self):
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.tracerLoc, ac=True, v=True)
        dyn = obj.checkTimeAliasingUTE(fcomment='_checkTimeAliasingUTE')
        plt.matshow(dyn[0]['im'][60,:,:])
        plt.matshow(dyn[1]['im'][60,:,:])
        plt.show()

    def _test_checkTimeAliasingCarney(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.tracerLoc, ac=True)
        dyn = obj.checkTimeAliasingCarney(fcomment='_checkTimeAliasingCarney')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()

    def _test_checkTimeHierarchiesCarney(self):
        import matplotlib.pyplot as plt
        obj = Reconstruction(self.tracerLoc, ac=True)
        dyn = obj.checkTimeHierarchiesCarney(fcomment='_checkTimeHierarchiesCarney')
        plt.matshow(dyn['im'][60,:,:])
        plt.matshow(dyn['im'][:,170,:])
        plt.matshow(dyn['im'][:,:,170])
        plt.show()



class TestMatlab(TestReconstruction):

    def test_MagicSquarePkg(self):
        import MagicSquarePkg
        mcr = MagicSquarePkg.initialize()
        print(mcr.makesqr(3))
        mcr.terminate()


    def test_construct_resolved_mcr(self):
        loc = '/home2/jjlee/Local/Pawel/HYGLY47/V2/FDG_V2-Converted'
        obj = Reconstruction(loc, ac = False, v = True)
        obj.createDynamicNAC(fcomment='_createDynamicNAC')
        import construct_resolved_mcr
        mcr = construct_resolved_mcr.initialize()
        mcr.construct_resolved('sessionsExpr', 'HYGLY47', 'visitsExpr', 'V2', 'ac', False)
        mcr.terminate()
        obj._ac = True
        obj.createDynamic2Carney(fcomment='_createDynamic2Carney')
        mcr = construct_resolved_mcr.initialize()
        mcr.construct_resolved('sessionsExpr', 'HYGLY47', 'visitsExpr', 'V2', 'ac', True)
        mcr.terminate()



# N.B.:  duplicates unittest actions within pycharm
#suite = unittest.TestLoader().loadTestsFromTestCase(TestReconstruction)
#unittest.TextTestRunner(verbosity=2).run(suite)
