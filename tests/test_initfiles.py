import unittest
import respet

class TestInitfiles(unittest.TestCase):

    cndaRawdataLocator = []
    testObj = []

    @classmethod
    def setUpClass(self):
        self.testObj = respet.recon.initfiles.Initfiles(self.cndaRawdataLocator)

    #@classmethod
    #def tearDownClass(self):

    def test_sampleStaticMethod(self):
        self.assertEqual(respet.recon.initfiles.Initfiles.sampleStaticMethod(), 0.1234)

suite = unittest.TestLoader().loadTestsFromTestCase(TestInitfiles)
unittest.TextTestRunner(verbosity=2).run(suite)
