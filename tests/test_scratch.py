'''
class TestScratch(TestCase):

    def test_reconAllTimes(self):
        self.testObj.reconAllTimes

    def test_reconTimeInterval(self):
        self.testObj.reconTimeInterval(3000, 3600, 65)

    def test_custom_mumap(self):
        self.testObj.custom_mumap

    def test_mMR_mumap(self):
        self.testObj.mMR_mumap

suite = TestLoader().loadTestsFromTestCase(TestNiftyPETy)
TextTestRunner(verbosity=2).run(suite)
'''

import unittest
import respet

def globalfun(x):
        return 'arbitrary global fun for ' + x

class TestScratch(unittest.TestCase):

    testvar = []

    @staticmethod
    def fun(x):
        return 'arbitrary fun for ' + x

    def test_fun(self):
        self.assertEqual(TestScratch.fun('me'), 'arbitrary fun for me')

    def test_globalfun(self):
        self.assertEqual(globalfun('me'), 'arbitrary global fun for me')

    def test_sampleStaticMethod(self):
        self.assertEqual(respet.recon.scratch.Scratch.sampleStaticMethod(), 0.1234)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_testvar(self):
        self.assertEqual(self.testvar, [])

if __name__ == '__main__':
    unittest.main()
