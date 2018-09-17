from unittest import TestCase
from .context import respet

def fun(x):
    return 'arbitrary fun for ' + x

'''
class TestNiftyPETy(TestCase):
    """exploratory unittest"""
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2017"

    def test_fun(self):
        self.assertEqual(fun('me'), 'arbitrary fun for me')

    def test_ctor(self):
        self.assertIsInstance(self.testObj, 'Reconstruction')

    def test_reconAllTimes(self):
        self.testObj.reconAllTimes

    def test_reconTimeInterval(self):
        self.testObj.reconTimeInterval(3000, 3600, 65)

    def test_custom_mumap(self):
        self.testObj.custom_mumap

    def test_mMR_mumap(self):
        self.testObj.mMR_mumap

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

suite = TestLoader().loadTestsFromTestCase(TestNiftyPETy)
TextTestRunner(verbosity=2).run(suite)
'''

