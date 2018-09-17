from unittest import TestCase
from .context import respet


def sampleFun(x):
    return x + 1

class TestReconstruction(TestCase):
    def test_sampleStaticMethod(self):
        self.fail()
    def test_sampleFun(self):
        self.assertEqual(sampleFun(3), 4)
