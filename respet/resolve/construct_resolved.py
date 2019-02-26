import numpy as np
import sys, os
import respet.recon
import respet.resolve

class ConstructResolved(object):
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2019"

    mcr = None # Matlab compiler runtime for PyConstructResolved

    def construct_resolved(self):
        """
        mlraichle.SessionData
        :param mlpipeline.StudyData:  Matlab class instance
        :param str sessionPath:
        :param integer vnumber:
        :param integer snumber:
        :param str tracer:
        :param bool ac:  attenuation correction
        :param numpy.ndarray tauIndices:
        :param float fractionalImageFrameThresh:
        :param str frameAlignMethod:  see mlfourdfp.FourdfpVisitor
        :param str compAlignMethod:  see mlfourdfp.FourdfpVisitor

        mlpet.TracerBuilder
        :param mlrois.IRoisBuilder roisBuilder:  Matlab class instance
        :param mlfourdfp.T4ResolveBuilder resolveBuilder:  Matlab class instance
        :param mlfourdfp.CompositeT4ResolveBuilder compositeResolveBuilder:  Matlab class instance
        :param mlsiemens.MMRBuilder vendorSupport:  Matlab class instance
        :param bool ac

        mlfourdfp.AbstractSessionBuilder
        :param mlpipeline.IStudyCensus census:
        :param mlpipeline.ISessionData sessionData:

        mlpipeline.AbstractBuilder
        :param mlfourdfp.FourdfpVisitor buildVisitor:  Matlab class instance
        :param str logPath:
        :param mlpipeline.ILogger logger:
        :param object product:
        """
        obj = self.mcr.construct_resolved()
        return obj

    def __init__(self):
        return

    def __enter__(self):
        """
        https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        :return:
        """
        import PyConstructResolved
        self.mcr = PyConstructResolved.initialize()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        """
        https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        :param exc_type:
        :param exc_value:
        :param traceback:
        :return:
        """
        self.mcr.terminate()
        return
