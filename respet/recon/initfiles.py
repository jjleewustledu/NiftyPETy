import os
import respet.recon


class Initfiles(object):
    __author__ = "John J. Lee"
    __copyright__ = "Copyright 2018"

    @property
    def cndaRawdataLocator(self):
        return self._cndaRawdataLocator

    @cndaRawdataLocator.setter
    def cndaRawdataLocator(self, s):
        self._cndaRawdataLocator = s

    @staticmethod
    def sampleStaticMethod():
        return 0.1234

    def initRawdata(self):
        pass

    def initHardwareUmap(self):
        # cp -r /data/nil-bluearc/raichle/jjlee/Local/JSRecon12/hardwareumaps mumap_hdw
        pass

    def initUmapSynth(self):
        # fslroi umapSynth  umapSynthFull -86 344 -86 344 0 -1
        # fslmaths umapSynthFull -s 1.8259 umapSynthFull_b43
        pass

    def __init__(self, cndaRawdataL=None, workLoc='/home2/jjlee/Local/Pawel/WorkLoc'):
        """
        :param:  loc specifies the origination location of tracer rawdata.
        :param:  self.localRawdataLocation contains Siemens sinograms, e.g.:
                  -rwxr-xr-x+  1 jjlee wheel   16814660 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.2016090913012239062507614.bf
                  -rwxr-xr-x+  1 jjlee wheel     141444 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.2016090913012239062507614.dcm
                  -rwxr-xr-x+  1 jjlee wheel  247141283 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000050.bf
                  -rwxr-xr-x+  1 jjlee wheel     151868 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000050.dcm
                  -rwxr-xr-x+  1 jjlee wheel 323404 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000048.bf
                  -rwxr-xr-x+  1 jjlee wheel 143938 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000048.dcm
                  -rwxr-xr-x+  1 jjlee wheel 6817490860 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.bf
                  -rwxr-xr-x+  1 jjlee wheel     145290 Sep 13  2016 1.3.12.2.1107.5.2.38.51010.30000016090616552364000000049.dcm
                  -rw-r--r--+  1 jjlee wheel    3081280 Nov 14 14:53 umapSynth_full_frame0.nii.gz
        """
        if cndaRawdataL:
            self.cndaRawdataLocator = cndaRawdataL
        self._workLocation = workLoc
