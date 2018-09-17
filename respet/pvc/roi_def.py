# defining ROIs using GIF labels
rois = {
    'wm'   :range(81,95),               #white matter
    'cmpst':[35,36,39,40,41,42,72,73,74],# composite: brain stem with cerebellum (whole)
    'bstem':[36],                       #brain stem
    'pons': [35],                       #pons
    'cblwh':[39,40,41,42,72,73,74],     #cerebellum whole
    'cblgm':[39,40,72,73,74],           #cerebellum GM
    #'cblwm':[41,42],                    #cerebellum WM
    'hppcm':[48,49],                    #hippocampus 
    #'phpcm':[171,172],                  #parahippocampal gyrus
    'pcglt':[167,168],                  #posterior cingulate gyrus
    'mcglt':[139,140],                  #middle cingulate gyrus
    'acglt':[101,102],                  #anterior 
    'precu':[169,170],                  #precuneus
    'partl':[175,176,199,200],          #parietal without precuneus
    'tmprl':[133,134,155,156,201,202,203,204], #temporal
    'neocx':[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 113, 114, 115, 116, 117, 118,
            119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 133, 134, 135, 136, 137, 138,
            139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
            155, 156, 157, 158, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
            173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
            191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
            207, 208] #neocortex
}

# pick the reference regions for reporting
refi = ['cmpst','cblwh','cblgm']
# pick the neocortex regions for reporting (boxplot)
roii = ['hppcm', 'acglt', 'mcglt', 'pcglt', 'precu', 'tmprl', 'partl', 'neocx']


# create segmentation/parcellation for PVC, with unique regions numbered from 0 onwards
pvcroi = [] 
pvcroi.append([66,67]+range(81,95)) #white matter
pvcroi.append([36])                 #brain stem
pvcroi.append([35])                 #pons
pvcroi.append([39,40,72,73,74])     #cerebellum GM
pvcroi.append([41,42])              #cerebellum WM
pvcroi.append([48,49])              #hippocampus
pvcroi.append([167,168])            #posterior cingulate gyrus
pvcroi.append([139,140])            #middle cingulate gyrus
pvcroi.append([101,102])            #anterior cingulate gyrus
pvcroi.append([169,170])            #precuneus
pvcroi.append([32,33])              #amygdala
pvcroi.append([37,38])              #caudate
pvcroi.append([56,57])              #pallidum
pvcroi.append([58,59])              #putamen
pvcroi.append([60,61])              #thalamus
pvcroi.append([175,176,199,200])    #parietal without precuneus
pvcroi.append([133,134,155,156,201,202,203,204])            #temporal
pvcroi.append([4, 5, 12, 16, 43, 44, 47, 50, 51, 52, 53])   #CSF
pvcroi.append([24, 31, 62, 63, 70, 76, 77, 96, 97])                    #basal ganglia + optic chiasm
pvcroi.append([103, 104, 105, 106, 107, 108, 109, 110, 113, 114, 115, 116, 117, 118,
               119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 135, 136, 137, 138,
               141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
               157, 158, 161, 162, 163, 164, 165, 166,  171, 172,
               173, 174, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
               191, 192, 193, 194, 195, 196, 197, 198, 205, 206, 207, 208] ) #remaining neocortex
