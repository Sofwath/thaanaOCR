# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from thaanaocr import ThaanaOCR
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties

def reversed_string(a_string):
    new_string = ''
    index = len(a_string)
    while index:
        index -= 1                    
        new_string += a_string[index] 
    return new_string

font = {'family': 'Thaana Unicode Akeh',
                'color':  'darkred',
                'weight': 'normal',
                'size': 12,
        }


ocr = ThaanaOCR()
ocr.loadweights('weights24.h5') # replace with your trained weight file 

sample = Image.open('sample.png')

ans = ocr.ocr_frompic(image = sample)
im = sample
bas=reversed_string(ans) # for some odd reason [::-1] did not work on image, so a quick hack. 
plt.xlabel('Prediction = \'%s\'' % (bas[::-1]),fontdict=font)

plt.imshow(im, cmap='Greys_r')
plt.savefig("testresults.png")
print (bas[::-1])