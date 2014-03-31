from os import environ
import os.path as op
import numpy as np
from surfer import Brain, io

import os
from os.path import join as pjoin
import numpy.random as random

def hsv_to_rgb(h, s, v):
    """Converts HSV value to RGB values
    Hue is in range 0-359 (degrees), value/saturation are in range 0-1 (float)

    Direct implementation of:
    http://en.wikipedia.org/wiki/HSL_and_HSV#Conversion_from_HSV_to_RGB
    """
    h, s, v = [float(x) for x in (h, s, v)]

    hi = (h / 60) % 6
    hi = int(round(hi))

    f = (h / 60) - (h / 60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if hi == 0:
        return v, t, p
    elif hi == 1:
        return q, v, p
    elif hi == 2:
        return p, v, t
    elif hi == 3:
        return p, q, v
    elif hi == 4:
        return t, p, v
    elif hi == 5:
        return v, p, q

from random import randint, uniform

def write_ctab(labellist, out_file="myctab.ctab"):
    f = open(out_file, "w")
    for idx, label in enumerate(labellist):
        h = randint(0, 300) # Select random green'ish hue from hue wheel
        s = uniform(0.8, 1)
        v = uniform(0.8, 1)

        r, g, b = hsv_to_rgb(h, s, v)
        print(r,g,b)

        r = randint(0,255)
        g = randint(0,255)
        b = randint(0,255)

        label = op.basename(label)
        label = label.replace('.label','')
        #f.write("%d %s %d %d %d %d \n" % (idx, label, r*256, g*256, b*256, 0))
        f.write("%d %s %d %d %d %d \n" % (idx, label, r, g, b, 0))
    
    f.close()
    return out_file