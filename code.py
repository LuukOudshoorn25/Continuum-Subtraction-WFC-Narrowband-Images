from astropy.io import fits
import numpy as np
import pyregion
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import aplpy
import pyregion
from astropy.coordinates import SkyCoord
from sklearn import linear_model, datasets
import scipy
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from scipy.integrate import trapz
from scipy import signal
import pandas as pd


flist = glob('../CoaddedFrames/*regrid*fits')

def run_sex(f):
    fname = f.split('/')[-1]
    os.system('sex '+f+' -c ./sexfiles/params.sex -CHECKIMAGE_NAME ./segmentations/'+fname.replace('.fits', '.seg.fits')+' -CATALOG_NAME ./catalogues/'+fname.replace('.fits', '.cat'))

Parallel(n_jobs=12)(delayed(run_sex)(i) for i in flist)




