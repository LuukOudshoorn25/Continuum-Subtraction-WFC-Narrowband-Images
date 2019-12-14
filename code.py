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
from scipy.odr import *
from scipy import signal
import pandas as pd
import os
import drizzlepac
import multiprocessing as mp
from drizzlepac import tweakreg
from reproject import reproject_exact, reproject_interp
import shutil

###########################################################
#### Align images to sub_pixel accuracy                 ###
###########################################################
if not os.path.exists('./originals/'):
    os.mkdir('./originals/')
shutil.copy('narrowband.fits', './originals/narrowband.fits')
shutil.copy('broadband.fits', './originals/broadband.fits')
narrowband_hdu = fits.open('narrowband.fits')[0]
broadband_hdu  = fits.open('broadband.fits')[0]
broadband_median = np.nanmedian(broadband_hdu.data)
narrowband_median = np.nanmedian(narrowband_hdu.data)

tweakreg.TweakReg('broadband.fits',
       imagefindcfg={'threshold' : 3*broadband_median, 'conv_width' : 3.5},
       refimage='narrowband.fits',
       refimagefindcfg={'threshold' : 3*narrowband_median, 'conv_width' : 3.5},
       updatehdr=True, shiftfile=True, outshifts='shift.txt')
narrowband_hdu = fits.open('narrowband.fits')[0]
broadband_hdu  = fits.open('broadband.fits')[0]

reprojected_data,_ = reproject_interp(broadband_hdu, narrowband_hdu.header)
broadband_hdu_updated_wcs = narrowband_hdu.copy()
broadband_hdu_updated_wcs.data = reprojected_data
broadband_hdu_updated_wcs.writeto('broadband_regrid.fits', overwrite=True)

os.system('rm -rf broadband.fits *.coo *.log *.txt *.list *.match shift_wcs.fits')






###########################################################
#### Run sextractor on both files to get flux, FWHM etc ###
###########################################################

if not os.path.exists('./catalogues/'):
    os.mkdir('./catalogues/')
if not os.path.exists('./segmentations/'):
    os.mkdir('./segmentations/')

# Do detection always on broadband


def run_sex(input_tuple):
    det_file, f = input_tuple
    fname = f.split('/')[-1]
    os.system('sex '+det_file+','+f+' -c ./sexfiles/params.sex -CHECKIMAGE_NAME ./segmentations/'+fname.replace('.fits', '.seg.fits')+' -CATALOG_NAME ./catalogues/'+fname.replace('.fits', '.cat'))

inputs = [('broadband_regrid.fits','narrowband.fits'), ('broadband_regrid.fits','broadband_regrid.fits')]

Parallel(n_jobs=2)(delayed(run_sex)(i) for i in inputs)

###########################################################
#### Load SExtractor catalogues into memory             ###
###########################################################
broadband_cat  = Table.read('./catalogues/broadband_regrid.cat', format="fits", hdu='LDAC_OBJECTS').to_pandas()
narrowband_cat = Table.read('./catalogues/narrowband.cat', format="fits", hdu='LDAC_OBJECTS').to_pandas()

broadband_cat = broadband_cat[broadband_cat.MAG_AUTO<99]
narrowband_cat = narrowband_cat[narrowband_cat.MAG_AUTO<99]

broadband_cat = broadband_cat[broadband_cat.MAGERR_AUTO<1]
narrowband_cat = narrowband_cat[narrowband_cat.MAGERR_AUTO<1]

# Match the two catalogues
broadband_cat = broadband_cat.set_index(['X_IMAGE','Y_IMAGE'])
narrowband_cat = narrowband_cat.set_index(['X_IMAGE','Y_IMAGE'])

joined = broadband_cat.join(narrowband_cat, lsuffix='_broad', rsuffix='_narrow', how='inner')

broadband_cat = broadband_cat.reset_index().set_index('NUMBER').loc[joined.NUMBER_broad].reset_index()
narrowband_cat = narrowband_cat.reset_index().set_index('NUMBER').loc[joined.NUMBER_narrow].reset_index()

###########################################################
#### Get initial scaling number                         ###
###########################################################
def lin_func(p, x):
     return x+p

# Create a model for fitting.
lin_model = Model(lin_func)

# Create a RealData object using our initiated data from above.
data = RealData(broadband_cat.MAG_AUTO, narrowband_cat.MAG_AUTO, sx=broadband_cat.MAGERR_AUTO, sy=narrowband_cat.MAGERR_AUTO)

# Set up ODR with the model and data.
odr = ODR(data, lin_model, beta0=[5])

# Run the regression.
out = odr.run()

# Get coefficient
beta = out.beta[0]

minmag, maxmag = np.min(broadband_cat.MAG_AUTO), np.max(broadband_cat.MAG_AUTO)
plt.scatter(broadband_cat.MAG_AUTO, narrowband_cat.MAG_AUTO,s=1)
plt.errorbar(broadband_cat.MAG_AUTO, narrowband_cat.MAG_AUTO, xerr=broadband_cat.MAGERR_AUTO,yerr= narrowband_cat.MAGERR_AUTO, linestyle='none')
plt.xlabel('Broadband magnitude')
plt.ylabel('Narrowband magnitude')
plt.plot(np.linspace(minmag, maxmag,2), beta+np.linspace(minmag, maxmag,2), color='red')
plt.show()


multiplication_factor = 10**(beta/-2.5)

###########################################################
#### Export FITS without convolution optimization       ###
###########################################################
pure_line = fits.open('narrowband.fits')[0]
pure_line.data = narrowband_hdu.data - multiplication_factor * broadband_hdu.data
pure_line.writeto('pureline_noconv.fits',overwrite=True)





###########################################################
#### Residual function to count offset within apertures ###
###########################################################
def residual(pure_line_data,narrowband_cat, iter_):
    # Derive the segmentation map
    segmentation_map = fits.open('./segmentations/narrowband.seg.fits')[0].data
    # Clip such that where the count is unity, we have stellar stuff
    segmentation_map = np.clip(segmentation_map,0,1)
    # The subtracted value should be close to the background of the pure line map
    if not os.path.exists('./working_dir/'):
        os.mkdir('./working_dir/')
    pure_line = fits.open('narrowband.fits')[0]
    pure_line.data = pure_line_data
    #pure_line.writeto('./working_dir/pureline{}.fits'.format(iter_),overwrite=True)
    #os.system('sex broadband_regrid.fits,./working_dir/pureline{}.fits -c ./sexfiles/params.sex -CATALOG_NAME ./working_dir/sex{}.cat'.format(iter_,iter_))

    cat = Table.read('./working_dir/sex{}.cat'.format(iter_), format="fits", hdu='LDAC_OBJECTS').to_pandas()

    cat = cat.set_index(['X_IMAGE','Y_IMAGE'])
    narrowband_cat = narrowband_cat.set_index(['X_IMAGE','Y_IMAGE'])

    joined = cat.join(narrowband_cat, lsuffix='_broad', rsuffix='_narrow', how='inner')
    cat = cat.reset_index().set_index('NUMBER').loc[joined.NUMBER_narrow].reset_index()

    penalty1 = ((cat.FLUX_AUTO - cat.ISOAREA_IMAGE*cat.BACKGROUND).abs().sum()
                 + (segmentation_map*pure_line_data)**2/(segmentation_map*pure_line_data))
    

    penalty2 = np.nanmedian((pure_line_data * segmentation_map) - 50*np.nanmedian(pure_line_data*np.where(segmentation_map==0,1,np.nan))**2)
    return penalty1,penalty2
    

alphas=np.round(np.linspace(0.8*multiplication_factor, 1.0*multiplication_factor,12),4)

def worker(iter_,alpha):
    data = narrowband_hdu.data - alpha * broadband_hdu.data
    penalty1,penalty2 = residual(data,narrowband_cat, iter_)
    return (penalty1,penalty2)


inputs = list(zip(np.arange(0,len(alphas)), alphas))
pool = mp.Pool(processes=8)
results = [pool.apply(worker, args=(x)) for x in inputs]
penalties1 = [w[0] for w in results]
penalties2 = [w[1] for w in results]

plt.figure(figsize=(8,5))
plt.subplot(121)
plt.plot(alphas,penalties2)
plt.subplot(122)
plt.plot(alphas,penalties1)
plt.show()


best_scaling = 0.5*(alphas[np.argmin(penalties1)]+alphas[np.argmin(penalties2)])

###########################################################
#### Export FITS without convolution optimization       ###
###########################################################
pure_line = fits.open('narrowband.fits')[0]
pure_line.data = narrowband_hdu.data - best_scaling * broadband_hdu.data
pure_line.writeto('pureline_noconv.fits',overwrite=True)





###########################################################
#### Get FWHM of stars in narrowband and broadband      ###
###########################################################
fwhm_broad  = broadband_cat.FWHM_IMAGE
fwhm_narrow = narrowband_cat.FWHM_IMAGE
fwhm_narrow.hist(bins=np.arange(3,10,0.1), histtype='step', linewidth=2, normed=True)
fwhm_broad.hist(bins=np.arange(3,10,0.1), histtype='step', linewidth=2, normed=True)














from glob import glob
flist = glob('../CalibratedFrames/*.fits')

seeings = [imhead(w,hdkey='SEEING', mode='get') for w in flist]
seeings_dict = {}
for i in range(len(flist)):
    seeings_dict[flist[i]] = seeings[i]


# We need to regrid everything to a new resolution
new_res = 7.6

for f in flist[-1:]:
    print(f)
    outputfits = f.replace('CalibratedFrames', 'SmoothenedFrames')
    outputfits = outputfits.replace('_calib.fits', '_calib_smooth.fits')
    inputimage = f.replace('.fits', '.image')
    inputimage = inputimage.replace('CalibratedFrames', 'SmoothenedFrames')
    outputimage = inputimage.replace('_calib.image', '_calib_smooth.image')
    os.system('rm -rf '+outputimage+' '+ outputfits)
    importfits(f, inputimage)
    beamsize = np.sqrt(new_res**2 - seeings_dict[f]**2)
    beamsize = str(beamsize)+'pix'
    print(beamsize)
    imsmooth(inputimage, outfile=outputimage, major=beamsize, minor=beamsize, pa='0deg')
    exportfits(outputimage, outputfits, overwrite=True)
    os.system('rm -rf ../SmoothenedFrames/*.image ../SmoothenedFrames/*.log')

#for f in flist:
#    beamsize = np.sqrt(new_res**2 - seeings_dict[f]**2)
#    print(seeings_dict[f], beamsize)




broadbandf  = '../CalibratedFrames/R_theli_all_regrid_calib.fits'
narrowbandf = '../CalibratedFrames/Ha_all_theli_regrid_calib.fits'

broadband=fits.open(broadbandf)[0]
narrowband=fits.open(narrowbandf)[0]


pure_line = narrowband.data - 0.064*broadband.data
hdu_out = narrowband.copy()
hdu_out.data = pure_line
hdu_out.writeto('../ContsubFrames/Ha_regrid_calib_smooth_sub.fits', overwrite=True)

