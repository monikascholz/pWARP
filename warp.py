
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:53:51 2014
Phase correlation drift correction.
Used papers Cross-correlation image tracking for drift correction and 
adsorbate analysis B. A. Mantooth, Z. J. Donhauser, K. F. Kelly, and P. S. Weiss
for inspiration.
For pump detection: entropy based feature detection.
@author: Monika Kauer
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import os, sys, re
import scipy.misc as misc

##===================================================#
#          I/O
## ==================================================#
def natural_sort(liste):
    """Natural sort to have frames in right order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(liste, key = alphanum_key)
    
def read_sequentially(params, intrvl = 1):
    """Function that reads sequentially with every call."""
    filenames = os.listdir(params['directory'])
    filenames  = [f for f in filenames if ".%s"% params["type"] in f]
    filenames = natural_sort(filenames)
    filenames2 = filenames[params["start"]:params["end"]:intrvl]
    #deal with non-divisor chunk lengths
    if (params["end"]-params["start"])%intrvl != 0:
        try:
            filenames2.append(filenames[params["end"]])
            params['nof'] = (params["end"]-params["start"])
        except IndexError:
            filenames2.append(filenames[-1])
            params['nof'] = (len(filenames)-params["start"])
    elif intrvl > 1 and (params["end"]-params["start"])%intrvl == 0:
        try:
            filenames2.append(filenames[params["end"]])
            params['nof'] = (params["end"]-params["start"])
        except IndexError:
            filenames2.append(filenames[-1])
            params['nof'] = (len(filenames)-params["start"])
    for filename in filenames2:
        yield read_img(os.path.join(params['directory'],filename), params)
  
  
def read_img(fname, params):
    """reads an image file as array."""
    try:
        image = misc.imread(fname, flatten=True)
        data  = np.asarray(image, dtype = np.int64)
        if params['rotate']:
            data = np.transpose(data)
        if params['cropx']:
            data = data[:,params['cropx'][0]:params['cropx'][1]]
        return data
    except IOError:
        print fname
        pass

def write_data(outdir, fname_out, data, ncol=1, header=None):
    """writes pumping traces to file.""" 
    s="%f "*ncol+"\n"
    with open(os.path.join(outdir,fname_out), 'w') as f:
        if header!=None:
            f.write(header)
        [f.write(s%tuple(x)) for x in data]
        
##===================================================#
#          image registration
## ==================================================#
        
def reg(im1, im2, hann):
    """Find image-image correlation and translation vector using FFTs.
    remove edge effects using a hanning window."""

    shape= np.array(im1.shape)
    
    fft_im1 = np.fft.fft2(im1*hann)
    fft_im2 = np.conj(np.fft.fft2(im2*hann))
    
    corr = np.fft.ifft2(fft_im1*fft_im2).real
 
    corr = ndimage.gaussian_filter(corr, .5) - ndimage.gaussian_filter(corr, 50)
    t0, t1 = np.unravel_index(np.argmax(corr), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return corr, [t0, 0]


def find_roi(params):
    """calculates image drift using registration via correlation."""
    im = read_sequentially(params, intrvl = params["chunk"])
    roi = [[0,0]]
    im_new = im.next()
    
    height, width = im_new.shape
    hann = np.sqrt(np.outer(np.hanning(height), np.hanning(width)))
    try:
        while True: #go through all image chunks from start to end
            im_old = im_new
            im_new = im.next()
            im1 = np.where(im_old>np.median(im_old), 1,0)
            im2 = np.where(im_new>np.median(im_new), 1,0)
            
            _,drift = reg(im1, im2, hann)    
            #drift = [registration(im1, im2),0] 
            roi.append(drift)
    except StopIteration:
        pass
    finally:
        del im
    return np.array(roi)

def interpol_drift(drift, params):
    """Returns linearly interpolated ROI.
     This uses drift calculation where drift comes from adjacent reference frame."""
    x = np.cumsum(drift[:,1])
    y = np.cumsum(drift[:,0])

    r = np.zeros((params['nof'],2))
    dr = params['nof']%params['chunk']
    
    for cnt in xrange(1,len(r)-dr):
        index = float(cnt)/(params["chunk"])
        i = int(index)+1
        vy, vx = y[i]-y[i-1], x[i]-x[i-1]
        r[cnt] = y[i-1]+(index%1)*vy,x[i-1]+(index%1)*vx
        
        if cnt == len(r)-dr-1:
            #this deals with leftover interval if images%chunk!=0
            for rest in xrange(1,dr+1):
                index = float(rest)/dr+1
                vy, vx = y[i]-y[i-1], x[i]-x[i-1]
                r[cnt+rest] = y[i-1]+(index%1*vy),x[i-1]+(index%1*vx)
    return r

##===================================================#
#          feature detection
## ==================================================#
def calc_entr(im):
    '''calculate image entropy using fast histogram method.'''
    bins = np.linspace(0.2,1,64)
    #bins = np.linspace()
    #print np.max(im)
    #bins = 64
    hist, _ = np.histogram(im, bins)
    # shift to avoid log 0
    hist = hist + 1
    area = 1.0*np.sum(hist)
    
    entr = hist/area
    cut = 10
    entr2 = hist[cut:]/1.0/np.sum(hist[cut:])
    return -entr*np.log(entr)*hist, -entr2*np.log(entr2)*hist[cut:]

def difference_entropy(im1,im2, cms, params):
    """Calculates the entropy of the difference image in the region of intrest."""
    size = params["roisize"]
    sizex = int(im1.shape[1]/3.)#params["roisize"]#
    #calculate difference image
    diff = (im2-im1)
    #crop to region of interest
    diff_small = np.abs(diff[max(0,cms[0]-size):cms[0]+size, max(0,cms[1]-sizex):cms[1]+sizex])
    #diff[max(0,cms[0]-size):cms[0]+size, max(0,cms[1]-sizex):cms[1]+sizex] = 0 
    # entropy of small snippet    
    entr2, entr1 = calc_entr(diff_small)
    #rest of image entropy    
    #entr1, area1 = calc_entr(np.abs(diff))

    return np.sum(entr2), np.sum(entr1)

def pumping(params, roi):
    """finds periodic motion from images using entropy based difference detection
    in a region of interest. Also calculates a kymograph."""
    images = read_sequentially(params)
    sim= []
    kymo = []
    try:
        img0 = ndimage.shift(images.next(), roi[0], mode="wrap")
        img1 = ndimage.shift(images.next(), roi[1], mode="wrap")
        cnt = 2
        sim = []
        cms_old = [params['y0'],params['x0']]
        #normalize images
        min1, max1 = 1.0*np.min(img0), 1.0*np.max(img0)
        img0 =(img0-min1)/(max1-min1)
        min1, max1 = 1.0*np.min(img1), 1.0*np.max(img1)
        img1 =(img1-min1)/(max1-min1)
        # histogram-equalize images
        #img0 = 1.0*(img0 - np.mean(img0)) / np.std(img0)
        #img1 = 1.0*(img1- np.mean(img1)) / np.std(img1)
        #img0[img0>0] = 0
        #img1[img1>0] = 0
        while True:
            entr_small, area_small = difference_entropy(img0,img1, cms_old,params)
            sim.append([entr_small, area_small,cms_old[0]])
            img0 = img1
            img1 = ndimage.shift(images.next(), roi[cnt], mode="wrap")
            # normalize images
            min1, max1 = 1.0*np.min(img1), 1.0*np.max(img1)
            img1 =(img1-min1)/(max1-min1)
            # histogram-equalize images
            #img1 = 1.0*(img1- np.mean(img1)) / np.std(img1)
            #img1[img1>0] = 0
            cnt += 1
            kymo.append(np.sum(img0, axis=1))
    except StopIteration:
        pass
    finally:
        del images
    return np.array(sim), kymo

##===================================================#
#          Main
## ==================================================#
def warp_detector(params):
    """ main script to run entropy-based change detection on a stack of images.
    If run directly, needs a dictionary like this:
    If run directly, needs a dictionary like this (all entries required):
    params = {'cropx': [0, -1], 
    'NPROCS': 16, 
    'end': 1801, 
    'chunk': 60, 
    'roisize': 120, 
    'start': 0, 
    'rotate': False, 
    'directory': '../images/df0432/',
    'y0': 469.84462300000001,
    'x0': 69.5, 'roi_file': 
    '../results/roi_df0432', 
    'type': 'png', 
    'basename': 'df0432',
    'outdir': '../results/'}
    """
    ##===================================================#
    #           Translation correction
    ## ==================================================#
    drift = find_roi(params) 
    drift = interpol_drift(drift, params)
    print "done with drift"
    #write to stdout
    sys.stdout.flush()
    ##===================================================#
    #           detect pumping
    ## ==================================================#
    coords, kymo = pumping(params, drift)
    if len(coords) > 0:
        time = np.arange(params["start"], params["start"]+len(coords),1)
       
        out_data = zip(time,coords[:,0],coords[:,1], coords[:,2])
        print "Analysis of: ",params["start"], params["end"]
        sys.stdout.flush()
        ##===================================================#
        #          write results and kymograph
        ## ==================================================#
        outputstring = "%s_%i_%i"%(params["basename"],params["start"],params["end"]-1)
        write_data(params["outdir"], outputstring+"_kymo", out_data, 4)
        
        #make unique figures needed for parallelization
        fig = plt.figure(params["start"]+1) 
        ax1 = fig.add_subplot(311)
        kymo=np.array(kymo)
        plt.imshow(np.transpose(kymo), aspect='auto', cmap= "gray", origin='lower')
        plt.plot(0, params['y0'], 'ro')
        ax1.set_xlim([0,len(kymo)])
        ax2 = fig.add_subplot(312)
        ax2.plot(coords[:,0])
        ax2.set_xlim([0,len(coords)])
        plt.ylabel('image entropy')
        plt.xlabel('time')
        ax3 = fig.add_subplot(313)
        ax3.plot(coords[:,1])
        ax3.set_xlim([0,len(coords)])
        plt.ylabel('image entropy')
        plt.xlabel('time')
        fig.savefig(params["outdir"]+"/"+outputstring+"_kym.png")
        
