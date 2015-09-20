
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:53:51 2014
Phase correlation drift correction.
Used papers Cross-correlation image tracking for drift correction and 
adsorbate analysis B. A. Mantooth, Z. J. Donhauser, K. F. Kelly, and P. S. Weiss
for inspiration.
@author: Monika Kauer
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import os, sys
import scipy.misc as misc
import pump_writer

##===================================================#
#          image generator in/output
## ==================================================#
def read_sequentially(params, intrvl = 1):
    """Function that reads sequentially with every call."""
    filenames = os.listdir(params['directory'])
    filenames  = [f for f in filenames if ".%s"% params["type"] in f]
    filenames = pump_writer.natural_sort(filenames)
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
        yield read_img(params['directory']+filename, params)

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
##===================================================#
#          image registration
## ==================================================#
        
def reg(im1, im2, params):
    """Find image-image correlation and translation vector using FFTs."""
    # use hanning window. Reduces the edge effect from finite size
    shape= np.array(im1.shape)
   
    fft_im1 = np.fft.fft2(im1)
    fft_im2 = np.conj(np.fft.fft2(im2))
    
    corr = np.fft.ifft2(fft_im1*fft_im2).real
 
    corr = ndimage.gaussian_filter(corr, .5) - ndimage.gaussian_filter(corr, 30)
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
      
    try:
        while True: #go through all image chunks from start to end
            im_old = im_new
            im_new = im.next()
            im1 = np.where(im_old>np.median(im_old), 1,0)
            im2 = np.where(im_new>np.median(im_new), 1,0)
            
            _,drift = reg(im1, im2, params)    
           
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
        
        #r[cnt] = (index%1*vy)+y[i-1],(index%1*vx)+x[i-1]
        r[cnt] = y[i-1]+(index%1)*vy,x[i-1]+(index%1)*vx
        
        if cnt == len(r)-dr-1:
            #this deals with leftover interval if images%chunk!=0
            for rest in xrange(1,dr+1):
                index = float(rest)/dr+1
                vy, vx = y[i]-y[i-1], x[i]-x[i-1]
                r[cnt+rest] = y[i-1]+(index%1*vy),x[i-1]+(index%1*vx)
    return r

##===================================================#
# feature detection for neuron physiological imaging
## ==================================================#
    
def fluorescence(params, roi):
    """finds a neuron from images using thresholding
    in a region of interest."""
    images = read_sequentially(params)
    values,locations = [],[]
    try:
        cnt = 0
        imgs = ndimage.shift(images.next(), roi[cnt], mode="wrap")
        cms_old = [params['y0'],params['x0']]
        #print "cms_old is ",cms_old
        val_old = []
        y0, x0 = cms_old
        height, width = imgs.shape
        cnt += 1
        while True:
            y1, x1, fluor, bg = similarity3(imgs, cms_old,[y0,x0], params)
            #implement a short memory of neuron position
            val_old.append([y1, x1])
            y0 = np.average([v[0] for v in val_old[-10:]])
            x0 = np.average([v[1] for v in val_old[-10:]])
            values.append([fluor, bg])
            locations.append([y1-roi[cnt-1][0],x1+params["cropx"][0]-roi[cnt-1][1]])
            imgs = ndimage.shift(images.next(), roi[cnt], mode="wrap")
            cnt += 1
    except StopIteration:
        pass
    finally:
        del images
    return np.array(values), np.array(locations)

def similarity3(im1, cms,old_coor, params):
    """Calculates fluorescence of neuron by thresholding."""
    bgsize = params["bgsize"]
    part1 = im1[max(0,cms[0]-bgsize):cms[0]+bgsize, max(0,cms[1]-bgsize):cms[1]+bgsize]
    offsety, offsetx = max(0,cms[0]-bgsize), max(0,cms[1]-bgsize)    
    height, width = part1.shape
    y0,x0 = old_coor #previous coords
    thresh = np.sort(part1, axis=None)[-int(params["thresh_pump"]*height*width)]
    #print "threshold is", thresh
    mask = np.where(part1 > thresh, 1, 0)
    mask = ndimage.binary_opening(mask,structure = np.ones((2,2)))
    mask = ndimage.binary_closing(mask)
    label_im, nb_labels = ndimage.label(mask)
    centroids = ndimage.measurements.center_of_mass(part1, label_im, xrange(1,nb_labels+1))
    dist = []
    for index, coord in enumerate(centroids):
        y,x= coord
        dist.append((y-y0+offsety)**2 + (x-x0+offsetx)**2)
    if min(dist)>2*params["max_movement"]**2:
        print dist, y0,x0, offsety, offsetx, 
        y,x = y0-offsety,x0-offsetx
        radius = params["roisize"]
        neuron = part1[max(0,y-radius):y+radius,max(0,x-radius):x+radius,]
        value = np.ma.average(np.sort(neuron, axis=None)[-20:])
    else:
        loc = np.argmin(dist)
        y,x = centroids[loc]
        remove_pixel = np.where(label_im ==loc+1,0,1)
        neuron = np.ma.masked_array(part1, remove_pixel)
        value = np.ma.average(neuron)
    try:
       
        radius = params["roisize"]
        mask1 = np.zeros(part1.shape, dtype=bool)
        mask1[max(0,y-radius):y+radius,max(0,x-radius):x+radius,] = True
        bg_mask = np.ma.mask_or(mask,mask1)
        bg = np.ma.masked_array(part1, bg_mask)
        bg_level = np.ma.average(bg)
    except IndexError:
        y,x=y0,x0
        value=0
        bg_level=0
    return y+offsety, x+offsetx, value, bg_level

##===================================================#
#          Main
## ==================================================#
def warp_detector(params):
    
    ##===================================================#
    #           Translation correction
    ## ==================================================#
    drift = find_roi(params) 
    drift = interpol_drift(drift, params)
    print "done with drift"
    
    sys.stdout.flush()
    ##===================================================#
    #           detect pumping
    ## ==================================================#
    coords = fluorescence(params, drift)
    time = np.arange(params["start"], params["start"]+len(coords),1)
    
    out_data = zip(time,coords[:,0], coords[:,1],coords[:,2])
    print "Analysis of: ",params["start"], params["end"]

    ##===================================================#
    #          write results and movie
    ## ==================================================#
    if len(coords) > 0:
        outputstring = "%s_%i_%i"%(params["basename"],params["start"],params["end"]-1)
        pump_writer.write_data(params["outdir"], outputstring+"_kymo", out_data, 4)
       
        images = read_sequentially(params)
        fig = plt.figure(params["start"]+1) #make unique figures needed for parallelization
        
        ax1 = fig.add_subplot(211)
        pump_writer.make_kymograph(images, params, diff=False, roi=drift)
        ax1.plot(cms[:,0]+drift[:,0])        
        ax1.set_xlim([0,len(cms)])
        ax2 = fig.add_subplot(212)
        ax2.plot(coords[:,0])
        ax2.plot(coords[:,1])
        ax2.set_xlim([0,len(coords)])
	     ax2.set_ylim(ax2.get_ylim()[::-1])
        
        fig.savefig(params["outdir"]+"/"+outputstring+"_kym.png")
        
