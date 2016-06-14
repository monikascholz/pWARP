# -*- coding: utf-8 -*-
"""
This module allows manual to interactively set up pumping analysis. The user input is transformed into
a batch submission script ready for cluster submission.
"""

import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc as misc
from scipy import ndimage
import os, argparse, re
from skimage.feature import match_template
from warp import natural_sort, read_img, write_data

##===================================================#
#          automatic template finder
## ==================================================#
def define_template(filenames, p):
    """defines a template region in an image stack.

    Template is an average image from the region around coords[0] +/- height 
    and coords[1] +/- width. Returns the average of that region of interest
    over all input files given by the filenames list.

    filenames: images readable by scipy.misc.imread
    """
    bulbs = []
    y,x = p.BULB
    height = p.SIZE
    #width =  int(0.8*height) 
    for fname in filenames:
        image = misc.imread(os.path.join(p.DIRC,fname))
        if p.ROT:
            image = np.transpose(image)
        if p.CROP !=None:
            image = image[:,p.CROP[0]:p.CROP[1]]
        width = int(image.shape[1]/3.)
        data1  = np.asarray(image, dtype = np.int64)
        bulbs.append(data1[max(y - height, 0):y + height, max(x - width, 0):x + width])
    bulbs = np.array(bulbs)
    bulbs = np.average(bulbs, axis = 0)
    bulbs = ndimage.gaussian_filter(bulbs,2)
    return bulbs

def find_bulb(image, templ):
    """finds the terminal bulb in an image using template correlation.

    Finds the best location (shifted cross-correlation) between image and template 
    return: location (x,y) and correlation value at the maximal correlation.

    """
    image = ndimage.gaussian_filter(image, 2) #- ndimage.gaussian_filter(res, 50)
    cut = int(0.1*image.shape[1])
    
    result = match_template(image, templ)
    xm = int(result.shape[1]/2.)
    res = result[:,max(0,-cut + xm):xm+cut]
    
    
    ij = np.unravel_index(np.argmax(res), res.shape)
    x0, y0 = ij[::-1]
    # calculate half template size
    t_half = int(templ.shape[0]/2.)
    conf = res[y0,x0]
    
    result1 = match_template(image, templ[t_half:,])
    res1 = result1[:,max(0,-cut + xm):xm+cut]
    ij = np.unravel_index(np.argmax(res1), res1.shape)
    
    x1, y1 = ij[::-1]
    conf1 = res1[y1,x1]
    if conf1 > conf:
        conf = conf1
        x0,y0 = x1,y1
        res = res1
        t_half = int(templ.shape[0]/4.)
            
    result2 = match_template(image, templ[:t_half,])
    res2 = result2[:,max(0,-cut + xm):xm+cut]

    ij = np.unravel_index(np.argmax(res2), res2.shape)
    x2, y2 = ij[::-1]
    conf2 = res2[y2,x2]
    if conf2 > conf:
        conf = conf2
        x0,y0 = x2,y2
        res = res2
        t_half = int(templ.shape[0])/4.
            
    x = max(0, min(x0+templ.shape[1]/2.+cut, image.shape[1]-1))
    y = max(0,min(y0+t_half, image.shape[0]-1))
    if conf < 0.4 or conf/np.std(res) < 2.5:
        conf = 0.0 
    return y,x, conf

def clean_auto_coords(p, time, coords, confs, spacing, n = 3):
    """Consolidates multiple template matches for a 
    best guess of the bulb location.

    Uses the information of multiple template matches in a movie to
    find and remove outliers (large deviations in location of the region of interest.)
    """
    ys = coords[:,0]
    ratio =  len(confs[confs==0])/1.0/len(confs)
    fig = plt.figure(figsize=(12,8), dpi=100,edgecolor='k',facecolor='w')
    plt.subplot(211)
    plt.title('Misstracked: %.2f'%ratio)
    plt.plot(np.diff(ys), confs[:-1],'o', lw = 2)
    
    # make an outlier cutoff
    indizes = []
    for i in range(0,len(ys),n):
        data = [ys[i+k] for k in range(n) if (k+i) < len(ys)]
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else [0]
        for k in range(len(s)):
            if s[k] < 3:
                indizes.append(i+k)
    
    y_new = ys[indizes]
    time2 = time[indizes]
    confs = confs[indizes]
    spacing = spacing[indizes]
    #calculate a best-guess scenario with weighted average
    final_y = []
    final_time = []
    final_spacing = []
    for i in range(0,len(y_new),n):
        avg = 0
        avg = np.sum([y_new[i+k]*confs[i+k] for k in range(n) if (k+i) < len(y_new)])  
        if avg > 0:
            conf_norm = np.sum([confs[i+k] for k in range(n) if (k+i) < len(y_new)])
            final_y.append(1.0*avg/conf_norm)
            final_time.append(time2[i])
            final_spacing.append(spacing[i]*n)
            
    plt.subplot(212)
    plt.plot(final_time,final_y,'o-', lw=2)
    plt.show(block = True)
    return final_time, final_y, final_spacing
        

##===================================================#
#          I/0
## ==================================================#


def boolean(string):
    """Convert commandline input to boolean values."""
    string = string.lower()
    if string in ['0', 'f', 'false', 'no', 'off']:
      return False
    elif string in ['1', 't', 'true', 'yes', 'on']:
      return True
    else:
      raise ValueError()    
          
def write_slurm_file(p):
    """makes submission file for sbatch.

    This is specific to the cluster and job handling tool used.
    It also prints out the line calling the image analysis script.
    """
    with open(os.path.join(p.SCRIPTDIR,p.BASENAME+".slurm"), 'w') as f:
        if p.ACCOUNT in ["d","dinner","pi-dinner"]:
            f.write("""#!/bin/sh \n#SBATCH --account=pi-dinner\n#SBATCH --job-name=%s\n#SBATCH --output=%s\n#SBATCH --exclusive\n#SBATCH --time=0:30:0\n\necho "start time: `date`"\n """%(p.BASENAME,p.BASENAME+'.out'))
        
        elif p.ACCOUNT in ["b", "biron", "pi-dbiron"]:
            f.write("""#!/bin/sh \n#SBATCH --account=pi-dbiron\n#SBATCH --job-name=%s\n#SBATCH --output=%s\n#SBATCH --exclusive\n#SBATCH --time=0:30:0\n\necho "start time: `date`"\n """%(p.BASENAME,p.BASENAME   +'.out'))
        
        elif p.ACCOUNT in ["weare-dinner", "wd", "weare"]:
            f.write("""#!/bin/sh \n#SBATCH --account=weare-dinner\n#SBATCH --job-name=%s\n#SBATCH --output=%s\n\
#SBATCH --exclusive\n#SBATCH --time=0:30:0\n#SBATCH --partition=weare-dinner\n#SBATCH --qos=weare-dinner\n
 \necho "start time: `date`"\n """%(p.BASENAME,p.BASENAME+'.out'))
        f.write('python WARP_parallel.py -nprocs %i -type %s -basename %s -directory "%s" -roi_file "%s" \
    -outdir "%s" -cropx %i %i -rotate %s -chunk %s -roisize %s -entropybins %s %s %s \n'%(p.NPROCS, p.TYP, p.BASENAME, p.DIRC,\
                os.path.join(p.OUTDIR, "roi_"+p.BASENAME), p.OUTDIR,p.CROP[0], p.CROP[1], p.ROT,p.CHUNK, p.ROISIZE, p.BINS[0], p.BINS[1], p.BINS[2]))
        
        f.write("""echo "end   time: `date`" """)
        f.write("""echo "end   time: `date`"\n""")
        f.write("""echo "time entropy area cms " > {0}.dat\n""".format(os.path.join(p.OUTDIR, p.BASENAME)))
        f.write("""cd {0}\n""".format(p.OUTDIR))
        f.write("""ls -v . | grep {0}_ | grep 'kymo$'| xargs -n 1 -I % cat % >> {1}.dat\n""".format(p.BASENAME, p.BASENAME))
        f.write("""ls . | grep {0}_ | grep 'kymo$'| xargs -n 1 -I % rm %""".format( p.BASENAME))
    print 'python WARP_parallel.py -nprocs %i -type %s -basename %s -directory "%s" -roi_file "%s" \
    -outdir "%s" -cropx %i %i -rotate %s -chunk %s -roisize %s -entropybins %s %s %s \n'%(p.NPROCS, p.TYP, p.BASENAME, p.DIRC,\
                os.path.join(p.OUTDIR, "roi_"+p.BASENAME), p.OUTDIR,p.CROP[0], p.CROP[1], p.ROT,p.CHUNK, p.ROISIZE, p.BINS[0], p.BINS[1], p.BINS[2])
        

##===================================================#
#          interactive class
## ==================================================#
class clickSaver:
    """Event handler for matplotlib GUI."""
    event = None
    xroi = []
    yroi = []
    key = 0
    
    def onclick(self,event):    
        self.event=event
        self.xroi.append(event.xdata)
        self.yroi.append(event.ydata)
    
    def binary_check(self, event):
        self.event = event
        self.key = event.button
        
    def onspace(self,event):
        self.event=event
        self.key.append(1)
    
    def reset_data(self):
        self.xroi = []
        self.yroi = []
        self.key = 0
       
##===================================================#
#          crop image
## ==================================================#       
         
def get_crop_coords(p, filenames):
    """interactive diialog to get cropping coordinates from user input."""
    plt.ion()
    fig = plt.figure(figsize=(12,8), dpi=100,edgecolor='k',facecolor='w')
    ax = fig.add_subplot(111)
    filename = filenames[0]
    #bn = p.BASENAME#'_'.join(p.BASENAME.split("_")[:-1])
    #filename = "%s_%s.%s"%(bn,str(p.START+1).zfill(4),p.TYP)
    img=mpimg.imread(os.path.join(p.DIRC,filename))
    
    if p.ROT:
        img = np.transpose(img)
    ok = False
    ax.imshow(img,cmap='gray', origin = 'lower')
    plt.tight_layout()
    plt.title("Click on the left and on the right of the worm to get ROI (crop x).")
    clicks=clickSaver()
    while ok !=True:
        clicks.reset_data()
        cid = fig.canvas.mpl_connect('button_press_event', clicks.onclick)
        lines = []
        for i in range(2):# wait for two clicks
            plt.waitforbuttonpress()
        crops = np.sort([x if x!=None else 0 for x in clicks.xroi])
        for x in crops:
            lines.append(ax.axvline(x, ymin=0.0, ymax = 1, linewidth=4, color='w'))
        fig.canvas.mpl_disconnect(cid)
        cid2 = fig.canvas.mpl_connect('button_press_event', clicks.binary_check)
        plt.title("left mouse button: accept ROI, right button: select a new ROI")
        plt.draw()
        plt.tight_layout()
        plt.waitforbuttonpress()
        ok = bool(clicks.key%3)
        fig.canvas.mpl_disconnect(cid2)
        [ax.lines.remove(l) for l in lines]
        plt.draw()
        plt.tight_layout()
    plt.close(fig)
    print 'CROP coords:', crops
    return np.array(crops, dtype=int)
    
##===================================================#
#         input bulb location first image
## ==================================================#

def get_bulb_coords(p, filenames):
    """Extract the terminal bulb location from user input."""
    plt.ion()
    fig = plt.figure(figsize=(12,8), dpi=100,edgecolor='k',facecolor='w')
    ax = fig.add_subplot(111)
    filename = filenames[0]
    img=mpimg.imread(os.path.join(p.DIRC,filename))
    
    if p.ROT:
        img = np.transpose(img)
    if p.CROP !=None:
        img = img[:,p.CROP[0]:p.CROP[1]]
    ok = False
    ax.imshow(img,cmap='gray', origin = 'lower')
    plt.tight_layout()
    plt.title("Click on the center of the bulb. Correct with right-klick, finish with left-click.")
    clicks=clickSaver()
    while ok !=True:
        clicks.reset_data()
        cid = fig.canvas.mpl_connect('button_press_event', clicks.onclick)
        plt.waitforbuttonpress()
        bulb = (clicks.yroi[-1],clicks.xroi[-1])
        ax.plot(clicks.xroi[-1],clicks.yroi[-1], 'wo')
        ax.set_ylim(0, img.shape[0])
        ax.plot(clicks.xroi[-1],clicks.yroi[-1], 'wo')
        ax.set_ylim(0, img.shape[0])
        ax.set_xlim(0, img.shape[1])
        fig.canvas.mpl_disconnect(cid)
        cid2 = fig.canvas.mpl_connect('button_press_event', clicks.binary_check)
        plt.title("left mouse button: accept bulb location, right button: select a new bulb location.")
        plt.draw()
        plt.tight_layout()
        plt.waitforbuttonpress()
        ok = bool(clicks.key%3)
        fig.canvas.mpl_disconnect(cid2)
        if ok != True:
            ax.get_lines()[-1].remove()
            plt.draw()
        plt.draw()
    plt.close(fig)
    return np.array(bulb, dtype=int)


def find_ROI(p, filenames):
    """Uses template matching to find the region of interest (terminal bulb) in images."""
    # define bulb template
    templ = define_template(filenames[:20], p)
    # find template location in all following images
    filenames = filenames[p.START:p.END:p.INTRVL]
    # initialize data arrays
    time = np.arange(p.START,p.END,p.INTRVL)
    spacing = np.array([p.INTRVL]*(len(time)-1)+[p.END - (len(time)-1)*p.INTRVL - p.START])
    locs = np.zeros((len(filenames),2))
    confs = np.zeros(len(filenames))
    
    for cnt,fn in enumerate(filenames):
        img=mpimg.imread(os.path.join(p.DIRC,fn))
        if p.ROT:
            img = np.transpose(img)
        if p.CROP !=None:
            img = img[:,p.CROP[0]:p.CROP[1]]
        
        # find bulb by template matching
        yr,xr, conf = find_bulb(img, templ)
          
        confs[cnt] = conf
        locs[cnt] = (yr,xr)

    time, yroi, spacing = clean_auto_coords(p, time, locs, confs, spacing, n = 3)
    xroi = np.ones(len(yroi))*img.shape[1]/2.
    #write data to file
    write_data(p.OUTDIR, "roi_"+p.BASENAME, zip(time,xroi,yroi, spacing), ncol=4)

def clean_roi(time,xroi, yroi):
    """removes None and cleans up region of interest coordinates."""
    xroi_clean = []
    yroi_clean = []
    time_clean = []
    for i in range(len(xroi)):
        if xroi[i] != None and yroi[i] != None:
            xroi_clean.append(xroi[i])
            yroi_clean.append(yroi[i])
            time_clean.append(time[i])
    return xroi_clean, yroi_clean , time_clean


def write_ROI(p, filenames):
    """writes all region of interests to file"""
    plt.ion()
    fig = plt.figure(figsize=(12,8), dpi=100,edgecolor='k',facecolor='w')
    clicks=clickSaver()
    clicks.reset_data()
    fig.canvas.mpl_connect('button_press_event', clicks.onclick)
    ax = fig.add_subplot(111)
    plt.title("Click on the bulb, if worm is out of frame click outside of image.")
    
    
    filenames = filenames[p.START:p.END:p.INTRVL]
    time = np.arange(p.START,p.END,p.INTRVL)
    spacing = [p.INTRVL]*(len(time)-1)+[p.END - (len(time)-1)*p.INTRVL - p.START]
    
    try:
        for cnt,fn in enumerate(filenames):
            img=mpimg.imread(os.path.join(p.DIRC,fn))
            if p.ROT:
                img = np.transpose(img)
            if p.CROP !=None:
                img = img[:,p.CROP[0]:p.CROP[1]]
            if cnt==0:
                im=ax.imshow(img,cmap='gray', origin = 'lower')
                text = plt.text(-1.5,0.9,"%s"%fn, transform = ax.transAxes)
                text2 = plt.text(-1.5,0.8,"%i\%i"%(float(cnt),(len(filenames)-1)), transform = ax.transAxes)
                plt.draw()
                plt.tight_layout()
                plt.waitforbuttonpress()
            else:
                im.set_data(img)
                text.set_text("%s"%fn)
                text2.set_text("%i\%i"%(float(cnt),(len(filenames)-1)))
                plt.draw()
                #plt.tight_layout()
                plt.waitforbuttonpress()
    except IOError:
        print "Problem with image?"
        pass
    finally:
         xroi, yroi, time = clean_roi(time,clicks.xroi, clicks.yroi)        
    #write data to file
    write_data(p.OUTDIR, "roi_"+p.BASENAME, zip(time,xroi,yroi, spacing), ncol=4)

def parser_fill(parser):
    # arguments only for this script
    parser.add_argument('-mode', type = str, dest = 'MODE', default = 'auto', help="manual determination of bulb location or automatic.")  
    parser.add_argument('-crop', type = boolean, dest = 'CROP', default = False, help="Open crop dialog.")  

    # parallelization arguments
    parser.add_argument('-nprocs', type=int, action='store',dest='NPROCS',default=16, help="number of processes in parallelization")
    parser.add_argument('-script_dir', type=str,dest='SCRIPTDIR', default='.', help="directory where warp.py for image analysis is located")
    parser.add_argument('-account', type = str, dest = 'ACCOUNT', default = "biron", help="midway account name for submission script header")    
    parser.add_argument('-intrvl', type=int,dest='INTRVL', default = 600, help="Spacing between ROi detection.")    
    
    # arguments about I/O
    # required positional arguments
    parser.add_argument('BASENAME', type=str,metavar='basename', help="name/identifier for outputand scipts eg. yl0027")
    parser.add_argument('DIRC', metavar='directory', type=str,help="directory containing images")
    # non-required arguments
    parser.add_argument('-outdir', type=str,default='../results/',dest='OUTDIR', help="directory for output")    
    parser.add_argument('-typ', type=str, action='store',dest='TYP',default='png', help="image type by extension")
    parser.add_argument('-start', type=int,default=0,dest='START', help="time stamp starting eg. frame 0 -> 0")
    parser.add_argument('-end', type=int,default=225001,dest='END',  help="time stamp ending in frame number")
    
    # arguments for image analysis
    parser.add_argument('-rotate', type=boolean,dest='ROT',default=False, help="rotate image, binary")
    parser.add_argument('-chunk', type = int, dest = 'CHUNK', default = 60, help="spacing between drift corrections.")
    parser.add_argument('-roisize', type=int, dest = 'ROISIZE', default=120,help="size [px] region of interest around bulb for image analysis.")
    parser.add_argument('-size', type = int, dest = 'SIZE', default = 85, help="size of matching template (half width).")
    parser.add_argument('-entropybins', type = float, nargs=3,dest = 'BINS', default = (0.2,1,64), help="histogram bins, arguments to numpy.linspace. (min, max, nbin)")
    
def main():
    #read arguments
    parser = argparse.ArgumentParser(description='Main_1.0: Interactive program \
    to get from raw images to slurm submission script with basic GUI.', version="1.0", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_fill(parser)
    p=parser.parse_args()
    # sort image files for later
    filenames = os.listdir(p.DIRC)
    filenames  = [f for f in filenames if ".%s"%p.TYP in f]
    filenames = np.array(natural_sort(filenames))[p.START:p.END]
    # define crop area
    if p.CROP:
        cropx = get_crop_coords(p, filenames)
    else:
        cropx=(0,-1)
    parser.add_argument('-cropx', type=int,nargs=2,dest='CROP', help="xmin and xmax for cropping image")
    parser.add_argument('-bulb', type=int,nargs=2,dest='BULB', help="bulb location first image")
    p=parser.parse_args()
    p.CROP = cropx
    
    # define bulb location first image
    bulb = get_bulb_coords(p, filenames)
    p.BULB = bulb
    print "Bulb location:",bulb
    
    if p.MODE == 'auto':
        # automatically determine bulb locations in following images with interval p.intrvl
        find_ROI(p, filenames)
        # create slurm submission script for midway
    elif p.MODE =='manual':
        write_ROI(p, filenames)
        
    write_slurm_file(p)
if __name__=="__main__":
    main()
                
