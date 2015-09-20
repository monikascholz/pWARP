#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:31:43 2013
This is a wrapper for the python pump detection to make multiple jobs and split movies into snippets.
@author: Monika Scholz
"""
import numpy as np
import argparse
import warp
from multiprocessing import Pool
import time
import traceback
import sys
##===================================================#
#          I/O 
## ==================================================#

def boolean(string):
      string = string.lower()
      if string in ['0', 'f', 'false', 'no', 'off']:
          return False
      elif string in ['1', 't', 'true', 'yes', 'on']:
          return True
      else:
          raise ValueError()

def read_data(fname):
    """reads traces to file.""" 
    data=[]
    with open(fname, 'r') as f:
        for line in f:
            line.strip()
            ln=line.split()
            data.append([float(l) for l in ln])
    return np.array(data)

def treat_bool(s):
    return s in ['True','true', '1']
#dispatch dict
safe_unpack={'int': int,
             'float':float,
             'bool':treat_bool
}


def make_chunks_of_works(dat, params):
    """creates dicts for each process to use."""
    chunks = []
    p_format={
        'start': 0,# start
        'end': 300,# start and end number of frames to evaluate
        'y0':210, # initial roi coords y
        'x0':30 # nd x
    }
    params.update(p_format)
    for t,x,y,incr in dat:
        par=params.copy()
        par['start'] = int(t)
        par['end'] = int(t+incr+1)
        par['x0'] = x
        par['y0'] = y
        chunks.append(par)
    return chunks

def run_func(chunk):
    """get a real traceback for exceptions even when multiprocessing"""
    try:
        return warp.warp_detector(chunk)
    except:
        # Put all exception text into an exception and raise that
        sys.stdout.flush()
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def parser_fill(parser):
    # parallelization
    parser.add_argument('-nprocs', type=int, action='store',dest='NPROCS',default=1, help="number of processes")    
    parser.add_argument('-roi_file', type=str, help="file with roi data (full path)", required=True)
    
    # image related
    parser.add_argument('-roisize', type=int, default=120,help="size [px] region of interest around bulb")
    parser.add_argument('-chunk', type = int, default = 60, help="spacing between drift corrections.")
    parser.add_argument('-type', type=str, default='png', help="image file ending eg. jpg")
    
    parser.add_argument('-basename', type=str,  help="basename images without frame number", required=True)
    parser.add_argument('-directory', type=str, help="video directory", required=True)
    parser.add_argument('-outdir', type=str, help="where results go directory", required=True)
    parser.add_argument('-cropx', type=int,nargs=2, help="xmin and xmax for cropping image")
    parser.add_argument('-rotate', type=boolean, default =False,help="rotate image")
    parser.add_argument('-entropybins', type = float, nargs=3, default = (0.2,1,64), help="histogram bins, arguments to numpy.linspace. (min, max, nbin)")
    

def localmain(params):
    p = Pool(params['NPROCS'])
    #create chunks of work by having dicts for each process
    dat = read_data(params['roi_file'])  
    chunks = make_chunks_of_works(dat, params)
    p.map(run_func,chunks)
    
if __name__ == "__main__":
    #uses up to 16 cores to parallelize processes.
    t1 = time.time()
    parser = argparse.ArgumentParser(description='WARP - parallel image analysis', version="1.1")
    parser_fill(parser)
    p=parser.parse_args()    
    params = vars(p)
    localmain(params)
    t2 = time.time()
    print "Task completed in %f seconds."%(t2-t1)
    
