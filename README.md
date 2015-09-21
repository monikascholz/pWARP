# pWARP
Automatically analyze C. elegans feeding behavior. 
A couple of python scripts to analyze large stacks of image data in worm feeding. 

## Dependencies
tested on:

* python 2.7+
* numpy 1.9.2
* matplotlib 1.4+
* scipy 0.13.3

## Files
### main script
check_movie.py - main script with a basic matplotlib based GUI. Performs these tasks in one clean main script.

1. Crop image (GUI based)
2. Select region of interest (ROI)  (GUI based)
3. 
  * Manual mode: select ROI for each subprocess
  * Automatic mode: Use template correlation to find ROI throughout movie.
4. write batch submission script

### under the hood
WARP_parallel.py - divides work to cores for parallelization using python's multiprocessing module

warp.py - workhorse, the actual image analysis code. This can also be run separarely for simpler tasks. See below.

fluowarp.py - work in progress: track GCamp neuron activity.

## Usage

check_movie essentially creates an automatic submission script.

Parameter details below.

```bash
positional arguments:
  basename              name/identifier for outputand scipts eg. yl0027
  directory             directory containing images
  outdir                directory for output

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -mode MODE            manual determination of bulb location or automatic.
  -crop CROP            Open crop dialog.
  -nprocs NPROCS        number of processes in parallelization
  -script_dir SCRIPTDIR
                        directory where warp.py for image analysis is located
  -account ACCOUNT      midway account name for submission script header
  -intrvl INTRVL        Spacing between ROi detection.
  -typ TYP              image type by extension
  -start START          time stamp starting eg. frame 0 -> 0
  -end END              time stamp ending in frame number
  -rotate ROT           rotate image, binary
  -chunk CHUNK          spacing between drift corrections.
  -roisize ROISIZE      size [px] region of interest around bulb for image
                        analysis.
  -size SIZE            size of matching template (half width).
  -entropybins BINS BINS BINS
                        histogram bins, arguments to numpy.linspace. (min,
                        max, nbin)

```
Running the script e.g. with these parameters:
```bash
python check_movie.py ser6 ../images/im_folder ../results/ -roisize 250 -entropybins 0.06 0.5 30 -typ bmp
```
results on this output file

```bash
#!/bin/sh 
#SBATCH --account=ACCTNAME
#SBATCH --job-name=JOBNAME
#SBATCH --output=JOBNAME.out
#SBATCH --exclusive
#SBATCH --time=1:0:0

echo "start time: `date`"
 python WARP_parallel.py -nprocs 16 -type bmp -basename ser6 -directory "../images/5Ht_10s_switch" -roi_file "../results/roi_ser6"     -outdir "../results/" -cropx 5 65 -rotate False -chunk 60 -roisize 250 -entropybins 0.06 0.5 30.0 
```

run this script using sbatch. Note the positional and optional arguments.

## Direct access to image analysis
The warp.py file can be used as an independent image analysis library.
If this is the case, import warp.py and call the function
```
warp_detector(params)
```
in your code. The argument is a dictionary of required parameters;

```python
params = {'cropx': [0, -1], 
    'chunk': 60, 
    'roisize': 120, 
    'start': 0, 
    'end': 1801, 
    'rotate': False, 
    'directory': '../images/df0432/',
    'y0': 469.84462300000001,
    'x0': 69.5, 
    'type': 'png', 
    'basename': 'df0432',
    'outdir': '../results/'}
```
The parameters correspond to the list in the section on Usage.
Additionally, x0 and y0 are the coordinates of the region of interest. The images are cropped to an area of +/- roisize around these coordinates after alignment.

## Contact
mscholz@uchicago.edu
