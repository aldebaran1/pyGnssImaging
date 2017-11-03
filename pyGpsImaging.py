#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:12:41 2017

@author: Sebastijan Mrak <smrak@bu.edu>
"""

import numpy as np
import os
import numpy.ma as ma
import datetime
import h5py
import multiprocessing  
import subprocess
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import interpolate
from argparse import ArgumentParser
from PIL import Image

YMLFN = ''

def getNeighbours(image,i,j):
    nbg = []
    for k in np.arange(i-1, i+2):
        for l in np.arange(j-1, j+2):
            try:
                nbg.append(image[k,l])
            except Exception as e:
                pass
    return np.array(nbg)

def filterMask(im,i,j, mask_size=3, ftype='mean'):
    nbg = []
    for k in np.arange(i-1, i+mask_size-1):
        for l in np.arange(j-1, j+mask_size-1):
                try:
                    if np.isfinite(im[k,l]):
                        nbg.append(im[k,l])
                except Exception as e:
                    pass
    if ftype == 'mean':
        return np.mean(np.array(nbg))
    elif ftype == 'median':
        return np.median(np.array(nbg))

def imageFilter(im, mask_size=3, ftype='mean'):
    im_filt = im.copy()
    for i in np.arange(0,im.shape[0]):
        for j in np.arange(0,im.shape[1]):
            im_filt[i,j] = filterMask(im,i,j, mask_size=mask_size, ftype=ftype)
    return im_filt
            
def fillPixels(im, N=1):
    for n in range(N):
        for i in np.arange(0,im.shape[0]):
            for j in np.arange(0,im.shape[1]):
                if np.isnan(im[i,j]):
                    nbg = getNeighbours(im,i,j)
                    if sum(np.isfinite(nbg)) >= 4:
                        ix = np.where(np.isfinite(nbg))[0]
                        avg = np.mean(nbg[ix])
                        im[i,j] = avg
    return im

def interpolateImage(im, xgrid=0, ygrid=0, res=1, method='cubic'):
    im_mask = np.ma.masked_invalid(im)
    
    xd = abs(xgrid[0,0] - xgrid[-1,0]) / res * 1j
    yd = abs(ygrid[0,-1] - ygrid[0,0]) / res * 1j
    xgrid2, ygrid2 = np.mgrid[xgrid[0,0] : xgrid[-1,0] : xd, 
                              ygrid[0,-1] : ygrid[0,0] : yd]
    
    x1 = xgrid[~im_mask.mask]
    y1 = ygrid[~im_mask.mask]
    newarr = im_mask[~im_mask.mask]
    
    GD = interpolate.griddata((x1,y1), newarr.ravel(), (xgrid2, ygrid2), method=method)
    
    return xgrid2, ygrid2, GD

def makeGrid(ylim=[25,50],xlim=[-110,-80],res=0.5):
    xd = abs(xlim[0] - xlim[1]) / res * 1j
    yd = abs(ylim[0] - ylim[1]) / res * 1j
    xgrid, ygrid = np.mgrid[xlim[0]:xlim[1]:xd, ylim[0]:ylim[1]:yd]
    z = np.nan*np.zeros((xgrid.shape[0], xgrid.shape[1]))
    
    return xgrid, ygrid, z

def returnIndex(x, i, delta):
    if sum(sum(np.isfinite(x))) > 0:
        idval = np.where(np.isfinite(x))[0][0]
        lst = np.arange(i-delta,i+delta,1)
        ix = lst[idval]
        return ix
    else:
        return np.nan
    
def getImageIndex(x, y, xlim, ylim, xgrid, ygrid):
    if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
        idy = abs(ygrid[0,:] - y).argmin()
        idx = abs(xgrid[:,0] - x).argmin()
    else:
        idy = np.nan
        idx = np.nan
    return idx, idy

def checkImagePath(save_dir):
    if not os.path.exists(save_dir):
        subprocess.call('mkdir {}'.format(save_dir), shell=True)
        
def plotTotalityMask(m,time):
    totality_path = h5py.File('/home/smrak/Documents/eclipse/totality.h5', 'r')
    lat = totality_path['path/center_lat'].value
    lon = totality_path['path/center_lon'].value
    Tt = totality_path['path/time'].value
    
    idt = abs(Tt - time).argmin()
    if abs(Tt - time).min() < 500:
        x,y = m(lon[idt], lat[idt])
        
        m.scatter(x, y, s=120, facecolors='none', edgecolors='m', linewidth=2)
        m.scatter(x, y, s=1500, facecolors='none', edgecolors='k', linewidth=0.5, linestyle='--')
        m.scatter(x, y, s=15000, facecolors='none', edgecolors='k', linewidth=0.5, linestyle='--')
        m.scatter(x, y, s=200000, facecolors='none', edgecolors='k', linewidth=0.5, linestyle='--')
        m.scatter(x, y, s=250000, facecolors='none', edgecolors='k', linewidth=0.5, linestyle='--')
        m.scatter(x, y, s=300000, facecolors='none', edgecolors='k', linewidth=0.5, linestyle='--')

def plotMap(latlim=[20, 65], lonlim=[-160, -70], center=[39, -86],
            parallels=[20,30,40,50], 
            meridians = [-120,-110, -100, -90, -80,-70],
            epoto=False, totality=True):
    
    (fig,ax) = plt.subplots(1,1,facecolor='w', figsize=(12,8))
    m = Basemap(lat_0=40, lon_0=-95,llcrnrlat=latlim[0],urcrnrlat=latlim[1],
                llcrnrlon=lonlim[0],urcrnrlon=lonlim[1],
                projection='merc')#, resolution='i', ax=ax)
    
#    m.drawparallels(parallels,labels=[False, True, True, True], linewidth=1)
#    m.drawmeridians(meridians,labels=[True,True,False,True], linewidth=1)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    if epoto == True:
        m.etopo()
    
    if totality:
        totality_path = h5py.File('/home/smrak/Documents/eclipse/totality.h5', 'r')
        north_lat = np.array(totality_path['path/north_lat'])
        north_lon = np.array(totality_path['path/north_lon'])
        south_lat = np.array(totality_path['path/south_lat'])
        south_lon = np.array(totality_path['path/south_lon'])
    
        X1,Y1 = m(north_lon, north_lat)
        X2,Y2 = m(south_lon, south_lat)
        m.plot(X1,Y1, c='b')
        m.plot(X2,Y2, c='b')
        
    return fig, ax, m

def plotImage(x,y,z, time=0, clim=[], cmap='jet', save_dir=''):
    fig = plt.figure(figsize=(12,12))
    title = datetime.datetime.utcfromtimestamp(time)
    plt.title(title)
    Zm = ma.masked_where(np.isnan(z),z)
    plt.pcolormesh(x, y, Zm, cmap=cmap)
#    im = Image.fromarray(Zm)
#    plt.pcolormesh(x, y, Zm, edgecolor='w', lw=0.005, cmap=cmap)
    plt.clim(clim)
    plt.colorbar()
    plt.savefig('{}{}.png'.format(save_dir,time))
#    im.save('mptest18/{}.tif'.format(time))
    plt.close(fig)
    
def plotImageMap(m,ax, xgrid,ygrid,z,time=0,clim=[],cmap='jet',
                 save_dir='', totality=False, raw_image=False):
    
    title = datetime.datetime.utcfromtimestamp(time)
    ax.set_title(title)
    x,y = m(xgrid, ygrid)
    Zm = ma.masked_where(np.isnan(z),z)
    gca = m.pcolormesh(x,y,Zm, cmap='jet')
    if totality:
        plotTotalityMask(m, time)
    gca.set_clim(clim)
    plt.colorbar(gca)
    if save_dir is not None:
        checkImagePath(save_dir)
        plt.savefig('{}{}.png'.format(save_dir,time))
    if raw_image:
        checkImagePath(save_dir+'tif')
        im = Image.fromarray(Zm)
        im.save('{}tif/{}.tif'.format(save_dir,time))

def singleImage(i):
    stream = yaml.load(open(YMLFN, 'r'))
    fname = stream.get('hdffilename')
    f = h5py.File(fname, 'r')
    ylim = stream.get('ylim')
    xlim = stream.get('xlim')
    im_resolution = stream.get('im_resolution')
    save_dir = stream.get('save_dir')
    delta = stream.get('delta')
    image_interpolate = stream.get('image_interpolate')
    interpolate_method = stream.get('interpolate_method')
    interpolate_resolution = stream.get('interpolate_resolution')
    fillpixel_iter = stream.get('fill_pixel_iter')
    image_mask_size = stream.get('image_mask_size')
    image_filter_type = stream.get('image_filter_type')
    clim = stream.get('clim')
    totality_mask = stream.get('totality_mask')
    raw_image = stream.get('raw_image')
    t = f['obstimes'].value
    
    xgrid, ygrid, im = makeGrid(ylim=ylim, xlim=xlim, res=im_resolution)
    for k in f.keys():
        if k != 'obstimes':
            tmp = f[k+'/lat'][i-delta : i+delta]
            idt = returnIndex(tmp, i, delta)
            if np.isfinite(idt):
                lat = f[k+'/lat'][idt]
                lon = f[k+'/lon'][idt]
                residual = f[k+'/res'][idt]
                
                for j in np.where(np.isfinite(residual))[0]:
                    idx, idy = getImageIndex(x = lon[j], y = lat[j],
                                             xlim = xlim, ylim = ylim,
                                             xgrid = xgrid, ygrid = ygrid)

                    if np.isfinite(idx) and np.isfinite(idy):
                        if np.isnan(im[idx,idy]):
                            im[idx,idy] = residual[j]
                        else:
                            im[idx,idy] = (im[idx,idy] + residual[j]) / 2
    if fillpixel_iter > 0:
        im = fillPixels(im, N=fillpixel_iter)
    if image_interpolate:
        xgrid, ygrid, im = interpolateImage(im, xgrid, ygrid,res=interpolate_resolution, method=interpolate_method)
    if image_filter_type is not None:
        im = imageFilter(im, mask_size=image_mask_size, ftype=image_filter_type)
        
    fig, ax, m = plotMap()
    plotImageMap(m,ax,xgrid,ygrid,im,time=t[i],clim=clim,cmap='jet',
                 save_dir=save_dir, totality=totality_mask, raw=raw_image)
    plt.close(fig)

def runImaging(f, iterate):
    for i in iterate:
        p = multiprocessing.Process(target=singleImage, args=(i,))
        p.start()
        p.join()

def main(config_file=None, datafile='', svdir='', tiff=False):
    global YMLFN
    
    if config_file is None:
        if svdir == '':
            svdir = 'images/'
        if datafile == '':
            datafile = '/media/smrak/Eclipse2017/Eclipse/hdf/linecut_trial_130_60_imode2_ed1000_14.h5'
        
        decimate = 30
        delta = int(decimate/2)
        im_resolution = 0.5
        interpolate_resolution = 0.25
        
        ylim=[22,50]
        xlim=[-125,-65]
        clim = [-0.2, 0.2]
        
        fill_pixel_iter = 3
        image_interpolate = False
        interpolate_method = 'nearest'
        image_filter_type = 'median'
        image_mask_size = 3
        
        skipimage = 3
        
        YMLFN = 'plottinparams.yaml'
        datadict = {'hdffilename': datafile, 
                    'decimate': decimate,
                    'xlim':xlim, 
                    'ylim':ylim, 
                    'im_resolution':im_resolution,
                    'delta': delta,
                    'skip_image': skipimage,
                    'save_dir': svdir, 
                    'fill_pixel_iter': fill_pixel_iter,
                    'image_interpolate': image_interpolate,
                    'interpolate_method': interpolate_method,
                    'interpolate_resolution': interpolate_resolution,
                    'image_filter_type': image_filter_type,
                    'image_mask_size': image_mask_size,
                    'clim': clim,
                    'raw_image': tiff}
        with open(YMLFN, 'w') as outfile:
            yaml.dump(datadict, outfile, default_flow_style=True) 
    else:
        YMLFN = config_file
        stream = yaml.load(open(config_file, 'r'))
        decimate = stream.get('decimate')
        skipimage = stream.get('skip_image')
        datafile = stream.get('hdffilename')
    
    f = h5py.File(datafile, 'r')
    timearray = f['obstimes'].value
    iterate = np.arange(int(decimate), timearray.shape[0]-decimate, decimate*skipimage)
    runImaging(datafile,iterate)

    
if __name__ == '__main__':
    
    p = ArgumentParser()
    p.add_argument('-d', "--data_file", help='path to the HDF data file',
                   default='', type=str)
    p.add_argument('-c', "--cfg_yaml_file", help='path to the configuration file',
                   default=None)
    p.add_argument('-s', "--save_directory", help='path for directory to save images',
                   default='', type=str)
    p.add_argument('--raw', "--raw", help="save raw images as.tif", default=False)
   
    P = p.parse_args()
#    print (P.save_directory)
    main(config_file=P.cfg_yaml_file, datafile=P.data_file,svdir=P.save_directory, tiff=P.raw)
