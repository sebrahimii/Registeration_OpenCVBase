#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:33:51 2022

@author: sami
"""

import cv2
import numpy as np
from deepdish.io import load
from tqdm import tqdm

from matplotlib import pyplot as plt
from skimage import exposure
from deepdish.io import load, save
from natsort import natsorted
import os, sys
NCT = '/home/sami/data/PROG/1-Code/-1-TORNGATS/CODE/'
RES = '/home/sami/data/PROG/1-Code/-1-TORNGATS/CODE/'
if RES not in sys.path:
    sys.path.append(RES)
if NCT not in sys.path:
    sys.path.append(NCT)
from nct1 import contrast_correction
from nct1 import multiplot
def plot_registered (data ,  x,y,reg_with='first',idx=None, reg_idxs=None):
    """
    Plot based on reg_idxs indeces

    Parameters
    ----------
    data : ndarray
        data sequence
    x : int
        number of rows.
    y : int
        number of cols.
    reg_with : string, optional
        value can be first, next, index.
        The default is 'first'.
    idx : int, optional
        the selected-index for generating overlay image. The default is None.
    reg_idxs : list, optional
        List of index for plot. The default is None.

    Returns
    -------
    None.

    """
    if  reg_idxs is None:
        reg_idxs = range(len(data))
        
    if reg_with =='first' :
        
        imgs=[]
        for i in reg_idxs:
            if i != len(data):
                imgs.append(overlay_images([data[0],data[i]]))
    elif reg_with =='next':
        
        imgs=[]
        for i in reg_idxs:
            if i != len(data):
                imgs.append(overlay_images([data[i],data[i+1]]))
        
    elif reg_with =="index":
        imgs =[]
        for i in reg_idxs:
            if i != len(data):
                imgs.append(overlay_images([data[i], data[idx]]))
    multiplot(imgs,x,y)
    
def overlay_images(imgs, equalize=False, aggregator=np.mean):
    '''
    Generate Overlay of images

    Parameters
    ----------
    
    imgs : list of image or 3d array
        list of image to plot as a overlay image
        
    equalize : Boolean, optional
        DESCRIPTION. The default is False.
        To enhance the visibility (contrast) of the image 
    aggregator : TYPE, optional
        DESCRIPTION. The default is np.mean.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)

    return aggregator(imgs, axis=0)

def image_show(image, cmap='gray'):
    """
    Plot image

    Parameters
    ----------
    image : ndarray
        Image to plot.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'gray'.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def register_cv2(im1, im2, wmode , gradient_base= False , cont_correction_base = False,
                 number_of_iter = 5000, term_eps = 1e-10 ):
    """
    

    Parameters
    ----------
    im1 : ndarray
        reference image.
    im2 : ndarray
        target image(misaligned image).
    wmode : string
        'affine', 'translation', 'homography', 'euclidean'.
    
    number_of_iter : int, optional
        number of iterations. The default is 5000.
    term_eps : TYPE, optional
        termination eps. The default is 1e-10.

    Returns
    -------
    im2_aligned: Aligned imag
    warp_matrix: transformation matrix

    """
    
    if len(im1.shape)==3:
        im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    if len(im2.shape)==3:
        im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    sz = im1.shape
    
    if wmode =='affine':
        warp_mode = cv2.MOTION_AFFINE
    elif wmode == 'translation':
        warp_mode = cv2.MOTION_TRANSLATION
    elif wmode == 'homography':
        warp_mode = cv2.MOTION_HOMOGRAPHY
    elif wmode == 'euclidean':
        warp_mode = cv2.MOTION_EUCLIDEAN

    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Specify the number of iterations.
    number_of_iterations = number_of_iter;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = term_eps;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    if gradient_base and cont_correction_base:
        
        (cc, warp_matrix) = cv2.findTransformECC (get_gradient(contrast_correction(im1)),get_gradient(contrast_correction(im2)),warp_matrix, warp_mode, criteria)
    elif gradient_base and not cont_correction_base:
        
        (cc, warp_matrix) = cv2.findTransformECC (get_gradient(im1),get_gradient(im2),warp_matrix, warp_mode, criteria)
    elif cont_correction_base and not gradient_base :
        (cc, warp_matrix) = cv2.findTransformECC (contrast_correction(im1),contrast_correction(im2),warp_matrix, warp_mode, criteria)
    else:
        (cc, warp_matrix) = cv2.findTransformECC (im1,im2,warp_matrix, warp_mode, criteria)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    return im2_aligned , warp_matrix
#%%
def register_by_mat(im , warp_matrix, wmode):
    '''
    

    Parameters
    ----------
    im : ndarray
        misaligned image.
    warp_matrix : ndarray
        transformation matrix.
    wmode : string
        transformation type.

    Returns
    -------
    im_aligned : ndarray
        aligned image.

    '''
    if len(im.shape)==3:
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    sz = im.shape
    
    # Define the motion model
    if wmode =='affine':
        warp_mode = cv2.MOTION_AFFINE
    elif wmode == 'translation':
        warp_mode = cv2.MOTION_TRANSLATION
    elif wmode == 'homography':
        warp_mode = cv2.MOTION_HOMOGRAPHY
    elif wmode == 'euclidean':
        warp_mode = cv2.MOTION_EUCLIDEAN
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        im_aligned = cv2.warpPerspective (im, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im_aligned = cv2.warpAffine(im, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    return im_aligned


def register_selectedIndex(data,mov_mat,wmode, gr_base ,cont_corr_base,n_iter= 5000, t_eps = 1e-10, show_overlay= False):
    '''
    

    Parameters
    ----------
    data : ndarray
        image sequence.
    mov_mat : list
        list of indeces to be aligned.
    wmode : ndarray
        transformation matrix.

    Returns
    -------
    new_seq : ndarray
        aligned image sequence

    '''
    frames,rows, cols = data.shape
    
    for idx, i in enumerate(tqdm(mov_mat)):
        
        print( idx, i)
        # create new_seq 
        if idx ==0:
            new_seq= data[:i+1]
            if len(mov_mat)==1:
                print('-----------------len movmat =1')
                for j in range(i+1,len(data) ):
                    
                    if gr_base or cont_corr_base:
                        _, mat = register_cv2(new_seq[-1], data[j], wmode,
                                          gradient_base=gr_base, cont_correction_base=(cont_corr_base),
                                          number_of_iter = n_iter, term_eps = t_eps  )
                        reg = register_by_mat(data[j], mat, wmode)
                    else:
                        reg, mat = register_cv2(new_seq[-1], data[j],
                                              wmode, number_of_iter = n_iter, term_eps = t_eps  )
                    
                    new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols)))
                    
                    print('stack image!')
                print(len(data)-i-1, ' images to stack!')
        # if first index AND if not reach to the list of index
        
        if idx ==0 and idx != len(mov_mat)-1:
            print(' -----------------idx=0 and not the last index!')
            # single-registeration 
            if gr_base or cont_corr_base:
                _, mat = register_cv2(new_seq[-1],data[i+1], wmode , 
                                      gradient_base= gr_base, cont_correction_base= cont_corr_base,
                                      number_of_iter = n_iter, term_eps = t_eps )
                reg = register_by_mat(data[i+1], mat, wmode)
                
            else:
                reg, mat = register_cv2(new_seq[-1], data[i+1], 
                                        wmode , gradient_base= gr_base, cont_correction_base= cont_corr_base,
                                        number_of_iter = n_iter, term_eps = t_eps )
            new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols)))
            print('stack image')
            
            #compare the index value in list to take a disition for next step
            # check if there is any frame to register based on computed tmats 
            if  (mov_mat[idx+1])- (i+1) >0 :
                print((mov_mat[idx+1] -i-1),' images to stack!')
                for j in range(i+2,mov_mat[idx+1]+1 ):
                    # print(j)
                    reg = register_by_mat(data[j], mat, wmode)
                
                    new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols)))
            else:
                print('Nothing to stack!')

        # if reach to the end of list 
        elif idx == len(mov_mat)-1:
            print('----------------- idx is last index')
            if gr_base or cont_corr_base:
                _, mat = register_cv2(new_seq[-1],data[i+1], wmode,
                                      gradient_base= gr_base, cont_correction_base= cont_corr_base,
                                      number_of_iter = n_iter, term_eps = t_eps)
                reg = register_by_mat(data[i+1], mat, wmode)
                
            else:
                reg, mat = register_cv2(new_seq[-1], data[i+1], 
                                        wmode, gradient_base= gr_base, cont_correction_base= cont_corr_base,
                                        number_of_iter = n_iter, term_eps = t_eps)
            new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols)))
            print('stack image')
            
            
            for j in range(i+2,len(data) ):
                # print(j)
                reg = register_by_mat(data[j], mat, wmode)
                
                new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols))) 
                
            print(len(data)- idx-1, ' images to stack!')
        # if not reached to the end and not the begining the list
        else:
            print('------------------- else loop!')
            if gr_base or cont_corr_base:
                _, mat = register_cv2(new_seq[-1], data[i+1], wmode,
                                      gradient_base= gr_base, cont_correction_base= cont_corr_base,
                                      number_of_iter = n_iter, term_eps = t_eps)
                reg = register_by_mat(data[i+1], mat, wmode)
            else:
                reg, mat = register_cv2(new_seq[-1], data[i+1], wmode,
                                        number_of_iter = 5000, term_eps = 1e-10)
            new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols)))
            print('stack image!')

            if  (mov_mat[idx+1])- (i+1) >0 :
                print((mov_mat[idx+1] -i-1),' images to stack!')
                for j in range(i+2,mov_mat[idx+1]+1 ):
                    
                    reg = register_by_mat(data[j], mat, wmode)
                
                    new_seq = np.vstack((new_seq, reg.reshape(1,rows,cols)))
            else:
                print('Nothing to stack!')
    if show_overlay:                    
        for i in mov_mat:
            plt.figure()
            plt.imshow(overlay_images([new_seq[i], new_seq[i+1]]))            
    return new_seq
#%%

def get_gradient(im) :
    """
    generate the gradient base image

    Parameters
    ----------
    im : ndarray (rows X Cols)
        image.

    Returns
    -------
    grad : ndarray
        gradient-base image.

    """
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
     
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def plot_movmat(data, mov_mat):
    """
    Plot the mov_mat (indexs which need registeration)
    plot the overlay of each index of mov-mat with the next image 

    Parameters
    ----------
    data : ndarray
        data to plot
    mov_mat : list
        list of indeces which need registeration.

    Returns
    -------
    None.

    """
    for i in mov_mat:
        image_show(overlay_images([data[i], data[i+1]]))

#%%

def compare_tmats(tmats, comp_with='previous', flatten = True, range_value=10):
    '''
    

    Parameters
    ----------
    tmats : ndarray- 3d
        DESCRIPTION. Transformation matrix for an stacked images
        
    comp_with : STRING , optional
        DESCRIPTION. The default is 'previous'.
        
    flatten : bool, optional
        DESCRIPTION. The default is True.
        
    range_value : TYPE, optional
        DESCRIPTION. The default is 10.
        the value which is >= range value will be returned.

    Returns
    -------
    subt_ftmats : ndarray
        the subtracted transformation correspond to comp_with value.
        
    range_value_list : list
        DESCRIPTION.
        List of image indexes which is higher than range_value
        
    '''
    
    frames, rows, cols = tmats.shape
    
    if flatten and comp_with == 'previous':
        tmats = np.round(tmats,2)
        tmats = np.asarray([t.flatten() for t in tmats])
        subt_ftmats=[]
        # subtract flatten transformation matrix from previous rows to find 
        #big movement according to range_value
        for i in range(frames):
            if i!= frames -1:  
                subt_ftmats.append(tmats[i]-tmats[i+1])
        subt_ftmats = np.asarray(subt_ftmats)
        
        # check all the columns for the desire indexes
        range_value_list = []
        for i in range(rows*cols):
            col=abs(subt_ftmats[:,i])
            mv = np.asarray( [ [n,i] for n,i in enumerate(col) if i>=range_value ]).astype(int)
            if len(mv)!=0:
                range_value_list.append(mv )
        # take union of the found index for final list 
        union_list=[]
        for i in range(len(range_value_list)):
            union_list = union_list+ list(range_value_list[i][:,0])
        union_list = natsorted(list(set(union_list)))
        # final list of indexes 
        range_value_list = union_list
    elif flatten and comp_with == 'first':
        print('COME BACK SOON TO DO! ')
        
    return subt_ftmats, range_value_list
#%%
if __name__ =="__main__":
    print()
        
    #%%
    