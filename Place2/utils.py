from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy import misc

def denormalize_images(images):
    return np.uint8((images+1)*127.5)

def images_on_grid(images, n_h, n_w, save_path=None):
    '''
    FUNC: draw images on a n_h x n_w grid (will save the grid image if save_path specified)
    Arguments:
        images: a numpy array containing images, where all images have the same size.
                the 0th dimension specifies which image, e.g. first_image = images[0]
        n_h, n_w: height and width of the grid, n_h*n_w must equal to #images
        save_path: path to save the grid 
    Returns:
        merged_images: a grid of images
    '''
    n_img = images.shape[0]
    assert(n_img==n_h*n_w)
    try:
        img_h, img_w, depth = images[0].shape
    except:
        img_h, img_w = images[0].shape
        depth = 1
    # merge images
    merged_images = np.zeros((n_h*img_h,n_w*img_w,depth), dtype=np.uint8)
    for i in xrange(n_img):
        i_img = images[i]
        y = i // n_h
        x = i % n_w
        merged_images[y*img_h:(y+1)*img_h, x*img_w:(x+1)*img_w, :] = i_img
    merged_images = merged_images.squeeze()
    # save or not
    if save_path is not None:
        save_image(merged_images, save_path)
    
    return merged_images

def make_gif():
    raise NotImeplementedError
        
def check_path(path):
    '''
    FUNC: convert path (can be realtive path or using ~) to absolute path and check
          the path exists.
    '''
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise NameError('Not such path as {}'.format(path))

    return path

def check_dir(dir_path):
    '''
    FUNC: check directory path, if no such directory, then create one
    '''
    dir_path = os.path.abspath(os.path.expanduser(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        if os.path.isdir(dir_path):
            return dir_path
        else:
            raise NameError('{} is not a directory'.format(dir_path))
    
    return dir_path

def save_image(img, filename, verbose=False):
    '''
    FUNC: save image at filename. If it's color image, channel follows r,g,b.
          Image shape can be (*,*), (*,*,3), (*,*,4)
    '''
    #filename = check_path(filename)
    if verbose:
        print('Save image at {}'.format(filename))
    misc.imsave(filename, img)

