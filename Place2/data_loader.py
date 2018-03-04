from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import misc
import os
import random

INTENSE_VERBOSE = False # true in debug mode

class indoor_input(object):
    def __init__(self, dataset_dir, batch_size, 
                 train_img_size, do_shuffle=True, verbose=False):
        '''
        Arguments:
            dataset_dir: a string (can be changed) specifies root directory of dataset
            batch_size: size of a minibatch
            train_img_size: size of images to be used in training
            do_shuffle: a boolean to determine whether dataset shuffle performed as an epoch ends
            verbose: verbose
        '''
        # input arguments
        self._dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
        self._batch_size = batch_size
        self._train_img_size = train_img_size
        self._do_shuffle = do_shuffle
        self._verbose = verbose
        # the index of first image in current batch
        self._read_head = 0
        # data structure to save dataset(image paths and corresponding labels)
        self._dataset = []
        # total number of samples in given dataset
        self._n_samples = None
        # labels' names
        self._label_names = []

        # parse dataset directory
        self._parse_dir() # self._dataset and self._n_samples should be updated
        # extra tape for indexing, used for shuffled index, --> done after n_samples updated 
        # access an image --> all_images_in_dataset[self._index[self._read_head]] 
        self._index = np.arange(self._n_samples)
        # number of classes --> done after self._label_names is updated
        self._n_classes = len(self._label_names)
        # print details
        if self._verbose:
            print('\nIndoor dataloader details:')
            print('\tdataset root directory: {}'.format(self._dataset_dir))
            print('\tbatch_size: {}'.format(self._batch_size))
            print('\tdo_shuffle: {}'.format(self._do_shuffle))
            print('\tn_classes: {}'.format(self._n_classes))
            print('\tn_samples: {}'.format(self._n_samples))
            print('\tlabel_names: {}'.format(self._label_names))
            print('\n')
        # initial shuffle --> done after self._n_samples is updated
        self.shuffle()

    def shuffle(self):
        '''
        FUNC: random shuffle entire dataset, only can be used after entire epoch completed
        '''
        if self._read_head==0:
            self._index = np.random.permutation(self._n_samples)
            if self._verbose:
                print('Dataset is shuffled')
        else:
            raise ValueError('Have not complete an entire epoch yet, cannot shuffle dataset')

    def next_batch(self):
        '''
        Returns:
            image_batch: RGB images, ndarray, shape=[batch_size,img_H,img_W,3]
            label_batch: sampled point pairs, ndarray, shape=[batch_size,n_pairs,5],
                        each image has n_pairs(may be different for different 
                        images), 5 is (y1,x1,y2,x2,relations)   
        '''
        # obtain current image and label batch
        image_batch = [None]*self._batch_size
        label_batch = [None]*self._batch_size
        for i in range(self._batch_size): 
            if INTENSE_VERBOSE:
                print('read head={}'.format(self._read_head))
            # idx determine index of data fetched
            idx = self._index[self._read_head]
            # fetch data
            img = misc.imread(self._dataset['image_paths'][idx])
            if len(img.shape)==3:
                label = self._dataset['labels'][idx]
            else: # to avoid grayscale image in the shitty dataset
                while(len(img.shape)!=3):
                    rescue_idx = random.randint(0,self._n_samples)
                    img = misc.imread(self._dataset['image_paths'][rescue_idx])
                    label = self._dataset['labels'][rescue_idx]
            # data preprocessing
            img = misc.imresize(img, (self._train_img_size,self._train_img_size))
            img = img.astype(np.float32)
            img = img/127.5 - 1.
            # add to batch
            image_batch[i] = img.copy()
            label_batch[i] = label
            # update read head
            self._read_head += 1
        # convert to numpy data structure
        image_batch = np.stack(image_batch)
        label_batch = np.stack(label_batch)
        label_batch = label_batch.astype(np.int32)

        # intense verbose
        if INTENSE_VERBOSE:
            print('image_batch_shape:{}, label_batch_shape:{}'\
                  .format(image_batch.shape, label_batch.shape))

        # precompute the next read head, to check if an epoch completes(the last possible batch) after current batch ends
        # now, self._read_head is the read head of next batch, and we check the read head of the batch after next batch
        if self._read_head+self._batch_size > self._n_samples:
            # an epoch completed, reset read head to 0
            if self._verbose:
                print('An epoch is done!')
            self._read_head = 0
            if self._do_shuffle:
                self.shuffle()
        
        return image_batch, label_batch

    def _parse_dir(self):
        '''
        FUNC: parse directory with structure of root directory containing
              several subdirectories with each subdirectory as one class
        '''
        self._n_samples = 0
        # get subdirectories
        subdirs = os.listdir(self._dataset_dir)
        # loop through all subdirectories
        dataset_imgpaths = []
        dataset_labels = []
        for i, cur_class in enumerate(subdirs):
            # get full path of current subdirectory
            cur_class_path = os.path.join(self._dataset_dir, cur_class)
            if os.path.isdir(cur_class_path): # eliminate directory . and ..
                # list all images' names in current subdirectory, not in full path yet
                images_path = os.listdir(cur_class_path)
                # append images_path as full paths
                images_path = [os.path.join(cur_class_path, img) for img in images_path]
                # append to dataset list
                dataset_imgpaths = dataset_imgpaths + images_path
                dataset_labels = dataset_labels + [i]*len(images_path)
                # update n_samples
                self._n_samples += len(images_path)
                # store labels' names
                self._label_names.append(cur_class)
        self._dataset = {'image_paths': dataset_imgpaths,
                         'labels': dataset_labels}
    
    # class property, protected
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def n_samples(self):
        return self._n_samples
    @property
    def label_names(self):
        return self._label_names
    def __str__(self):
       return 'Hi, I am an instance of indoor_input <3' 

