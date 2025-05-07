# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import constants
import tensorflow as tf

np.random.seed(constants.seed_value)
tf.random.set_seed(constants.seed_value)  # tf cpu fix seed
tf.keras.utils.set_random_seed(constants.seed_value)
tf.config.experimental.enable_op_determinism()

columns = 32
rows = 32
sq_patch_size=22

def get_training(fold):
    return _get_segment(_TRAINING_SEGMENT, fold)

def get_filling(fold):
    return _get_segment(_FILLING_SEGMENT, fold)

def get_testing(fold, noised = False, partial = False):
    return _get_segment(_TESTING_SEGMENT, fold, noised, partial)

def _get_segment(segment, fold, noised = False, partial = False):
    if (_get_segment.data is None) \
       or (_get_segment.noised is None and noised) \
       or (_get_segment.partial is None and partial) \
       or (_get_segment.labels is None):
        _get_segment.data, _get_segment.noised, _get_segment.partial, _get_segment.labels = \
            _load_dataset(noised, partial)
    print('Delimiting segment of data.')
    total = len(_get_segment.labels)
    training = int(total*constants.nn_training_percent)
    filling = int(total*constants.am_filling_percent)
    testing = int(total*constants.am_testing_percent)
    step = total / constants.n_folds
    i = int(fold * step)
    j = (i + training) % total
    k = (j + filling)  % total
    l = (k + testing)  % total
    n, m = None, None
    if segment == _TRAINING_SEGMENT:
        n, m = i, j
    elif segment == _FILLING_SEGMENT:
        n, m = j, k
    elif segment == _TESTING_SEGMENT:
        n, m = k, l

    data = constants.get_data_in_range(_get_segment.noised, n, m) \
            if noised \
               else constants.get_data_in_range(_get_segment.partial, n, m) \
                    if partial \
                       else constants.get_data_in_range(_get_segment.data, n, m)
    labels = constants.get_data_in_range(_get_segment.labels, n, m)
    return data, labels

_get_segment.data = None
_get_segment.noised = None
_get_segment.labels = None
_get_segment.partial = None

_TRAINING_SEGMENT = 0
_FILLING_SEGMENT = 1
_TESTING_SEGMENT = 2

def _load_dataset(noised, partial):
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    data = np.concatenate((train_images, test_images), axis=0)
    data = data.astype('float32') / 255.0
    
    labels = np.concatenate((train_labels, test_labels), axis= 0)
    labels = labels.flatten()

    noised_data, partial_data = None, None

    if noised or partial:
        scale = constants.noise_scale
        noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
        noised_data = data + noise
        noised_data = np.clip(noised_data, 0.0, 1.0)

        lower_patch_val=(columns-sq_patch_size)//2
        higher_patch_val=columns-lower_patch_val
        # ===> Black patch
        mask = np.ones(data.shape, dtype="float32")
        mask[:,lower_patch_val:higher_patch_val,lower_patch_val:higher_patch_val,:]=0.
        partial_data = data*mask
        # ===> Pink/turquoise patch
        # partial_data = np.copy(data)
        # for i in range(sq_patch_size):
        #     diag = lower_patch_val+i
        #     partial_data[:,diag:higher_patch_val,diag,:] = np.array([245.,0.,135.])/255.
        #     partial_data[:,lower_patch_val:diag,diag,:] = np.array([8.,232.,222.])/255.
    
    return data, noised_data, partial_data, labels

