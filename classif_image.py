import numpy as np
import constants

## Find the classification of the chosen (normal|noised) images for all
## sigma values defined in constants.sigma_values

## This is used in experiments 2 and 3, and figures 4 and 5, respectively.
## First you have to run: python eam.py -r --domain=$n --runpath=runs-$n

## Set domain
domain = 256
constants.domain = domain

## Set run path
dirname = f'runs-{domain}'
constants.run_path=dirname

## Experimental settings
prefix = constants.memory_parameters_prefix
filename = constants.csv_filename(prefix)
parameters = \
             np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
es = constants.ExperimentSettings(parameters)

#
label = 7
idx = 727
fold = 1

msize = 16 ## This is the memory size chosen to run experiment 2 and on

# Uncomment for regular input
#prefix = constants.classification_name(es)
# Uncomment for noised input
#prefix = constants.noised_classification_name(es)
# Uncomment for partial input
prefix = constants.partial_classification_name(es)
fname = constants.data_filename(prefix, es, fold)
classif = np.load(fname) # Labels of testing (normal|noised) data according to classifier
class_num = classif[idx] if classif[idx]!=10 else '-'
sigmas = constants.sigma_values
prefix += constants.int_suffix(msize, 'msz')

print(f'\t\t',end='')
labels = ''

for s in sigmas:
    suffix = constants.float_suffix(s, 'sgm')
    fname = prefix + suffix
    fname = constants.data_filename(fname, es, fold)
    classif = np.load(fname)
    class_num = classif[idx] if classif[idx]!=10 else '-'
    print(f'{s}',end='\t')
    labels +='\t'+str(class_num)
    
print(f'\n"{label}\t{class_num}', end='')
print(f'{labels}"')
    
