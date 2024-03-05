# Weighted Entropic Associative Memory (W-EAM) using a single register to store real world images

This repository contains the procedures to replicate the experiments presented in the paper:

&nbsp;&nbsp;&nbsp;Noé Hernández, Rafael Morales & Luis A. Pineda (under review). _Entropic Associative Memory on real world images_.

The retrieved images are available in the folder [runs-1024/images/chosen-images-grid](https://github.com/nohernan/W-EAM_Cifar10/blob/main/runs-1024/images/chosen-images-grid)

The code was written in Python 3, using the Anaconda Distribution, and was run on a desktop computer with the following specifications:
* CPU: Intel Core i7-6700 at 3.40 GHz
* GPU: Nvidia GeForce GTX TITAN X
* OS: Ubuntu 16.04 Xenial
* RAM: 64GB

To clone the Anaconda environment [weam_cifar10.yml](https://github.com/nohernan/W-EAM_Cifar10/blob/main/weam_cifar10.yml) used in the the experiments follow the instruction ``$ conda env create -f weam_cifar10.yml``.

### Use

The script ``run_first.sh`` trains the autoencoder and classifier, obtains the features of all data and runs experiment 1 of the paper for the memory sizes ``m``: 64, 128, 256, 512 and 1024. The system saves output files in the corresponding ``runs-m`` folder. The file ``mem_params.csv`` with the values of the parameters _iota_, _kappa_, _xi_, and _sigma_ must exists in such ``runs-m`` folders before the execution.

The code in ``mcols_stdev.py`` computes the mean and standard deviation of the precision, the recall and the entropy of the memories for all number of columns and rows according to experiment 1. We analyse these metrics to determine the optimal memory size: 1024x16.

We classify the test data added with noise and choose random images of each class running the code in ``noised_classif.py`` and ``choose.py``, respectively. The output of ``choose.py`` is ``chosen.csv`` containing the ids of the selected images. The ``chosen.csv`` file is copied to the ``runs-1024`` folder.

Experiments 2 through 5 are run with the script ``run_second.sh``, where the variable ``n`` is first set to 1024. Next, the classification of the retrieved images in experiments 2 and 3, and in experiments 4 and 5 is carried out by ``classif.py`` and ``classif_dreams.py``, respectively. 

The script ``select_imgs.sh`` arranges all retrieved images and their classes in the grids shown in Figs. 8-11 of the paper. The script is run as:

```./select_imgs.sh runs-1024/chosen.csv```

Consider the comments in all previous files before executing them to appropriately set all variables and comment/uncomment lines of code if necessary.

The source code also includes:
* ``associative.py`` implements AMRs and memory operations.
* ``constants.py`` defines values for the operation of the system and functions for file management.
* ``dataset.py`` obtains and manipulates the images of Cifar10 adding noise and partitioning the data set.
* ``eam.py`` controls the execution of the system as a whole, carries out the experiments described in the paper, performs quantization and its inverse, computes the memory performance and generates the corresponding graphs.
* ``neural_net.py`` defines and trains the autoencoder and classifier, and extracts features from data.
* ``parse_history.py`` computes for each domain size the _accuracy_ and _decoder\_root\_mean\_squared\_error_ of the classifier and the autoencoder, respectively, on the testing data.
* ``nnet_stats.py`` computes _accuracy_ of the classifier for each domain size.
* ``check_chosen.py`` checks the class of the chosen images.
* ``system_stats.py`` computes the mean value of the precision and the recall for the selected memory with original and noisy cues using the _sigma_ values: 0.03, 0.05, 0.07, 0.09 and 0.11.
