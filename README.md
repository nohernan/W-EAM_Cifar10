# Weighted Entropic Associative Memory (W-EAM) using a single register storing real world images

This repository contains the procedures to replicate the experiments presented in the paper:

&nbsp;&nbsp;&nbsp;No\'e Hern\'andez, Rafael Morales & Luis A. Pineda (under review). _Entropic Associative Memory on real world images_.

The retrieved images are available in the folder [runs-1024/images/chosen-images-grid](https://github.com/nohernan/W-EAM_Cifar10/blob/main/runs-1024/images/chosen-images-grid)

The code was written in Python 3, using the Anaconda Distribution, and was run on a desktop computer with the following specifications:
* CPU: Intel Core i7-6700 at 3.40 GHz
* GPU: Nvidia GeForce GTX TITAN X
* OS: Ubuntu 16.04 Xenial
* RAM: 64GB

To clone the Anaconda environment [weam_cifar10.yml](https://github.com/nohernan/W-EAM_Cifar10/blob/main/weam_cifar10.yml) used in the the experiments follow the instruction ``$ conda env create -f weam_cifar10.yml``.

### Use

The script ``run_first.sh`` trains the autoencoder and classifier, obtains the features of the data in the corpus and runs experiment 1 of the paper for the different memory sizes (64, 128, 256, 512 and 1024). Each execution saves data in the corresponding ``runs`` folder; for example, ``runs-1024``, such that the values of the ``iota``, ``kappa``, ``xi``, and ``sigma`` parameters have to be set beforehand in the file ``mem_params.csv`` within each ``runs`` folder.

The code in ``mcols_stdev.py`` computes the mean and standard deviation of the precision, recall and entropy values of the memories for all number of columns and rows, which we analyse to determine the optimal memory size.

In order to run the reamining experiments, we need to classify the noisy test data and choose from the Test Corpus images of each class that are correctly classified. The files ``noised_classif.py`` and ``choose.py`` accomplished these tasks, respectivelty, provided the selected memory size. The output of ``choose.py`` is ``chosen.csv`` with the images id contituing the grid shown in Figs. 8-11 of the paper, also ``chosen.csv`` must be saved within the ``runs`` folder of the selected memory size.

Experiments 2 through 5 are run with the script ``run_second.sh``, but first the variable ``n`` is assigned to the selected memory domain.

