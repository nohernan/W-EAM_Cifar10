# Weighted Entropic Associative Memory (W-EAM) using a single register storing real world images

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

The script ``run_first.sh`` trains the autoencoder and classifier, obtains the features of all data and runs experiment 1 of the paper for the memory sizes: 64, 128, 256, 512 and 1024. Each execution saves output files in the corresponding ``runs`` folder; for example, ``runs-1024``, such that ``mem_params.csv`` with the values of the ``iota``, ``kappa``, ``xi``, and ``sigma`` parameters exists in each ``runs`` folder before the respective execution.

The code in ``mcols_stdev.py`` computes the mean and standard deviation of the precision, the recall and the entropy values of the memories for all number of columns and rows according to experiment 1. We analyse such values to determine the optimal memory size.

In order to run the reamining experiments, we need to classify the noisy test data and choose images of each class that are correctly classified. The files ``noised_classif.py`` and ``choose.py`` accomplished these tasks, respectivelty, provided with the selected memory size. The output of ``choose.py`` is ``chosen.csv`` containing the ids of the images shown in the paper, also ``chosen.csv`` must be saved within the ``runs`` folder of the selected memory size.

Experiments 2 through 5 are run with the script ``run_second.sh``, where the variable ``n`` is first set to the selected memory domain. The classification of the retreived images in experiments 2 and 3, and in experiments 4 and 5 is carried out by ``classif.py`` and ``classif_dreams.py``, respectively. Consider the comments in these two files before executing them to appropriately set all variables and comment/uncomment lines of code if necessary.

The classification of the retrieved images is added to ``select_imgs.sh``, which arranges all images and labels in the grids shown in Figs. 8-11 of the paper using the instruction:

```./select_imgs.sh runs-1024/chosen.csv```

The source code also includes:
* ``associative.py`` implements AMRs and memory operations.
* ``constants.py`` defines values for the operation of the system and functions for file management.
* ``dataset.py`` obtains and manipulates the images of Cifar10 adding noise and partitioning the data set.
* ``eam.py`` controls the execution of the system as a whole, carries out the experiments described in the paper, performs quantization and its inverse, computes the memory performance and generates the corresponding graphs.
*``neural_net.py`` defines and trains the autoencoder and classifier, and extracts features from data.
* ``parse_history.py`` computes for each domain size the _accuracy_ and _decoder\_root\_mean\_squared\_error_ of the classifier and the autoencoder, respectively, on the testing data.
*``nnet_stats.py`` computes _accuracy_ of the classifier for each domain size.
* ``check_chosen.py`` checks the class of the chosen images.
* ``system_stats.py`` computes the mean value of the precision and the recall for the selected memory with original and noisy cues using the sigma values: 0.03, 0.05, 0.07, 0.09 and 0.11.
