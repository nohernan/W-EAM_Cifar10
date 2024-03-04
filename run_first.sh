#!/bin/bash

for n in 64 128 256 512 1024; do
    printf 'Train autoencoder with domain %d next ...\n\n' "$n" &&
    	python eam.py -n --domain=$n --runpath=runs-$n &&
    	printf '\n\nTrain classifier with domain %d next ...\n\n' "$n" &&
    	python eam.py -c --domain=$n --runpath=runs-$n &&
	printf '\n\nObtain features with domain %d next ...\n\n' "$n" &&
    	python eam.py -f --domain=$n --runpath=runs-$n &&
	printf '\n\nRun experiment one with domain %d next ...\n\n' "$n" &&
    	python eam.py -e 1 --domain=$n --runpath=runs-$n
done
