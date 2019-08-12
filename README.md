# Sparse Aerial Depth Completion

## Description

This project aims to provide a model for predicting depth from aerial images with (potentially noisy) sparse depth. This repository contains code to generate datasets from arbitrary paired satellite image and DSM datasets, to train several depth completion models, and to evalute this models. In addition, we've included non-machine learning baselines for comparison.

## Organization

This repository is organized to be as modular as possible, so that you can easily add additional models, change training procedures, or change hyperparameters without radically altering existing code. 

The main folders are:

- data: This folder contains the code for generating data and for loading data while training.
- models: This folder contains the baseline, non-ML models as well as the architecture for our neural networks.
- trainers: This folder contains the basic training procedure, base_trainer.py, as well as subclasses for particular models and loss functions. 
- configs: This folder contains .yml files that outline the hyperparameter settings for training code.

These modules are directed by main.py, the entry point for training our neural networks. 

## Suggested Data

## Getting Started

### Install Pre-requisites 

### Training

### Testing

### Debugging


