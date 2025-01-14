# MR-SS-GAN: Manifold-Regularized Semi-Supervised GAN
This repository uses the Marmousi2 model data as an example to demonstrate the application of MR-SS-GAN in gas-bearing reservoir prediction. 

The input data are in the form of *.npy files, which contain the time-frequency data of the Marmousi2 model and have been processed with Synchrosqueezing Transform (SST). 

The specific implementation of the network architecture refers to the following two URLs:

https://github.com/UCI-ML-course-team/GAN-manifold-regularization-PyTorch

https://github.com/bruno-31/GAN-manifold-regularization

The output includes the categories of gas-bearing and non-gas-bearing reservoirs along with their corresponding probabilities.

#  Requirements
The repo supports python + pytorch

# Run the Code
python main.py
