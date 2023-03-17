# Machine-Learning-Technologies-on-EMG-for-Hand-Gesture-Recognition
Machine learning code implemented for hand gesture recognition using EMG data from the Ninapro db1 database 

## Overview

This is the code for my final year master's engineering project. This project uses a mixture of machine 
learning and deep learning classifiers to classify 15 different hand gestures from 27 intact subjects from 
the Ninapro DB1 database. The machine learning models implemented were linear discriminant analysis and a 
support vector machine and the deep learning models developed were a 1D and a 2D convolutional neural network, 
a long short-term memory network and a hybrid network architecture constructed from a combination of 
these two networks.

## Project Structure 

1. Functions.py - contains all the required functions to run the classifier model scripts and the Sort script

2. Sort.py - loads and segments the sEMG data from the Matlab files for each subject and produces three 
CSV files for each subject: training, validation and test (**code must be ammended with the accurate storage location
of Ninapro DB1 data for effective use**)


3. LDA.py - implements linear discriminant analysis classifier 

4. SVM.py - implements support vector machine classifier

5. 1DCNN.py - implements 1d convolutional neural network classifier

6. 2DCNN.py - implements 2d convoltuional neural network classifer

7. LSTM.py - implements long short-term memory network classifier

8. CNN_LSTM.py - impelemnts hybrid classifier consiting of both a 1d convolutional neural network and a long short-term memory network

## Author 

Louis Chapo-Saunders
