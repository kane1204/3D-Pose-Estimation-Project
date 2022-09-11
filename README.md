# 3D Pose Estimation Project
This is the first part of my masters project
## Abstract
3D pose estimation has been a very popular part of computer vision for a long time
now. The main challenge these days is to find data that can be used to train a 3D pose
estimation algorithm since it requires 3 dimensional data. Our project addresses this
problem by using a synthetic dataset based on the Sungaya stick insect. The dataset
used is artificially generated to provide photo realistic samples. These samples are used
to train our 3D pose estimation pipeline which consists of 2 deep neural networks. The
first of these networks detects the 2D pose of a stick insect and the second network
detects the 3D pose of the stick insect based of the 2D pose data from the previous
network. The project also investigates how well these networks trained on synthetic data
and carries out experiments on both models to determine how well the models are able
to perform when faced with real data. Using the results provided by our investigations
we determine that this pipeline is a good step into creating a 3D pose estimator for a
stick insect which is able to detect 3D pose only trained on synthetic data. 
