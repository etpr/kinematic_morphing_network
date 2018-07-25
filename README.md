# Kinematic Morphing Network (KMN)

The code contains an implementation of KMNs, which are deep neural networks that map a depth image and point cloud to parameters of a kinematic model. The parameters are defined relative to a prototype model that serves as a reference to describe models. The concatenation property of affine transformations and the ability to convert point clouds to depth images are used to perform multi step predictions. The network is trained on data generated in a simulator and on augmented data that is created with its own predictions.

The package implements the training and prediction algorithms of KMNs, which can be found in kmn_model.py and kmn_model_conf.py. A detailed description of KMNs can be found in:
> P. Englert, M. Toussaint:
> [Kinematic Morphing Networks for Manipulation Skill Transfer](https://arxiv.org/pdf/1803.01777.pdf)
> In Proceedings of the International Conference on Intelligent Robotics Systems, 2018

## kmn/notebooks
This folder contains jupyter notebooks for running and evaluating experiments.

## kmn/scenes
The following scenes are implemented:

| Scene name  | Number of transformation parameters | Number of configuration parameters |
| :---        |     :---:                   |          :---:             |
| box         | 1                           | 0                          |
| box_trans   | 3                           | 0                          |
| box_complex | 5                           | 0                          |
| door        | 3                           | 4                          |

The package contains multiple configs and data samples for each scene. Feel free to contact me if you are interested in the full dataset.

## kmn/models
This folder contains different convolutional neural network parametrizations.
