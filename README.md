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

| Scene name  | Number of transformation parameters | Number of configuration parameters | Link to dataset |
| :---        |     :---:                   |          :---:             | :---: |
| box         | 1                           | 0                          | [box.tar.gz](https://ipvs.informatik.uni-stuttgart.de/mlr/peter/kmn/box.tar.gz) |
| box_xy   | 2                           | 0                          | [box_xy.tar.gz](https://ipvs.informatik.uni-stuttgart.de/mlr/peter/kmn/box_xy.tar.gz) |
| box_trans   | 3                           | 0                          | [box_trans.tar.gz](https://ipvs.informatik.uni-stuttgart.de/mlr/peter/kmn/box_trans.tar.gz) |
| box_complex | 5                           | 0                          | [box_complex.tar.gz](https://ipvs.informatik.uni-stuttgart.de/mlr/peter/kmn/box_complex.tar.gz) |
| door        | 3                           | 4                          | [door.tar.gz](https://ipvs.informatik.uni-stuttgart.de/mlr/peter/kmn/door.tar.gz) |

## kmn/models
This folder contains different convolutional neural network parametrizations.

## Getting started
```
git clone https://github.com/etpr/kinematic_morphing_network.git
cd kinematic_morphing_network/kmn/
python3 setup.py install --user
wget https://ipvs.informatik.uni-stuttgart.de/mlr/peter/kmn/box.tar.gz
tar xzf box.tar.gz
cd ..
ipython3 nbconvert kmn/notebooks/run_experiment.ipynb --to python
ipython3 run_experiment.py
```
