## ROS-pytorch control repository [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jjffkkgg/pytorch_practice/master) [<img src="https://jupyter.org/assets/main-logo.svg" height="20" title="JupyterLab">](https://mybinder.org/v2/gh/jjffkkgg/pytorch_practice/master?urlpath=lab) [![nbviewer](https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg)](http://nbviewer.jupyter.org/github/jjffkkgg/pytorch_practice/tree/master)

This repository is available for use of pytorch with following libraries.

## Local Installation

* Install anaconda.
* Use enviornment.yml to setup a new conda environment.

```bash
$ conda env create -f environment.yml
$ conda activate pyt
$ jupyter lab
```

## ROS setup (noetic, Ubuntu 20.04 LTS)

Follow installation guide in ROS neotic.

To setup the catkin, use following command in terminal. (python(3)-catkin-tools not working)
```
$ pip3 install --user git+https://github.com/catkin/catkin_tools.git
$ catkin init
$ catkin build
$ . ./devel/setup.bash
```