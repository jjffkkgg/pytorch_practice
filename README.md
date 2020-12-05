## ROS-pytorch control repository [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jjffkkgg/ros-torch-control/master) [<img src="https://jupyter.org/assets/main-logo.svg" height="20" title="JupyterLab">](https://mybinder.org/v2/gh/jjffkkgg/ros-torch-control/master?urlpath=lab) [![nbviewer](https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg)](http://nbviewer.jupyter.org/github/jjffkkgg/ros-torch-control/tree/master)

This repository is available for use of pytorch with following libraries.

## Local Installation

* Install anaconda.
* Use enviornment.yml to setup a new conda environment.

```bash
$ conda env create -f environment.yml
$ conda activate pyt
$ jupyter lab
```
***
## Python simulation

In test_system/quadrotor
* controlled_u
    * Training the quadcopter w/ transfer function controller.
    * System follows the guided trajectory with feedback loop input, the gain of the error is the learning objective

![reward_history](https://user-images.githubusercontent.com/49571274/100534754-b24e0a80-3255-11eb-9ad5-b1b1b3ba91fe.png)
![flight_data](https://user-images.githubusercontent.com/49571274/100534893-52f0fa00-3257-11eb-8539-7851c514a082.png)
![flight](https://user-images.githubusercontent.com/49571274/100534943-c6930700-3257-11eb-80b6-4a795c60a935.gif)

* free_u
    * Training without controller, free input
    * Similar to inverted pendulum learning, train to maintain hovering in turbulent outside forces

***
## ROS setup (noetic, Ubuntu 20.04 LTS) [Not Running]

Follow installation guide in ROS neotic.

To setup the catkin, use following command in terminal. (python(3)-catkin-tools not working)
```
$ pip3 install --user git+https://github.com/catkin/catkin_tools.git
$ catkin init
$ catkin build
$ . ./devel/setup.bash
$ roslaunch quadrotor_a2c quadrotor.launch
```