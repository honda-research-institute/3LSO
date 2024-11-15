#!/bin/bash
gnome-terminal -e
cd ~/CARLA_0.9.8
./CarlaUE4.sh

gnome-terminal -e
source ~/.bashrc
conda activate carla-env
source ~/carla-lane-change-setup/catkin_ws/devel/setup.bash
roslaunch demo_entrance scenario_loader_t_junction_demo.launch


gnome-terminal -e
source ~/.bashrc
conda activate carla-env
source ~/carla-lane-change-setup/catkin_ws/devel/setup.bash
python ~/NNMPC_CARLA/init_ros_node_junction.py


gnome-terminal -e
cd ~/scenario_runner
source ~/.bashrc
conda activate carla-env
source ~/carla-lane-change-setup/catkin_ws/devel/setup.bash
python scenario_runner.py --scenario HRIVehicleTurning_6Demo --waitForEgo --report_enable --output --repetitions 5