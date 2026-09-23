# Vision-Based Imitation Learning for Robotic Manipulation with LeRobot and MuJoCo

Franka Panda visual imitation learning in robosuite/MuJoCo.

## Environment layout

- Simulation: `/share/$USER-local/envs/lerobot-mujoco-sim`
- Dataset and ACT training: `/share/$USER-local/envs/lerobot-mujoco-imitation`
- Large data, checkpoints, logs, and caches: `/share/$USER-local/` outside this repository

The environments are separate because robosuite 1.5.2 requires NumPy <2
through mink 0.0.5, while LeRobot 0.6.1 requires NumPy >=2.
