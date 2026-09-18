# ALOHA MoveIt 2 adapter

This ROS 2 package does not implement a motion planner. It loads the official
Trossen/Interbotix ALOHA VX300S description, the official MoveIt LMA kinematics
plugin, and MoveIt's OMPL `geometric::RRTConnect` planning pipeline. Rosetta owns
only these integration responsibilities:

- compose the two vendor single-arm models into one collision scene;
- map Gym-ALOHA joint and calibration-site frames into that scene;
- apply MoveIt-native joint path constraints and verify every returned waypoint
  against the registered execution margin;
- expose a bounded JSON-lines process protocol to the simulator evaluator; and
- reject non-finite, out-of-bounds, colliding, or unverifiable solutions.

The generated URDF, SRDF, and identity manifest are created during the pinned
Docker build and installed under this package's `config/` directory.
