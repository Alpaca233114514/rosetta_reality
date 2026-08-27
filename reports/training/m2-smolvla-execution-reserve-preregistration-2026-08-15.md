# M2 SmolVLA execution-reserve preregistration (2026-08-15)

## Outcome

Plan `028` is preregistered as the next train-only exact diagnostic. It uses
standard robust constraint tightening: the physical joint-limit safety margin
remains `0.01` rad, while the command-feasible set is eroded by the
`0.03540462255477905`-rad maximum same-direction tracking overshoot measured
across every arm joint and executed step in Plan `027`. The resulting uniform
command margin is `0.04540462255477905` rad.

The frozen plan is
`configs/sim/aloha_insertion_geometry_teacher_028.yaml`, SHA-256
`536e882ec2222b67dcf7c81006f3b4860190823e0843ccc6b5fe6f8285b39899`.
Its source audit and exact report SHA-256 values are respectively
`610273b3fdacb9a4b38a563751a2f183846be531bf9b1c5faddb48a447f8d955`
and `7dd47ab9eada44ba8ca1105050ba6af784259f39bd55eb079f7025fe53999b73`.

## Official architecture boundary

No new IK or path planner was introduced. The evaluator continues to use
upstream Mink `ConfigurationLimit` for QP joint-position inequalities and
official MoveIt `moveit_msgs/JointConstraint` path constraints for start,
goal, every returned trajectory waypoint and the bounded next waypoint. Plan
`028` changes only the margin supplied to those two registered components.
Mink documents `ConfigurationLimit` as a joint-position inequality, and MoveIt
documents `JointConstraint` as its single-DOF joint constraint:

- https://kevinzakka.github.io/mink/api/limits.html
- https://moveit.picknik.ai/main/api/html/classkinematic__constraints_1_1JointConstraint.html

The repository-specific work remains a thin, hash-bound constraint-tightening
policy around those official interfaces. Mink `1.2.0`, DAQP `0.8.7`, MoveIt
`2.5.9`, OMPL RRTConnect, LMA, the runtime binary and robot-description hashes
are unchanged.

## Fail-closed boundary

The Plan `027` diagnostics remain non-causal and continue measuring the
physical `0.01`-rad boundary. Plan `028` separately binds the causal guard to
the source audit, exact-report hash, reserve metric and official limit types.
Both Mink and MoveIt must receive physical margin plus reserve. A command or
post-step observation entering the physical band now fails stage acceptance,
even if task reward would otherwise pass.

This uniform bound is derived from one train-only exact trace and therefore is
not a generalization claim. Only train episode 2 / seed 10 exact may run next.
Tuning seed 1900, development seeds 2000--2004, collection seeds 3000--3004,
policy-Gate seeds 1000--1004, validation, hidden and recovery labels remain
sealed. Neither pose gate nor physical joint margin may be relaxed.

## Local verification

The evaluator SHA-256 is
`806ae93d357c041155c6f7f99b8e506a0b3939ae841219c3d9eeefedbc8c1068`;
the protocol-test SHA-256 is
`3bf981c0464d14eb858aad3349a7d3397fe0fb0d7d8f2a73dcf0444af2f049ce`.
The teacher, Mink adapter, MoveIt adapter and protected Gym adapter hashes did
not change.

Docker Desktop's CLI was invoked from WSL Bash after the relocated WSL setup
did not expose the `docker` command directly. The pinned Linux image ran with
networking disabled, a read-only repository mount and read-only root
filesystem. Ruff passed, the 14 protocol tests passed, and all 43 focused
geometry/Mink/MoveIt/Gym tests passed. An initial protocol run was blocked by
one transcribed `package.xml` hash with an extra trailing character; correcting
that identity field exposed no runtime-code failure.

No SSH, exact simulation, CUDA, optimizer, model/data download or label write
was used. The next remote action remains a create-only content-addressed
workspace followed by train-only exact after the user explicitly reopens SSH.
