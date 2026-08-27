# M2 SmolVLA Athena Plan035 local exact audit

Status: **failed safely `0/1`; the static feedforward executed but was
insufficient; no fourth local exact is authorized**.

Plan035 completed all 500 train-only steps with one accepted official MoveIt
path. It advanced through all 28 subsequent waypoints, reused the retained
reference 402 times, activated terminal control once at step 153 and executed
347 compensated commands. The maximum correction was `0.0169158762352822` rad,
below the registered `0.05`-rad cap, and its minimum command margin was
`0.84494201442145` rad. There were no IK failures, clips, commanded or observed
margin breaches, unexpected collisions, validation/hidden access or labels.

The result did not meet the unchanged 12 mm approach gate. The final target
position error settled at `0.026163499802350998` m and the final joint-space L1
distance to the reference was `0.039295138703090406` rad. Relative to Plan032's
`0.02851971797645092` m error, static feedforward improved the residual by only
`0.00235621817409992` m (`8.26%`). The MoveIt goal itself remained accurate:
maximum goal position/orientation error was `1.7309e-05` m / `0.0004171` rad.

Therefore the new controller path is real and safe, but static equilibrium
compensation alone is not the missing closed-loop behavior. Exact has not
passed, so tuning, development, collection, policy-Gate, validation, hidden
and recovery labels remain sealed. This consumed the third permitted local
exact attempt; the next action is offline equilibrium/feedback diagnosis, not
another run.

Exact report SHA-256:
`2d1fa4e3afd3748a218c82220ea678972cc25a97514f9bacec6d71eca6a6174f`.
