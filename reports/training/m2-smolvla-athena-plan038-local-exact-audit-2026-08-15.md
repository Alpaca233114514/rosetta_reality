# Athena Plan038 local exact audit

Plan038 did not reach the newly registered 750-step horizon. Its first official MoveIt trajectory completed and triggered the registered terminal refresh, but the second request failed at step 223 because subgroup LMA IK returned no bimanual solution within the separate 0.10-second IK timeout.

The failure was not a safety escape: teacher failure, action clipping, unexpected collision, and commanded/observed joint-margin breaches were all zero. The pre-request minimum physical joint margin was 59.66 mrad. The Mink diagnostic errors of 7.77/7.74 mm identify the need for a new global path request; they do not relax the frozen 1/3 mm acceptance gates.

MoveIt passes the configured timeout into its IK solver. Plan039 may therefore change only `path_planner_ik_timeout_s` from 0.10 to 0.50 seconds. OMPL planning time, RRTConnect, LMA, every geometric and safety threshold, the 750-step horizon, and all seed/label seals remain unchanged. The JSON companion is the exact authority.
