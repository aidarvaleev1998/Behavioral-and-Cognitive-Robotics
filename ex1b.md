## PendulumEnv
1. Observation vector is a tuple of three values representing the angle and the angular velocity: (cos(θ), sin(θ), dθ).
2. Action vector is a float number representing the torque (bounded from -2.0 to 2.0)
3. Reward is calculated as a negative weighted sum of squared angle, velocity and torque, so we want to minimize them. 
   0 angle corresponds to upwards direction.
4. The reset method generates the angle (-pi to pi) and angular velocity (-1, 1) from the uniform distribution.
5. Termination condition is always False.

## AcrobotEnv
1. Observation vector is a tuple of six values representing the state of the system - the angles of the two links 
   (second is relative to the first) and the angular velocities of the links. 
   So, observation = (cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2).
2. Action vector is an integer from 0 to 2 that match to the -1 to 1 torques, 
   which then used to calculate new state by integral approximation.
3. Reward is -1 if termination conditions are not met and 0 otherwise.
4. The reset method generates angles and angular velocities (in radians) from the (-0.1, 0.1) uniform distribution.
5. Termination condition is whether the far end of the bot is one length of the link above the center.