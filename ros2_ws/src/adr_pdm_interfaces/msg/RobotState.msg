# Proprioceptive scalar reading from an ADR (autonomous delivery robot).
# header.frame_id carries the robot id (fleet namespace).
std_msgs/Header header
float32 temperature   # [C]
float32 vibration     # [g]
float32 torque        # [Nm]
float32 current       # [A]
