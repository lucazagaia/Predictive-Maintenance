# RUL estimate + planning from the cloud prognostics node.
std_msgs/Header header
int32 rul_cycles
string urgency               # immediate | urgent | soon | planned
string maintenance_window
float32 confidence
