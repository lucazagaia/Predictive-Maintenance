# Real-time health status from the edge anomaly detector.
std_msgs/Header header
string status         # healthy | warning | anomaly
float32 confidence
float32 anomaly_score
string model
string details
