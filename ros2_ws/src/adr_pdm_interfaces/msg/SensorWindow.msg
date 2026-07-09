# Multivariate time-series window the anomaly detector consumes.
# data is row-major (n_signals x n_timesteps), flattened.
std_msgs/Header header
uint16 n_signals
uint16 n_timesteps
float32[] data
