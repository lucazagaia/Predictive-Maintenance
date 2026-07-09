cmake_minimum_required(VERSION 3.8)
project(adr_pdm_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotState.msg"
  "msg/SensorWindow.msg"
  "msg/Detection.msg"
  "msg/Prediction.msg"
  "msg/Decision.msg"
  DEPENDENCIES std_msgs
)

ament_package()
