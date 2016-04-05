cmake_minimum_required(VERSION 2.8.3)
project(aman_sandbox)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  #roscpp
  #rospy
  #std_msgs
  #sensor_msgs
  #visualization_msgs
  #tf2
)

#find_package(Eigen REQUIRED)
#find_package(PCL REQUIRED COMPONENTS common io)
#find_package(OpenCV REQUIRED)
#find_package(Boost REQUIRED)
#find_package(NUMPY REQUIRED)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES aman_sandbox
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

#include_directories(
#  include
#  ${catkin_INCLUDE_DIRS}
#)

#target_link_libraries(${PY_PROJECT_NAME}
#    ${catkin_LIBRARIES}
#    ${Boost_LIBRARIES}
#)
