cmake_minimum_required(VERSION 3.5)
project(loc_odo)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io registration)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED) 

find_package(fast_gicp REQUIRED)
message(STATUS "fast_gicp_INCLUDE_DIRS=${fast_gicp_INCLUDE_DIRS}")
message(STATUS "fast_gicp_LIBRARIES=${fast_gicp_LIBRARIES}")

#find_package(port_msgs REQUIRED)         # 车辆状态
#find_package(chcnv_cgi_msgs REQUIRED)    # Devpvt

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(publisher src/bagPublisher.cpp)
ament_target_dependencies(publisher 
  rclcpp
  std_msgs
  sensor_msgs
  )

add_executable(subscriber src/bagSubscriber.cpp)
ament_target_dependencies(subscriber rclcpp std_msgs sensor_msgs)

add_executable(icp_node src/icp_node.cpp)
ament_target_dependencies(icp_node
  rclcpp
  std_msgs
  sensor_msgs
  pcl_conversions
  PCL
  pcl_ros
  tf2_ros
  geometry_msgs
  nav_msgs
  )

include_directories(${PCL_INCLUDE_DIRS})

if(DEFINED fast_gicp_INCLUDE_DIRS)
  target_include_directories(icp_node SYSTEM PUBLIC ${fast_gicp_INCLUDE_DIRS})
  target_include_directories(icp_node SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
endif()

target_link_libraries(icp_node
  ${PCL_LIBRARIES}
  ${fast_gicp_LIBRARIES}
  Eigen3::Eigen
)


install(TARGETS
  publisher 
  subscriber
  icp_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()