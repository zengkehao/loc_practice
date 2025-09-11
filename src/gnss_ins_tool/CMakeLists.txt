cmake_minimum_required(VERSION 3.5)
project(gnss_ins_tool)

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
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(chcnv_cgi_msgs REQUIRED)
#find_package(GeographicLib REQUIRED)


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

add_executable(gnss_ins_pose_node src/gnss_ins_pose_node.cpp)
ament_target_dependencies(gnss_ins_pose_node
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  chcnv_cgi_msgs
)

#target_link_libraries(gnss_ins_pose_node GeographicLib::GeographicLib)

find_path(GEOGRAPHICLIB_INCLUDE_DIR
  NAMES GeographicLib/UTMUPS.hpp
  PATHS /usr/include /usr/local/include
)
if(NOT GEOGRAPHICLIB_INCLUDE_DIR)
  message(FATAL_ERROR "GeographicLib headers not found (UTMUPS.hpp). Please install libgeographic-dev.")
endif()

find_library(GEOGRAPHICLIB_LIB
  NAMES Geographic GeographicLib            
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
)
if(NOT GEOGRAPHICLIB_LIB)
  message(FATAL_ERROR "GeographicLib library not found (libGeographic.so). Please install libgeographic-dev.")
endif()

target_include_directories(gnss_ins_pose_node PRIVATE ${GEOGRAPHICLIB_INCLUDE_DIR})
target_link_libraries(gnss_ins_pose_node ${GEOGRAPHICLIB_LIB})

install(TARGETS
  gnss_ins_pose_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
