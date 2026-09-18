cmake_minimum_required(VERSION 3.8)
project(rosetta_aloha_moveit2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ompl REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(aloha_moveit_planner src/aloha_moveit_planner.cpp)
ament_target_dependencies(
  aloha_moveit_planner
  moveit_core
  moveit_msgs
  moveit_ros_planning
  rclcpp
)
target_include_directories(aloha_moveit_planner PRIVATE ${OMPL_INCLUDE_DIRS})
target_link_libraries(
  aloha_moveit_planner
  nlohmann_json::nlohmann_json
  ${OMPL_LIBRARIES}
)
target_compile_options(aloha_moveit_planner PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(
  aloha_moveit_planner
  PRIVATE
    ROSETTA_MOVEIT_VERSION="2.5.9"
    ROSETTA_OMPL_VERSION="1.7.0"
)

install(TARGETS aloha_moveit_planner DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
