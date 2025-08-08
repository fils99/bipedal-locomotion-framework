# cmake_minimum_required(VERSION 3.10)

# # Project name
# project(PD-gravity-compensation-control)

# # Set Python module paths
# find_package(Python3 COMPONENTS Interpreter)

# yarp_configure_external_installation(PD-gravity-compensation-control)

# # Define the installation directories
# set(INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin")

# # Add the Python script to the installation list
# install(PROGRAMS ${PROJECT_SOURCE_DIR}/PD-gravity-compensation-control.py
#         DESTINATION ${INSTALL_BIN_DIR})

# # Optionally, create an uninstall file (for future removal of installed files)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/uninstall.cmake.in
#                ${CMAKE_BINARY_DIR}/uninstall.cmake @ONLY)

# # Install configuration files
# install(DIRECTORY config/robots DESTINATION ${CMAKE_INSTALL_DATADIR}/PD-gravity-compensation-control)



if(FRAMEWORK_COMPILE_PDGravityCompensationControlApplication)

  install(FILES blf-PD-gravity-compensation-control.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION "${CMAKE_INSTALL_BINDIR}")

  install_ini_files(${CMAKE_CURRENT_SOURCE_DIR}/config)
  # debug print of the config files path
  message(STATUS "PD-gravity-compensation-control config files installed to: ${CMAKE_INSTALL_DATADIR}/PD-gravity-compensation-control/config")

endif()