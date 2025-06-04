# This software may be modified and distributed under the terms of the
# BSD-3-Clause license.
# Authors: Filippo Passerini

if(FRAMEWORK_COMPILE_JointTorqueTrackingApplication)

  install(FILES blf-joint-torque-tracking.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION "${CMAKE_INSTALL_BINDIR}")

  install_ini_files(${CMAKE_CURRENT_SOURCE_DIR}/config)

endif()
