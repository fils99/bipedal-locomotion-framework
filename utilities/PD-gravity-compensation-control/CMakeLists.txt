if(FRAMEWORK_COMPILE_PDGravityCompensationControlApplication)

  install(FILES blf-PD-gravity-compensation-control.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION "${CMAKE_INSTALL_BINDIR}")

  install_ini_files(${CMAKE_CURRENT_SOURCE_DIR}/config)
  # debug print of the config files path
  message(STATUS "PD-gravity-compensation-control config files installed to: ${CMAKE_INSTALL_DATADIR}/PD-gravity-compensation-control/config")

endif()