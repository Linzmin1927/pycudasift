if (CUDASIFT_WITH_LIB)
  cuda_add_library(cudasift ${cuda_sources_absolute} SHARED)
  install(TARGETS cudasift DESTINATION lib)
endif()
