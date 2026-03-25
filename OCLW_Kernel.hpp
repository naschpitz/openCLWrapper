#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#endif

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

#ifndef OCLW_KERNEL_H
#define OCLW_KERNEL_H

#include <CL/opencl.hpp>

namespace OpenCLWrapper
{
  struct Kernel {
      std::string name;
      cl_ulong nElementsX; // global size dim 0
      cl_ulong nElementsY = 1; // global size dim 1 (1 = effectively 1D)
      cl_ulong offset = 0; // offset in dim 0 (kept for fraction splitting)
      cl::Kernel kernel;
      cl_ulong argsCount = 0;
      cl_ulong localWorkSizeX = 0; // 0 = let OpenCL decide
      cl_ulong localWorkSizeY = 0; // 0 = let OpenCL decide
  };
}

#endif // OCLW_KERNEL_H
