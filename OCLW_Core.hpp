#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#endif

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

#ifndef OCLW_CORE_H
#define OCLW_CORE_H

#include <CL/opencl.hpp>
#include <mutex>
#include <unistd.h>

#include "OCLW_CU.hpp"

namespace OpenCLWrapper
{
  class Core
  {
    public:
      //-- Constructors / Destructor --//
      Core(bool useMultipleGPUs = false);
      Core(int deviceIndex); // Constructor to use a specific GPU device by index
      ~Core();

      //-- Verbose --//
      void setVerbose(bool verbose);
      bool isVerbose() const;

      //-- Fast math --//
      void setFastMath(bool enabled);

      //-- Source management --//
      void addSourceFile(std::string fileName);

      //-- Kernel management --//
      void addKernel(const std::string& kernelName, ulong nElements, ulong offset = 0);
      void addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset = 0);
      void addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset,
                     ulong localWorkSize);
      void addKernel(const std::string& id, const std::string& kernelName, ulong nElementsX, ulong nElementsY,
                     ulong localWorkSizeX, ulong localWorkSizeY);
      void clearKernels();
      std::vector<std::vector<Kernel>> saveKernels();
      void restoreKernels(const std::vector<std::vector<Kernel>>& kernels);

      //-- Buffer management --//
      template <class T> void allocateBuffer(const std::string& name, ulong size);

      template <class T> void writeBuffer(const std::vector<T>& hBuffer);
      template <class T>
      void writeBuffer(const std::string& name, const std::vector<T>& hBuffer, ulong dBufferWriteOffset);

      template <class T> void readBuffer(std::vector<T>& hBuffer);
      template <class T> void readBuffer(const std::string& name, std::vector<T>& hBuffer, ulong dBufferReadOffset);

      template <class T> void fillBuffer(const std::string& name, const T& pattern, ulong count);

      template <class T> void syncDevicesBuffers(std::vector<T>& hBuffer);

      //-- Kernel arguments --//
      template <class T> void addArgument(const std::string& kernelName, const std::string& bufferName);
      template <class T> void addArgument(const std::string& kernelName, const std::vector<T>& hBuffer);
      template <class T> void addArgument(const std::string& kernelName, const T& variable);

      //-- Profiling --//
      void setProfiling(bool enabled);
      void printProfilingResults() const;
      std::vector<KernelTiming> getKernelTimings() const;
      void resetProfilingResults();

      //-- Execution --//
      void run();

      //-- Static methods --//
      static void initialize(bool verbose = true);
      static std::map<const cl::Device*, uint>& getDevicesUsage();
      static size_t getNumDevices();

    private:
      //-- Compute unit management --//
      void buildComputeUnits();
      void buildComputeUnitForDevice(int deviceIndex);

      //-- Device job tracking --//
      void addJobToDevice(const cl::Device& device);
      void removeJobFromDevice(const cl::Device& device);

      //-- Static initialization helpers --//
      static void buildPlatforms();
      static void buildDevices();
      static void buildDevicesUsageMap();
      static const cl::Device& getAvailableDevice();
      static void printDevicesInfo();

      //-- Instance members --//
      std::vector<ComputeUnit> computeUnits;
      cl::Program::Sources sources;
      bool useMultipleGPUs;
      std::vector<const cl::Device*> devicesInUse;

      //-- Static members --//
      static std::mutex mutex;
      static std::vector<cl::Platform> platforms;
      static std::vector<cl::Device> devices;
      static std::map<const cl::Device*, uint> devicesUsage;
      static bool verbose;
  };

  template <class T> void Core::allocateBuffer(const std::string& name, ulong size)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->allocateBuffer<T>(name, size);
    }

    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->waitFinish();
    }
  }

  template <class T> void Core::writeBuffer(const std::vector<T>& hBuffer)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->writeBuffer<T>(hBuffer);
    }

    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->waitFinish();
    }
  }

  template <class T> void Core::writeBuffer(const std::string& name, const std::vector<T>& hBuffer, ulong dBufferOffset)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->writeBuffer<T>(name, hBuffer, dBufferOffset);
    }

    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->waitFinish();
    }
  }

  template <class T> void Core::readBuffer(std::vector<T>& hBuffer)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->readBuffer<T>(hBuffer);
    }

    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->waitFinish();
    }
  }

  template <class T> void Core::readBuffer(const std::string& name, std::vector<T>& hBuffer, ulong dBufferOffset)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->readBuffer<T>(name, hBuffer, dBufferOffset);
    }

    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->waitFinish();
    }
  }

  template <class T> void Core::fillBuffer(const std::string& name, const T& pattern, ulong count)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->fillBuffer<T>(name, pattern, count);
    }

    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->waitFinish();
    }
  }

  template <class T> void Core::syncDevicesBuffers(std::vector<T>& hBuffer)
  {
    this->readBuffer<T>(hBuffer);
    this->writeBuffer<T>(hBuffer);
  }

  template <class T> void Core::addArgument(const std::string& kernelName, const std::string& bufferName)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->addArgument<T>(kernelName, bufferName);
    }
  }

  template <class T> void Core::addArgument(const std::string& kernelName, const std::vector<T>& hBuffer)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->addArgument(kernelName, hBuffer);
    }
  }

  template <class T> void Core::addArgument(const std::string& kernelName, const T& variable)
  {
    for (std::vector<ComputeUnit>::iterator it = this->computeUnits.begin(); it != this->computeUnits.end(); it++) {
      it->addArgument(kernelName, variable);
    }
  }
}

#endif // OCLW_CORE_H
