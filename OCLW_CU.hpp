#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#endif

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

#ifndef OCLW_CU_H
#define OCLW_CU_H

#include <CL/opencl.hpp>
#include <unistd.h>
#include <map>
#include <string>

#include "OCLW_Kernel.hpp"

namespace OpenCLWrapper
{
  class ComputeUnit
  {
    public:
      //-- Constructors --//
      ComputeUnit();
      ComputeUnit(const cl::Device& device, uint index, double fraction, bool last, bool verbose = true);

      //-- Verbose --//
      void setVerbose(bool verbose);
      bool isVerbose() const;

      //-- Profiling --//
      void setProfiling(bool enabled);
      bool isProfiling() const;
      void printProfilingResults() const;
      void resetProfilingResults();

      //-- Source management --//
      void addSource(const std::string& sourceCode);

      //-- Kernel management --//
      void addKernel(const std::string& kernelName, ulong nElements, ulong offset = 0);
      void addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset = 0);
      void addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset, ulong localWorkSize);
      void clearKernels();

      //-- Buffer management --//
      template<class T> void allocateBuffer(const std::string& name, ulong size);

      template<class T> void writeBuffer(const std::vector<T>& hBuffer);
      template<class T> void writeBuffer(const std::string& name, const std::vector<T>& hBuffer, ulong dBufferWriteOffset);

      template<class T> void readBuffer(std::vector<T>& hBuffer);
      template<class T> void readBuffer(const std::string& name, std::vector<T>& hBuffer, ulong dBufferReadOffset);

      //-- Kernel arguments --//
      template<class T> void addArgument(const std::string& kernelName, const std::string& bufferName);
      template<class T> void addArgument(const std::string& kernelName, const std::vector<T>& hBuffer);
      template<class T> void addArgument(const std::string& kernelName, const T& variable);

      //-- Execution --//
      void run();
      void waitFinish();

    private:
      //-- Build helpers --//
      void buildContext();
      void buildQueue();
      void buildProgram();

      //-- Configuration --//
      uint index;
      double fraction;
      bool last;
      bool programBuilt = false;
      bool verbose = true;
      bool profiling = false;

      //-- Profiling data --//
      std::map<std::string, double> kernelTotalTime;   // accumulated ms per kernel name
      std::map<std::string, ulong> kernelCallCount;    // call count per kernel name

      //-- OpenCL state --//
      cl::Device device;
      cl::Context context;
      cl::CommandQueue queue;
      cl::Program program;
      std::vector<Kernel> kernels;

      //-- Source storage --//
      cl::Program::Sources sources;
      std::vector<std::string> sourceStrings;  // Store actual strings to keep pointers valid

      //-- Buffer maps --//
      std::map<void*, cl::Buffer> hdBufferMap;
      std::map<std::string, cl::Buffer> dBufferMap;
  };

  template<class T> void ComputeUnit::allocateBuffer(const std::string& name, ulong size)
  {
    std::map<std::string, cl::Buffer>::const_iterator it = this->dBufferMap.find(name);
    cl::Buffer dBuffer;

    if(it != this->dBufferMap.end())
      dBuffer = it->second;

    else {
      dBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(T) * size);
      this->dBufferMap[name] = dBuffer;
    }
  }

  template<class T> void ComputeUnit::writeBuffer(const std::vector<T> &hBuffer)
  {
    void* hBufferKey = static_cast<void*>(const_cast<std::vector<T>*>(&hBuffer));
    std::map<void*, cl::Buffer>::const_iterator it = this->hdBufferMap.find(hBufferKey);
    cl::Buffer dBuffer;

    if(it != this->hdBufferMap.end())
      dBuffer = it->second;

    else {
      dBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(T) * hBuffer.size());
      this->hdBufferMap[hBufferKey] = dBuffer;
    }

    this->queue.enqueueWriteBuffer(dBuffer, CL_FALSE, 0, sizeof(T) * hBuffer.size(), hBuffer.data());
  }

  template<class T> void ComputeUnit::writeBuffer(const std::string& name, const std::vector<T>& hBuffer, ulong dBufferWriteOffset)
  {
    std::map<std::string, cl::Buffer>::const_iterator it = this->dBufferMap.find(name);
    cl::Buffer dBuffer;

    if(it != this->dBufferMap.end())
      dBuffer = it->second;

    else {
      dBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(T) * hBuffer.size());
      this->dBufferMap[name] = dBuffer;
    }

    this->queue.enqueueWriteBuffer(dBuffer, CL_FALSE, sizeof(T) * dBufferWriteOffset, sizeof(T) * hBuffer.size(), hBuffer.data());
  }

  template<class T> void ComputeUnit::readBuffer(std::vector<T>& hBuffer)
  {
    void* hBufferKey = static_cast<void*>(const_cast<std::vector<T>*>(&hBuffer));
    cl::Buffer dBuffer = this->hdBufferMap[hBufferKey];

    uint count = hBuffer.size() * this->fraction;
    uint bufferOffset = this->index * count;  // Same offset for both device read and host write

    if(this->last)
      count = hBuffer.size() - bufferOffset;

    this->queue.enqueueReadBuffer(dBuffer, CL_TRUE, sizeof(T) * bufferOffset, sizeof(T) * count, hBuffer.data() + bufferOffset);
  }

  template<class T> void ComputeUnit::readBuffer(const std::string& name, std::vector<T>& hBuffer, ulong dBufferReadOffset)
  {
    cl::Buffer dBuffer = this->dBufferMap[name];

    uint count = hBuffer.size() * this->fraction;
    uint hBufferWriteOffset = this->index * count;
    uint deviceReadOffset = hBufferWriteOffset + dBufferReadOffset;

    if(this->last)
      count = hBuffer.size() - hBufferWriteOffset;

    this->queue.enqueueReadBuffer(dBuffer, CL_TRUE, sizeof(T) * deviceReadOffset, sizeof(T) * count, hBuffer.data() + hBufferWriteOffset);
  }

  template<class T> void ComputeUnit::addArgument(const std::string& kernelName, const std::string& bufferName)
  {
    cl::Buffer dBuffer = this->dBufferMap[bufferName];

    for(std::vector<Kernel>::iterator it = this->kernels.begin(); it != this->kernels.end(); it++) {
      if(it->name.compare(kernelName) == 0) {
        it->kernel.setArg(it->argsCount, dBuffer);
        it->argsCount++;
      }
    }
  }

  template<class T> void ComputeUnit::addArgument(const std::string& kernelName, const std::vector<T>& hBuffer)
  {
    void* hBufferKey = static_cast<void*>(const_cast<std::vector<T>*>(&hBuffer));
    cl::Buffer dBuffer = this->hdBufferMap[hBufferKey];

    for(std::vector<Kernel>::iterator it = this->kernels.begin(); it != this->kernels.end(); it++) {
      if(it->name.compare(kernelName) == 0) {
        it->kernel.setArg(it->argsCount, dBuffer);
        it->argsCount++;
      }
    }
  }

  template<class T> void ComputeUnit::addArgument(const std::string& kernelName, const T& variable)
  {
    for(std::vector<Kernel>::iterator it = this->kernels.begin(); it != this->kernels.end(); it++) {
      if(it->name.compare(kernelName) == 0) {
        it->kernel.setArg(it->argsCount, variable);
        it->argsCount++;
      }
    }
  }
}

#endif // OCLW_CU_H
