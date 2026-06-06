#include "OCLW_CU.hpp"

#include <iostream>

using namespace OpenCLWrapper;

//===================================================================================================================//
//-- Constructors --//
//===================================================================================================================//

ComputeUnit::ComputeUnit()
{
  this->programBuilt = false;
}

//===================================================================================================================//

ComputeUnit::ComputeUnit(const cl::Device& device, uint index, double fraction, bool last, bool verbose)
{
  this->programBuilt = false;
  this->verbose = verbose;

  this->device = device;
  this->index = index;
  this->fraction = fraction;
  this->last = last;

  this->buildContext();
  this->buildQueue();
}

//===================================================================================================================//
//-- Verbose --//
//===================================================================================================================//

void ComputeUnit::setVerbose(bool verbose)
{
  this->verbose = verbose;
}

//===================================================================================================================//

bool ComputeUnit::isVerbose() const
{
  return this->verbose;
}

//===================================================================================================================//
//-- Profiling --//
//===================================================================================================================//

void ComputeUnit::setProfiling(bool enabled)
{
  this->profiling = enabled;
  // Rebuild queue with profiling flag
  this->buildQueue();
}

bool ComputeUnit::isProfiling() const
{
  return this->profiling;
}

void ComputeUnit::printProfilingResults() const
{
  std::vector<KernelTiming> timings = this->getKernelTimings();

  if (timings.empty())
    return;

  std::cout << "\n=== OpenCL Kernel Profiling Results ===\n";

  double totalMs = 0;
  for (const auto& kt : timings)
    totalMs += kt.totalMs;

  for (const auto& kt : timings) {
    double avgMs = kt.callCount > 0 ? kt.totalMs / kt.callCount : 0;
    double pct = totalMs > 0 ? 100.0 * kt.totalMs / totalMs : 0;
    std::cout << "  " << kt.kernelName << ": " << kt.totalMs << " ms total, " << kt.callCount << " calls, " << avgMs
              << " ms/call, " << pct << "%\n";
  }

  std::cout << "  TOTAL: " << totalMs << " ms\n";
  std::cout << "==========================================\n";
  std::cout.flush();
}

std::vector<KernelTiming> ComputeUnit::getKernelTimings() const
{
  std::vector<KernelTiming> result;
  result.reserve(this->kernelTotalTime.size());

  for (const auto& pair : this->kernelTotalTime) {
    auto countIt = this->kernelCallCount.find(pair.first);
    ulong count = countIt != this->kernelCallCount.end() ? countIt->second : 0;
    result.push_back({pair.first, pair.second, count});
  }

  return result;
}

void ComputeUnit::resetProfilingResults()
{
  this->kernelTotalTime.clear();
  this->kernelCallCount.clear();
}

//===================================================================================================================//
//-- Source management --//
//===================================================================================================================//

void ComputeUnit::addSource(const std::string& sourceCode)
{
  // Store the string - sources will be rebuilt from sourceStrings in buildProgram()
  this->sourceStrings.push_back(sourceCode);
}

//===================================================================================================================//
//-- Kernel management --//
//===================================================================================================================//

void ComputeUnit::addKernel(const std::string& kernelName, ulong nElements, ulong offset)
{
  this->addKernel(kernelName, kernelName, nElements, offset);
}

//===================================================================================================================//

void ComputeUnit::addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset)
{
  if (!this->programBuilt)
    this->buildProgram();

  Kernel kernel;
  kernel.name = id; // Use id for argument lookup
  kernel.nElementsX = nElements;
  kernel.nElementsY = 1;
  kernel.offset = offset;

  if (this->verbose)
    std::cout << "Building kernel " << kernelName << " (id: " << id << ")...\n";

  cl_int result;
  kernel.kernel = cl::Kernel(this->program, kernelName.c_str(), &result); // Use kernelName for OpenCL lookup

  if (result != CL_SUCCESS) {
    std::cout << " Error creating kernel " << kernelName << ": " << result << "\n";
    exit(1);
  }

  this->kernels.push_back(kernel);

  if (this->verbose) {
    std::cout << " Done! (nElements=" << nElements << ")\n";
    std::cout.flush();
  }
}

//===================================================================================================================//

void ComputeUnit::addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset,
                            ulong localWorkSize)
{
  this->addKernel(id, kernelName, nElements, offset);

  // Set localWorkSize on the just-added kernel
  this->kernels.back().localWorkSizeX = localWorkSize;
}

//===================================================================================================================//

void ComputeUnit::addKernel(const std::string& id, const std::string& kernelName, ulong nElementsX, ulong nElementsY,
                            ulong localWorkSizeX, ulong localWorkSizeY)
{
  if (!this->programBuilt)
    this->buildProgram();

  Kernel kernel;
  kernel.name = id;
  kernel.nElementsX = nElementsX;
  kernel.nElementsY = nElementsY;
  kernel.offset = 0; // 2D kernels don't use fraction/offset splitting

  if (this->verbose)
    std::cout << "Building kernel " << kernelName << " (id: " << id << ")...\n";

  cl_int result;
  kernel.kernel = cl::Kernel(this->program, kernelName.c_str(), &result);

  if (result != CL_SUCCESS) {
    std::cout << " Error creating kernel " << kernelName << ": " << result << "\n";
    exit(1);
  }

  kernel.localWorkSizeX = localWorkSizeX;
  kernel.localWorkSizeY = localWorkSizeY;

  this->kernels.push_back(kernel);

  if (this->verbose) {
    std::cout << " Done! (nElementsX=" << nElementsX << ", nElementsY=" << nElementsY << ")\n";
    std::cout.flush();
  }
}

//===================================================================================================================//

std::vector<Kernel> ComputeUnit::saveKernels()
{
  return this->kernels;
}

//===================================================================================================================//

void ComputeUnit::restoreKernels(const std::vector<Kernel>& kernels)
{
  this->kernels = kernels;
}

//===================================================================================================================//

void ComputeUnit::clearKernels()
{
  this->kernels.clear();

  if (this->verbose) {
    std::cout << "Kernels cleaned\n";
    std::cout.flush();
  }
}

//===================================================================================================================//
//-- Execution --//
//===================================================================================================================//

void ComputeUnit::run()
{
  std::vector<cl::Event> events;

  if (this->profiling)
    events.resize(this->kernels.size());

  size_t idx = 0;
  for (std::vector<Kernel>::iterator it = this->kernels.begin(); it != this->kernels.end(); it++, idx++) {
    // Dim 0: fraction/offset splitting (existing behavior)
    uint countX = it->nElementsX * this->fraction;
    uint offsetX = this->index * countX + it->offset;

    if (this->last)
      countX = it->nElementsX - offsetX;

    // Dim 1: always full range (no splitting)
    uint countY = it->nElementsY;

    cl::Event* eventPtr = this->profiling ? &events[idx] : nullptr;

    cl::NDRange localRange = cl::NullRange;

    if (it->localWorkSizeX > 0) {
      ulong lwsX = it->localWorkSizeX;
      countX = ((countX + lwsX - 1) / lwsX) * lwsX; // round up dim 0

      // Use lwsY=1 when not specified, to keep local range 2D matching global range
      ulong lwsY = (it->localWorkSizeY > 0) ? it->localWorkSizeY : 1;
      countY = ((countY + lwsY - 1) / lwsY) * lwsY; // round up dim 1
      localRange = cl::NDRange(lwsX, lwsY);
    }

    cl::NDRange globalRange(countX, countY);
    cl::NDRange offsetRange(offsetX, 0);

    cl_int result =
      this->queue.enqueueNDRangeKernel(it->kernel, offsetRange, globalRange, localRange, nullptr, eventPtr);

    if (result != CL_SUCCESS) {
      std::cout << " Error enqueueing kernel " << it->name << ": " << result << "\n";
      exit(1);
    }

    this->queue.enqueueBarrier();
  }

  this->queue.flush();

  if (this->profiling) {
    this->queue.finish();
    idx = 0;
    for (std::vector<Kernel>::iterator it = this->kernels.begin(); it != this->kernels.end(); it++, idx++) {
      events[idx].wait();
      cl_ulong start = events[idx].getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end = events[idx].getProfilingInfo<CL_PROFILING_COMMAND_END>();
      double ms = (end - start) / 1e6;
      this->kernelTotalTime[it->name] += ms;
      this->kernelCallCount[it->name]++;
    }
  }

  std::cout.flush();
}

//===================================================================================================================//

void ComputeUnit::waitFinish()
{
  this->queue.finish();
}

//===================================================================================================================//
//-- Build helpers (private) --//
//===================================================================================================================//

void ComputeUnit::buildContext()
{
  if (this->verbose)
    std::cout << "Building context...\n";

  cl_int result;

  this->context = cl::Context(this->device, NULL, NULL, NULL, &result);

  if (result != CL_SUCCESS) {
    std::cout << " Error creating context: " << result << "\n";
    exit(1);
  }

  if (this->verbose) {
    std::cout << " Done!\n";
    std::cout.flush();
  }
}

//===================================================================================================================//

void ComputeUnit::buildQueue()
{
  if (this->verbose)
    std::cout << "Building queue...\n";

  cl_command_queue_properties props = this->profiling ? CL_QUEUE_PROFILING_ENABLE : 0;
  this->queue = cl::CommandQueue(this->context, this->device, props);

  if (this->verbose) {
    std::cout << " Done!\n";
    std::cout.flush();
  }
}

//===================================================================================================================//

void ComputeUnit::buildProgram()
{
  if (this->verbose)
    std::cout << "Building program...\n";

  // Rebuild sources from sourceStrings to ensure pointers are valid
  // (vector reallocation may have invalidated previous c_str() pointers)
  this->sources.clear();
  for (const std::string& str : this->sourceStrings) {
    this->sources.push_back({str.c_str(), str.length()});
  }

  cl_int result;

  this->program = cl::Program(this->context, this->sources, &result);

  if (result != CL_SUCCESS) {
    std::cout << " Error creating program: " << result << "\n";
    exit(1);
  }

  //if(this->program.build({this->device}, "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-strict-aliasing -cl-denorms-are-zero -I ./") != CL_SUCCESS) {
  if (this->program.build({this->device}, "-I ./") != CL_SUCCESS) {
    std::cout << " Error building: " << this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device) << "\n";
    std::cout.flush();
    exit(1);
  }

  this->programBuilt = true;

  if (this->verbose) {
    std::cout << " Done!\n";
    std::cout.flush();
  }
}
