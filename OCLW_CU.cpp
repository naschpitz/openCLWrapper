#include "OCLW_CU.hpp"

#include <iostream>

using namespace OpenCLWrapper;

ComputeUnit::ComputeUnit()
{
  this->programBuilt = false;
}

ComputeUnit::ComputeUnit(const cl::Device& device, uint index, double fraction, bool last)
{
  this->programBuilt = false;

  this->device = device;
  this->index = index;
  this->fraction = fraction;
  this->last = last;

  this->buildContext();
  this->buildQueue();
}

void ComputeUnit::buildContext()
{
  std::cout << "Building context...\n";

  cl_int result;

  this->context = cl::Context(this->device, NULL, NULL, NULL, &result);

  if(result != CL_SUCCESS) {
    std::cout << " Error creating context: " << result << "\n";
    exit(1);
  }

  std::cout << " Done!\n";
  std::cout.flush();
}

void ComputeUnit::buildQueue()
{
  std::cout << "Building queue...\n";

  this->queue = cl::CommandQueue(this->context, this->device);

  std::cout << " Done!\n";
  std::cout.flush();
}

void ComputeUnit::buildProgram()
{
  std::cout << "Building program...\n";

  // Rebuild sources from sourceStrings to ensure pointers are valid
  // (vector reallocation may have invalidated previous c_str() pointers)
  this->sources.clear();
  for (const std::string& str : this->sourceStrings) {
    this->sources.push_back({str.c_str(), str.length()});
  }

  cl_int result;

  this->program = cl::Program(this->context, this->sources, &result);

  if(result != CL_SUCCESS) {
    std::cout << " Error creating program: " << result << "\n";
    exit(1);
  }

  //if(this->program.build({this->device}, "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-strict-aliasing -cl-denorms-are-zero -I ./") != CL_SUCCESS) {
  if(this->program.build({this->device}, "-I ./") != CL_SUCCESS) {
    std::cout << " Error building: " << this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device) << "\n";
    std::cout.flush();
    exit(1);
  }

  this->programBuilt = true;

  std::cout << " Done!\n";
  std::cout.flush();
}

void ComputeUnit::addSource(const std::string& sourceCode)
{
  // Store the string - sources will be rebuilt from sourceStrings in buildProgram()
  this->sourceStrings.push_back(sourceCode);
}

void ComputeUnit::setVerbose(bool verbose)
{
  this->verbose = verbose;
}

void ComputeUnit::addKernel(const std::string& kernelName, ulong nElements, ulong offset)
{
  this->addKernel(kernelName, kernelName, nElements, offset);
}

void ComputeUnit::addKernel(const std::string& id, const std::string& kernelName, ulong nElements, ulong offset)
{
  if(!this->programBuilt)
    this->buildProgram();

  Kernel kernel;
  kernel.name = id;  // Use id for argument lookup
  kernel.nElements = nElements;
  kernel.offset = offset;

  if(this->verbose)
    std::cout << "Building kernel " << kernelName << " (id: " << id << ")...\n";

  cl_int result;
  kernel.kernel = cl::Kernel(this->program, kernelName.c_str(), &result);  // Use kernelName for OpenCL lookup

  if(result != CL_SUCCESS) {
    std::cout << " Error creating kernel " << kernelName << ": " << result << "\n";
    exit(1);
  }

  this->kernels.push_back(kernel);

  if(this->verbose) {
    std::cout << " Done!\n";
    std::cout.flush();
  }
}

void ComputeUnit::clearKernels()
{
  this->kernels.clear();

  if(this->verbose) {
    std::cout << "Kernels cleaned\n";
    std::cout.flush();
  }
}

void ComputeUnit::run()
{
  for(std::vector<Kernel>::iterator it = this->kernels.begin(); it != this->kernels.end(); it++) {
    uint count = it->nElements * this->fraction;
    uint offset = this->index * count + it->offset;

    if(this->last)
      count = it->nElements - offset;

    cl_int result = this->queue.enqueueNDRangeKernel(it->kernel, offset, cl::NDRange(count), cl::NullRange);

    if(result != CL_SUCCESS) {
      std::cout << " Error enqueueing kernel " << it->name << ": " << result << "\n";
      exit(1);
    }

    this->queue.enqueueBarrier();
  }

  this->queue.flush();
  std::cout.flush();
}

void ComputeUnit::waitFinish()
{
  this->queue.finish();
}
