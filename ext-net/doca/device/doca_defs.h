/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef DOCA_DEFS_H_
#define DOCA_DEFS_H_

#define NCCL_DOCA_AR_THRESHOLD 8192
#define NCCL_SHARED_STEPS 16

#include <doca_error.h>

struct ncclNetDeviceDocaMr {
  struct doca_gpu_buf_arr *bufArrayGpu;
  int elemNum;
	size_t elemSize;
  uintptr_t addr;
  size_t size;
};

struct ncclNetDeviceDocaRemoteFifo {
  uint64_t   remAddr;
  uint64_t   size;
  uint32_t   count;
  uint32_t   rkey;
  uint32_t   position;
};

// This is 256-bit aligned
struct ncclNetDeviceDocaFifoElement {
  uint64_t addr;
  int      size;
  int      tag;
  uint32_t rkey;
  uint32_t ready;
};

// This is 256-bit aligned
struct ncclNetDeviceDocaFifoElementPadded {
  ncclNetDeviceDocaFifoElement elem;
  uint32_t padding[2];
};

struct ncclNetDeviceDocaFifo {
  ncclNetDeviceDocaFifoElementPadded* elems;
  uint64_t   size;
  uint32_t   count;
  uint32_t   position;
};

typedef struct {
  uintptr_t localFifoAddr;
  uint64_t localFifoSize;
  uint32_t localFifoElemsMask;
  uint32_t localFifoPosition;
  struct doca_gpu_buf_arr *localFifoBufArrayGpu;

  uintptr_t remoteFifoAddr;
  uint64_t remoteFifoSize;
  uint32_t remoteFifoElemsMask;
  uint32_t remoteFifoPosition;
  struct doca_gpu_buf_arr *remoteFifoBufArrayGpu;

  struct doca_gpu_dev_rdma *rdmaGpu;
  struct doca_gpu_dev_rdma_r *rdmaGpuR;
  uint32_t numRecvElements;
  uint32_t numSendElements;
  uint16_t rqpTail; // Tail is where we enqueue
  uint16_t sqpTail; // Tail is where we enqueue
  uint8_t phase; //???
} ncclNetDocaHandle;

// Per-SM
struct docaShmem {
};

// Per-Group
struct docaGroupShmem {
  ncclNetDocaHandle handle;
};

#endif // DOCA_DEFS_H_
