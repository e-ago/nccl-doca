/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef DOCA_H_
#define DOCA_H_

#include "device.h"
#include "doca_defs.h"

#define DOCA_HDR_FILE 0

#if DOCA_HDR_FILE == 1
#include "doca_gpunetio_nccl.cuh"
#else
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_rdma.cuh>
#endif

#define LOG_DOCA_KERNEL 0
#define DOCA_TIMES 1

#if DOCA_TIMES
#define DOCA_GPUNETIO_DEVICE_GET_TIME(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))
#endif

#if __CUDA_ARCH__ >= 700

#define bswap_doca_32(x) \
  ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |           \
   (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))


// Map internal association of handle with group and peer index (called once at init time)
inline __device__ void ncclNetDeviceDocaSetup(void* oHandle, const int group) {
  ncclNetDocaHandle* handle = (ncclNetDocaHandle*) oHandle;
  doca_gpu_dev_rdma_get_recv(handle->rdmaGpu, &handle->rdmaGpuR);
  ncclShmem.groups[group].devicePlugin.doca.handle = *handle;
}

// Map internal association of handle with group and peer index (called once at init time)
inline __device__ void ncclNetDeviceDocaStore(void* oHandle, const int group) {
  ncclNetDocaHandle* handle = (ncclNetDocaHandle*) oHandle;
  *handle = ncclShmem.groups[group].devicePlugin.doca.handle;
}

// Returns quantity of CQE found
// If a CQE is found, returns the new connStepCache value equal to maxStepSeen + stepInc
// Referenced code from device_poll_cq() in device.cuh 
inline __device__ uint64_t ncclNetDeviceDocaCompletion(const int group, uint64_t connStepCache, bool isSendNotRecv, int stepInc, int* abort) {
  uint32_t num_ops_tx = 0;
  uint32_t num_ops_rx = 0;
  ncclNetDocaHandle* handle = &ncclShmem.groups[group].devicePlugin.doca.handle;

#if DOCA_TIMES
  unsigned long long start_ns = 0, end_ns = 0;
  DOCA_GPUNETIO_DEVICE_GET_TIME(start_ns);
#endif

  if (isSendNotRecv) {
    doca_gpu_dev_rdma_wait_all(handle->rdmaGpu, &num_ops_tx);
    #if DOCA_TIMES
    if (num_ops_tx > 0) {
      DOCA_GPUNETIO_DEVICE_GET_TIME(end_ns);
      printf("ncclNetDeviceDocaCompletion send %ld ns\n", end_ns - start_ns);
    }
    #endif
    return connStepCache + (stepInc * num_ops_tx);
  }
  else {
    doca_gpu_dev_rdma_recv_wait_all(handle->rdmaGpuR, DOCA_GPU_RDMA_RECV_WAIT_FLAG_NB, &num_ops_rx, nullptr);
    #if DOCA_TIMES
    if (num_ops_rx > 0) {
      DOCA_GPUNETIO_DEVICE_GET_TIME(end_ns);
      printf("ncclNetDeviceDocaCompletion recv %ld ns\n", end_ns - start_ns);
    }
    #endif
    return connStepCache + (stepInc * num_ops_rx);
  }
  #if LOG_DOCA_KERNEL == 1
  printf("[b:%u][t:%u] ncclNetDeviceDocaCompletion num_ops_tx %d num_ops_rx %d\n", blockIdx.x, threadIdx.x, num_ops_tx, num_ops_rx);
  #endif
}

// These need to be inlne for now. Would prefer these to not be but will manage
inline __device__ int ncclNetDeviceDocaSend(const int offset, const int size, const uint64_t step, const void* mhandle, const int group) {
  ncclNetDocaHandle* handle = &ncclShmem.groups[group].devicePlugin.doca.handle;
  struct doca_gpu_buf *localFifoBuf;
  struct doca_gpu_buf *localDataBuf;
#if DOCA_HDR_FILE == 0
  struct doca_gpu_buf *remoteDataBuf;
#endif
  uintptr_t localFifoBufAddr;

  // Wait for CTS
  // If fifoSlot was -1, assign it to the head of the fifo. fifoSlot can be specified for retrying old sends
  doca_gpu_dev_buf_get_buf(handle->localFifoBufArrayGpu, handle->localFifoPosition & handle->localFifoElemsMask, &localFifoBuf);
  doca_gpu_dev_buf_get_addr(localFifoBuf, &localFifoBufAddr);
  volatile ncclNetDeviceDocaFifoElementPadded* localElem = (volatile ncclNetDeviceDocaFifoElementPadded*)localFifoBufAddr;

  #if LOG_DOCA_KERNEL == 1
  printf("[b:%u][t:%u] ncclNetDeviceDocaSend waiting on READY flag[%d]=%lx step=%d group=%d\n",
        blockIdx.x, threadIdx.x, handle->localFifoPosition, localFifoBufAddr, step, group);
  #endif

  if (!localElem->elem.ready) return 0;

#if DOCA_TIMES
  unsigned long long start_ns = 0, end_ns = 0;
  DOCA_GPUNETIO_DEVICE_GET_TIME(start_ns);
#endif

  #if LOG_DOCA_KERNEL == 1
  printf("[b:%u][t:%u] ncclNetDeviceDocaSend IS READY flag[%d]=%lx step=%d group=%d\n",
        blockIdx.x, threadIdx.x, handle->localFifoPosition, localFifoBufAddr, step, group);
  #endif

#if DOCA_HDR_FILE == 0
  doca_gpu_dev_buf_create(localElem->elem.addr, localElem->elem.rkey, &remoteDataBuf);
#endif

  doca_gpu_dev_buf_get_buf(((ncclNetDeviceDocaMr*) mhandle)->bufArrayGpu, 0, &localDataBuf);

#if LOG_DOCA_KERNEL == 1
  printf("[b:%u][t:%u] ncclNetDeviceDocaSend rdmaGpu %p create remote buffer from addr %lx mkey %x size %d position %d handle->localFifoPosition %d\n",
          blockIdx.x, threadIdx.x, handle->rdmaGpu,
          localElem->elem.addr, localElem->elem.rkey, localElem->elem.size,
          handle->sqpTail & 0xFFFF, handle->localFifoPosition);
#endif

  if (size >= NCCL_DOCA_AR_THRESHOLD) {
#if DOCA_HDR_FILE == 1
    doca_gpu_dev_rdma_write_nccl(handle->rdmaGpu, 0,
            localElem->elem.addr, localElem->elem.rkey, 0,
            localDataBuf, offset,
            size,
            0,
            DOCA_GPU_RDMA_WRITE_FLAG_NONE,
            handle->sqpTail);
#else
    doca_gpu_dev_rdma_write_weak(handle->rdmaGpu, 0,
            remoteDataBuf, 0,
            localDataBuf, offset,
            size,
            0,
            DOCA_GPU_RDMA_WRITE_FLAG_NONE,
            handle->sqpTail);
#endif

    doca_gpu_dev_rdma_send_weak(handle->rdmaGpu, 0,
            nullptr, 0, 0,
            handle->localFifoPosition,
            DOCA_GPU_RDMA_SEND_FLAG_IMM,
            (handle->sqpTail+1) & 0xFFFF);

    doca_gpu_dev_rdma_commit_weak(handle->rdmaGpu, 0, 2);
    // Posted 2 send here
    handle->sqpTail++;
  } else if (size == 0) {
    doca_gpu_dev_rdma_send_weak(handle->rdmaGpu, 0,
            nullptr, 0, 0,
            handle->localFifoPosition,
            DOCA_GPU_RDMA_SEND_FLAG_IMM,
            handle->sqpTail);

    doca_gpu_dev_rdma_commit_weak(handle->rdmaGpu, 0, 1);
  } else {
#if DOCA_HDR_FILE == 1
  doca_gpu_dev_rdma_write_nccl(handle->rdmaGpu, 0,
            localElem->elem.addr, localElem->elem.rkey, 0,
            localDataBuf, offset,
            size,
            handle->localFifoPosition,
            DOCA_GPU_RDMA_WRITE_FLAG_IMM,
            handle->sqpTail);
#else
  doca_gpu_dev_rdma_write_weak(handle->rdmaGpu, 0,
            remoteDataBuf, 0,
            localDataBuf, offset,
            size,
            handle->localFifoPosition,
            DOCA_GPU_RDMA_WRITE_FLAG_IMM,
            handle->sqpTail);
#endif

    doca_gpu_dev_rdma_commit_weak(handle->rdmaGpu, 0, 1);
  }

  handle->sqpTail++;
  handle->localFifoPosition++; // = (handle->localFifoPosition + 1) & handle->localFifoElemsMask;
  localElem->elem.ready = 0;

  #if DOCA_TIMES
    DOCA_GPUNETIO_DEVICE_GET_TIME(end_ns);
    printf("ncclNetDeviceDocaSend %ld ns\n", end_ns - start_ns);
  #endif

  // Return num messages sent
  return 1;
}

inline __device__ void ncclNetDeviceDocaRecv(const int group, const void* buffer, const int offset, const int size, const uint64_t step, const void* mhandle) {
  struct doca_gpu_buf *localBuf;
  struct doca_gpu_buf *remoteBuf;
  struct doca_gpu_buf *dataBuf;
  uintptr_t localBufAddr;
  // uint32_t curr_position;
	// uint32_t mask_max_position;
  uint32_t dataMkey;

#if DOCA_TIMES
  unsigned long long start_ns = 0, end_ns = 0;
  DOCA_GPUNETIO_DEVICE_GET_TIME(start_ns);
#endif

  ncclNetDocaHandle* handle = &ncclShmem.groups[group].devicePlugin.doca.handle;

  const void* ptr = ((uint8_t*) buffer) + offset;
  // Recv QP needs to post a recv to be consumed for a CQE to be generated on send (write with immediate) completion

  /* Prepare and post recv */
  doca_gpu_dev_rdma_recv_weak(handle->rdmaGpuR, NULL, 0, 0, 0, handle->localFifoPosition & 0xFFFF); //curr_position);
  doca_gpu_dev_rdma_recv_commit_weak(handle->rdmaGpuR, 1);

  /* Send buffer info through RDMA Write and Fifo */
  doca_gpu_dev_buf_get_buf(handle->remoteFifoBufArrayGpu, handle->localFifoPosition & handle->localFifoElemsMask, &remoteBuf);
  /* Send buffer info through RDMA Write and Fifo */
  doca_gpu_dev_buf_get_buf(handle->localFifoBufArrayGpu, handle->localFifoPosition & handle->localFifoElemsMask, &localBuf);
  doca_gpu_dev_buf_get_addr(localBuf, &localBufAddr);
  
  //Replace with inline
  volatile ncclNetDeviceDocaFifoElementPadded* localElem = (volatile ncclNetDeviceDocaFifoElementPadded*)localBufAddr;
  doca_gpu_dev_buf_get_buf(((ncclNetDeviceDocaMr*) mhandle)->bufArrayGpu, 0, &dataBuf);
  doca_gpu_dev_buf_get_mkey(dataBuf, &dataMkey);
  localElem->elem.addr  = (uint64_t)ptr;
  localElem->elem.ready = 1;
  localElem->elem.rkey  = dataMkey;
  localElem->elem.size  = size;

  doca_gpu_dev_rdma_write_weak(handle->rdmaGpu, 0,
					remoteBuf, 0,
					localBuf, 0,
					sizeof(ncclNetDeviceDocaFifoElementPadded),
          0, //imm
					DOCA_GPU_RDMA_WRITE_FLAG_NONE,
          handle->localFifoPosition & 0xFFFF);
//     result = doca_gpu_dev_rdma_write_inline_weak(handle->rdmaGpu,
//                                         remoteBuf, 0,
//                                         (uint8_t *)localElem, sizeof(ncclNetDeviceDocaFifoElement),
// 				        0, //imm
//                                         DOCA_GPU_RDMA_WRITE_FLAG_NONE,
//                                         curr_position);

  doca_gpu_dev_rdma_commit_weak(handle->rdmaGpu, 0, 1);

  #if LOG_DOCA_KERNEL == 1
  printf("[b:%u][t:%u] ncclNetDeviceDocaRecv() rdmaGpu %p step %ld sent READY FIFO [%d]=%lx wqe %d addr %p = buffer %p + offset %d\n", // rkey %x size = %d\n",
    blockIdx.x, threadIdx.x, handle->rdmaGpu, step, handle->localFifoPosition, remoteBufAddr, curr_position, ptr, buffer, offset); //, dataMkey, size);
  #endif

  handle->localFifoPosition++; // = (handle->localFifoPosition+1) & handle->localFifoElemsMask;

  #if DOCA_TIMES
    DOCA_GPUNETIO_DEVICE_GET_TIME(end_ns);
    printf("ncclNetDeviceDocaRecv %ld ns\n", end_ns - start_ns);
  #endif

}

#else
// Define stubs
inline __device__ void ncclNetDeviceDocaSetup(void* /*oHandle*/, const int /*group*/) {}
inline __device__ void ncclNetDeviceDocaStore(void* /*oHandle*/, const int /*group*/) {}
inline __device__ uint64_t ncclNetDeviceDocaCompletion(const int /*group*/, uint64_t /*connStepCache*/, bool /*isSendNotRecv*/, int /*stepInc*/, int* /*abort*/) {}
inline __device__ int ncclNetDeviceDocaSend(const int /*size*/, const uint64_t /*step*/, const void* /*mhandle*/, const int /*group*/) {}
inline __device__ void ncclNetDeviceDocaRecv(const int /*group*/, const void* /*buffer*/, const int /*offset*/, const int /*size*/, const uint64_t /*step*/, const void* /*mhandle*/) {}

#endif

#endif // DOCA_H_
