/*************************************************************************
 * Copyright (c) 2023-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_DEVICE_H_
#define NET_DEVICE_H_

#define NCCL_NET_DEVICE_INVALID_VERSION      0x0
#define NCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define NCCL_NET_DEVICE_DOCA_VERSION_P2P_OFFLOAD 0x9
#define NCCL_NET_DEVICE_DOCA_VERSION NCCL_NET_DEVICE_DOCA_VERSION_P2P_OFFLOAD

typedef enum {NCCL_NET_DEVICE_HOST=0, NCCL_NET_DEVICE_UNPACK=1, NCCL_NET_DEVICE_IBGDA_DEVX=2, NCCL_NET_DEVICE_DOCA=3} ncclNetDeviceType;

typedef struct {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;
typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_t;

#endif
