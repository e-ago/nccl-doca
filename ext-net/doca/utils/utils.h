/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE for license information
 ************************************************************************/

#ifndef NCCL_GIC_MLX5DV_PLUGIN_UTILITIES_H_
#define NCCL_GIC_MLX5DV_PLUGIN_UTILITIES_H_

#include "net.h"

#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdarg.h>
#include <unistd.h>
#include <stdlib.h>

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
static bool matchIf(const char* string, const char* ref, bool matchExact);
static bool matchPort(const int port1, const int port2);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

#define NCCL_NET_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

extern ncclDebugLogger_t nccl_log_func;

#define WARN(fmt, ...)                                                  \
  (*nccl_log_func)(NCCL_LOG_WARN, NCCL_ALL, __PRETTY_FUNCTION__,        \
  __LINE__, fmt, ##__VA_ARGS__)

#define INFO(flags, fmt, ...)                           \
  (*nccl_log_func)(NCCL_LOG_INFO, flags,                \
  __PRETTY_FUNCTION__, __LINE__, fmt,           \
  ##__VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(flags, fmt, ...)                          \
  (*nccl_log_func)(NCCL_LOG_TRACE, flags,               \
  __PRETTY_FUNCTION__, __LINE__, fmt,           \
  ##__VA_ARGS__)
#else
#define TRACE(...)
#endif

#define NCCL_PARAM(name, env, default_value) \
pthread_mutex_t ncclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t ncclParam##name() { \
  static_assert(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL; \
  pthread_mutex_lock(&ncclParamMutex##name); \
  if (value == -1LL) { \
    value = default_value; \
    char* str = getenv("NCCL_" env); \
    if (str && strlen(str) > 0) { \
      errno = 0; \
      int64_t v = strtoll(str, NULL, 0); \
      if (errno) { \
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lu.", str, "NCCL_" env, value); \
      } else { \
        value = v; \
        INFO(NCCL_ALL,"%s set by environment to %lu.", "NCCL_" env, value);  \
      } \
    } \
  } \
  pthread_mutex_unlock(&ncclParamMutex##name); \
  return value; \
}

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

extern thread_local int ncclDebugNoWarn;
#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess && res != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, ncclSystemError);    \
    return ncclSystemError;     \
  }                             \
} while (0);

#define EQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

#define SYSCHECKGOTO(statement, res, label) do { \
  if ((statement) == -1) {    \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
               ncclBfloat16   = 9,
               ncclNumTypes   = 10
#else
               ncclNumTypes   = 9
#endif
} ncclDataType_t;

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#include <malloc.h>

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}

#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  if (nelem < oldNelem) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* oldp = *ptr;
  T* p = (T*)malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memcpy(p, oldp, oldNelem*sizeof(T));
  free(oldp);
  memset(p+oldNelem, 0, (nelem-oldNelem)*sizeof(T));
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*sizeof(T), nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

#endif  // NCCL_GIC_MLX5DV_PLUGIN_UTILITIES_H_
