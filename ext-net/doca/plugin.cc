/*************************************************************************
* Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
*
* This file implements an external NCCL network plugin implementing
* DOCA GPUNetIO (GPUDirect Async) using DOCA libraries for
* management operations.
* 
* See LICENSE.txt for license information
************************************************************************/

#include "net.h"

#define __hidden __attribute__ ((visibility("hidden")))

// C++ headers
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>

// Plugin utility headers
#include "limits.h"
#include "utils.h"
#include "socket.h"

// CUDA plugin header
#include "cudawrap.h"

// Mellanox headers
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include <doca_log.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_dev.h>
#include <doca_rdma.h>
#include <doca_rdma_bridge.h>
#include <doca_mmap.h>
#include <doca_buf_array.h>
#include <doca_dpdk.h>
#include <rte_ethdev.h>
#include <rte_common.h>

// IBGDA Device-side definitions
#include "doca_defs.h"

#define DOCA_MAX_CONN_DET 512
#define DOCA_QUEUE_DEPTH 512
#define DOCA_PAGE_BITS 16
#define DOCA_PAGE_SIZE (1ULL << DOCA_PAGE_BITS)

template <typename T>
inline T DOCA_ILOG2(T _n) {
return (T)ceil(log2((double)_n));
}

#define DOCA_ILOG2_OR0(_n)                                   \
( ((_n) == 0) ? 0 : DOCA_ILOG2(_n) )

#define DOCA_ROUND_UP_POW2(_n)                               \
({                                                      \
		uint32_t pow2 = 0;                               \
		assert((_n) >= 1);                                 \
		for (pow2 = 1; pow2 < (_n); pow2 <<= 1);           \
		pow2;                                              \
})

#define DOCA_ROUND_UP_POW2_OR_0(_n)                          \
( ((_n) == 0) ? 0 : DOCA_ROUND_UP_POW2(_n) )

#define MAXNAMESIZE 64

#define DOCA_MLX5_QPC_ATOMIC_MODE_UP_TO_64BIT 0x3
#define DOCA_LOG_MAX_MSG_SIZE 30    // 30 is max allowed on IB QPs
#define DOCA_MIN_RNR_NAK 12
#define DOCA_DB_REG_SIZE 8

static char ncclDocaIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclDocaIfAddr;
thread_local int ncclDebugNoWarn = 0;

struct ncclDocaMem {
	int refs;
	int pages;
	uintptr_t addr;
	bool hostMem;
	size_t size;
	size_t alignment;
	int elemNum;
	size_t elemSize;
	int dmabuf_fd;
	struct doca_gpu *gpuDev;
	struct doca_dev *docaDev;
	struct doca_mmap *mmap;
	void *rdmaExport;      /* RDMA export object to share with remote peer */
	size_t rdmaExportLen;	    /* RDMA export object size */
	struct doca_buf_arr *bufArray;	      /* DOCA buffer array */
	struct doca_gpu_buf_arr *bufArrayGpu; /* DOCA buffer array GPU obj */
};

struct ncclDocaMrCache {
	struct ncclDocaMem *slots;
	int capacity, population;
};

static int ncclNIbDevs = -1;
struct alignas(64) ncclDocaDev {
	pthread_mutex_t lock;

	int cuda_device;
	uint64_t guid;
	uint8_t port;
	uint8_t link_layer;
	int speed;
	int pdRefs;
	bool peerMemEnabled;

	int ib_device;
	struct doca_dev *docaDev;
	struct ibv_pd* pd;
	char devName[MAXNAMESIZE];
	char* pciPath;
	int realPort;
	int maxQp;
	
	struct doca_gpu *gpuDev;	    /* DOCA GPU device */

	struct ncclDocaMrCache mrCache;
	// struct ibv_srq *srq;
	// struct mlx5dv_srq dv_srq;

	// struct ibv_cq *srq_cq;
	// struct mlx5dv_cq dv_srq_cq;

	uint8_t endianness_mode; // Queried from the device telling us the endianness?
	int max_qp_rd_atom;
};

#define MAX_IB_PORT 15
#define MAX_IB_DEVS 16
struct ncclDocaDev ncclDocaDevs[MAX_IB_DEVS];
pthread_mutex_t ncclDocaLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclDocaRelaxedOrderingEnabled = 0;

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", 0);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 18);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbTrafficClass, "IB_TRAFFIC_CLASS", 0);
NCCL_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}

static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}


#if 0
// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

void ncclSetThreadName(pthread_t thread, const char *fmt, ...) {
	// pthread_setname_np is nonstandard GNU extension
	// needs the following feature test macro
	#ifdef _GNU_SOURCE
		if (ncclParamSetThreadName() != 1) return;
		char threadName[NCCL_THREAD_NAMELEN];
		va_list vargs;
		va_start(vargs, fmt);
		vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
		va_end(vargs);
		pthread_setname_np(thread, threadName);
	#endif
}
#endif

// Probably should be host-mapped memory since there could be errors in the initialization phase (need to put that stuff here?)
#if 0
pthread_t ncclIbgdaAsyncThread;
static void* ncclIbgdaAsyncThreadMain(void* args) {
	struct ibv_context* context = (struct ibv_context*)args;
	while (1) {
		struct ibv_async_event event;
		if (ncclSuccess != ibv_get_async_event(context, &event)) { break; }
		const char *str = ibv_event_type_str(event.event_type);
		if (str == nullptr) { break; }
		if (event.event_type != IBV_EVENT_COMM_EST)
			WARN("NET/DOCA : Got async event : %s", str);
		ibv_ack_async_event(&event);
	}
	return NULL;
}
#endif

NCCL_PARAM(DocaDisable, "DOCA_DISABLE", 0);

static ncclResult_t ncclDocaCreateMmap(struct ncclDocaMem *mem, void *addr, size_t size, bool relaxed, bool dmabuf) {
	doca_error_t result;
	ncclResult_t resultNccl;
	unsigned int flags = DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ;
	// if (relaxed) flags |= DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING;

	result = doca_mmap_create(&(mem->mmap));
	if (result != DOCA_SUCCESS) {
		WARN("doca_mmap_create: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}
	
	result = doca_mmap_set_permissions(mem->mmap, flags);
	if (result != DOCA_SUCCESS) {
		WARN("doca_mmap_set_permissions: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	if (dmabuf) {
		/* Map GPU memory buffer used to receive packets with DMABuf */
		result = doca_gpu_dmabuf_fd(mem->gpuDev, (void*)addr, size, &(mem->dmabuf_fd));
		if (result != DOCA_SUCCESS) {
			WARN("doca_gpu_dmabuf_fd: %d", doca_error_get_descr(result));
			return ncclInternalError;
		}

		result = doca_mmap_set_dmabuf_memrange(mem->mmap, mem->dmabuf_fd, (void*)addr, 0, size);
		if (result != DOCA_SUCCESS) {
			WARN("doca_mmap_set_dmabuf_memrange: %d", doca_error_get_descr(result));
			return ncclInternalError;
		}
	} else {
		result = doca_mmap_set_memrange(mem->mmap, addr, size);
		if (result != DOCA_SUCCESS) {
			WARN("doca_mmap_set_memrange: %d", doca_error_get_descr(result));
			return ncclInternalError;
		}
	}

	result = doca_mmap_add_dev(mem->mmap, mem->docaDev);
	if (result != DOCA_SUCCESS) {
		WARN("doca_mmap_add_dev: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_mmap_start(mem->mmap);
	if (result != DOCA_SUCCESS) {
		WARN("doca_mmap_start: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	/* export mmap for rdma */
	result = doca_mmap_export_rdma(mem->mmap, mem->docaDev, (const void**)&(mem->rdmaExport), (size_t*)&(mem->rdmaExportLen));
	if (result != DOCA_SUCCESS) {
		WARN("doca_mmap_export_rdma: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}
	INFO(NCCL_NET|NCCL_ENV, "NET/DOCA : Created local MMAP with addr %lx size %zd", addr, size);

	return ncclSuccess;
}

static ncclResult_t ncclDocaCreateBufArray(struct ncclDocaMem *mem) {
	doca_error_t result;

	result = doca_buf_arr_create(mem->elemNum, &mem->bufArray);
	if (result != DOCA_SUCCESS) {
		WARN("doca_buf_arr_create: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_buf_arr_set_params(mem->bufArray, mem->mmap, mem->elemSize, 0);
	if (result != DOCA_SUCCESS) {
		WARN("doca_buf_arr_set_params: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_buf_arr_set_target_gpu(mem->bufArray, mem->gpuDev);
	if (result != DOCA_SUCCESS) {
		WARN("doca_buf_arr_set_target_gpu: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_buf_arr_start(mem->bufArray);
	if (result != DOCA_SUCCESS) {
		WARN("doca_buf_arr_start: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_buf_arr_get_gpu_handle(mem->bufArray, &(mem->bufArrayGpu));
	if (result != DOCA_SUCCESS) {
		WARN("doca_buf_arr_get_gpu_handle: %d", doca_error_get_descr(result));
		return ncclInternalError;
	}

	return ncclSuccess;
}

static ncclResult_t ncclDocaCreateMemory(uintptr_t addr, size_t size, size_t alignment, uint32_t elemNum, size_t elemSize, struct doca_dev *docaDev, struct doca_gpu *gpuDev, bool relaxed, bool dmabuf, struct ncclDocaMem *mem)
{
	doca_error_t result;
	ncclResult_t ncclResult;
	cudaPointerAttributes attributes;

	mem->gpuDev = gpuDev;
	mem->docaDev = docaDev;
	mem->elemNum = elemNum;
	mem->elemSize = elemSize;
	mem->alignment = alignment;
	mem->addr = addr;

	ncclResult = ncclDocaCreateMmap(mem, (void*)addr, size, relaxed, dmabuf);
	if (ncclResult != ncclSuccess) {
		WARN("ncclDocaCreateMmap: %d", ncclResult);
		return ncclInternalError;
	}

	if (mem->rdmaExportLen > DOCA_MAX_CONN_DET) {
		WARN("MMAP mem->rdmaExportLen %d > DOCA_MAX_CONN_DET %d",
			mem->rdmaExportLen, DOCA_MAX_CONN_DET);
		return ncclInternalError;
	}

	ncclResult = ncclDocaCreateBufArray(mem);
	if (ncclResult != ncclSuccess) {
		WARN("ncclDocaCreateMmap: %d", ncclResult);
		return ncclInternalError;
	}

	return ncclSuccess;
}

static ncclResult_t ncclDocaGetPciPath(char* devName, char** path, int* realPort) {
	char devicePath[PATH_MAX];

	snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);

	char* p = realpath(devicePath, NULL);
	if (p == NULL) {
		WARN("Could not find real path of %s (%s)", devName, devicePath);
	} else {
		// Merge multi-port NICs into the same PCI device
		p[strlen(p)-1] = '0';
		// Also merge virtual functions (VF) into the same device
		p[strlen(p)-3] = '0';
		// And keep the real port aside (the ibv port is always 1 on recent cards)
		*realPort = 0;
		for (int d=0; d<ncclNIbDevs; d++)
			if (strcmp(p, ncclDocaDevs[d].pciPath) == 0) (*realPort)++;
	}
	*path = p;

	return ncclSuccess;
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
	int roMode = ncclParamIbPciRelaxedOrdering();
	ncclResult_t r = ncclInternalError;
	if (roMode == 1 || roMode == 2) {
		// Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
		// pd, addr, length, iova, access
		// r = ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0);
	}

	return 1; //r == ncclInternalError ? 0 : 1;
}

ncclResult_t ncclDocaInit(ncclDebugLogger_t logFunction) {
	struct netIf userIfs[MAX_IB_DEVS];
	struct doca_devinfo **dev_list;
	char devname[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
	int res;
	size_t i;
	static int shownIbHcaEnv = 0;
	// Detect IB cards
	uint32_t nIbDevs;
	struct ibv_pd *pd;
	struct ibv_device_attr devAttr;
	struct ibv_port_attr portAttr;
	uint64_t port_active_rate;
	doca_error_t result;
	int port = 0, portNum = 1;
	char *eal_param[128];
	int arg = 0;
	for (int i = 0; i < 128; i++) eal_param[i] = (char*)calloc(128, sizeof(char));
	strncpy(eal_param[arg++], "", 128 - 1);
	strncpy(eal_param[arg++], "-a", 128 - 1);
	strncpy(eal_param[arg++], std::string("00:00.0").c_str(), 128 - 1);
	strncpy(eal_param[arg++], "--in-memory", 128 - 1);
#if 0
    struct doca_log_backend *stdout_logger = nullptr;

    result = doca_log_backend_create_with_file_sdk(stdout, &stdout_logger);
    if (result != DOCA_SUCCESS)
            return ncclInternalError;

    result = doca_log_backend_set_sdk_level(stdout_logger, DOCA_LOG_LEVEL_TRACE);
    if (result != DOCA_SUCCESS)
            return ncclInternalError;
#endif
	nccl_log_func = logFunction;
	if (ncclParamDocaDisable()) return ncclInternalError;
	//sleep(30);
	if (ncclNIbDevs == -1) {
		pthread_mutex_lock(&ncclDocaLock);
//		char *eal_param[5] = {"-a", "00:00.0", "-m", "1024", "--in-memory"};
		res = rte_eal_init(arg, eal_param);
		if (res < 0) {
			INFO(NCCL_NET|NCCL_ENV, "NET/DOCA: DPDK iniit failed res %d param 0: %s 1: %s 2: %s 3: %s 4: %s 5: %s\n",
				res,
				eal_param[0], eal_param[1], eal_param[2], eal_param[3], eal_param[4], eal_param[5]);
			WARN("DPDK iniit failed: %d", res);
			return ncclInternalError;
		}
		NCCLCHECK(ncclCudaLibraryInit());
		if (ncclNIbDevs == -1) {
			ncclNIbDevs = 0;
			if (ncclFindInterfaces(ncclDocaIfName, &ncclDocaIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
				WARN("NET/DOCA : No IP interface found.");
				return ncclInternalError;
			}

			// Check if user defined which IB device:port to use
			char* userIbEnv = getenv("NCCL_IB_HCA");
			if (userIbEnv != NULL && shownIbHcaEnv++ == 0)
				INFO(NCCL_NET|NCCL_ENV, "NET/DOCA: NCCL_IB_HCA set to %s", userIbEnv);

			bool searchNot = userIbEnv && userIbEnv[0] == '^';
			if (searchNot) userIbEnv++;

			bool searchExact = userIbEnv && userIbEnv[0] == '=';
			if (searchExact) userIbEnv++;

			int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

			result = doca_devinfo_create_list(&dev_list, &nIbDevs);
			if (result != DOCA_SUCCESS) {
				WARN("Failed to load doca devices list: %s", doca_error_get_descr(result));
				return ncclInternalError;
			}

			if (nIbDevs == 0) return ncclInternalError;

			/* Search */
			for (int d = 0; d < nIbDevs && ncclNIbDevs < MAX_IB_DEVS; d++) {
				result = doca_devinfo_get_ibdev_name(dev_list[d], devname, DOCA_DEVINFO_IBDEV_NAME_SIZE);
					if (result != DOCA_SUCCESS) {
					WARN("Failed to load doca devices list: %s", doca_error_get_descr(result));
					return ncclInternalError;
				}
				// Does DOCA distinguish between device and port?
				INFO(NCCL_NET|NCCL_ENV, "NET/DOCA : Check device %s port %d", devname, port);
				if (! (matchIfList(devname, port, userIfs, nUserIfs, searchExact) ^ searchNot)) {
					INFO(NCCL_NET|NCCL_ENV, "NET/DOCA : matchIfList failed");
					continue;
				}
					

				/* If any special capabilities are needed */
				// if (func != NULL && func(dev_list[d]) != DOCA_SUCCESS)
				// continue;

				pthread_mutex_init(&ncclDocaDevs[ncclNIbDevs].lock, NULL);

				/* Get the bandwidth of the device port */
				result = doca_devinfo_get_active_rate(dev_list[d], &port_active_rate);
				if (result != DOCA_SUCCESS) {
					WARN("Error: Failed to get active rate for DOCA device\n");
					return ncclInternalError;
				}
				/* convert bandwidth from bits/s to MB/s units */
				port_active_rate = ((port_active_rate / 1000000) / 8);

				/* if device can be opened */
				result = doca_dev_open(dev_list[d], &(ncclDocaDevs[ncclNIbDevs].docaDev));
				if (result != DOCA_SUCCESS) {
					WARN("NET/DOCA : Unable to doca_dev_open dev %s reason %s", devname, doca_error_get_descr(result));
					doca_devinfo_destroy_list(dev_list);
					return ncclInternalError;
				}

				result = doca_rdma_bridge_get_dev_pd(ncclDocaDevs[ncclNIbDevs].docaDev, &pd);
				if (result != DOCA_SUCCESS) {
					WARN("NET/DOCA : Unable to doca_rdma_bridge_get_dev_pd %s", doca_error_get_descr(result));
					doca_devinfo_destroy_list(dev_list);
					return ncclInternalError;
				}

				memset(&devAttr, 0, sizeof(devAttr));
				if (ibv_query_device(pd->context, &devAttr)) {
					WARN("NET/DOCA : Unable to query device %s", doca_error_get_descr(result));
					continue;
				}

				if (res = ibv_query_port(pd->context, portNum, &portAttr)) {
					WARN("NET/DOCA : Unable to query port %d res %d", port, res);
					continue;
				}

				TRACE(NCCL_INIT|NCCL_NET,"NET/DOCA: [%d] %s:%d/%s ", d, devname, port,
					portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");

				ncclDocaDevs[ncclNIbDevs].ib_device = d;
				ncclDocaDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
				ncclDocaDevs[ncclNIbDevs].port = port;
				ncclDocaDevs[ncclNIbDevs].link_layer = portAttr.link_layer;
				// Check DOCA capabilities
				ncclDocaDevs[ncclNIbDevs].speed = (int)port_active_rate * ncclIbWidth(portAttr.active_width);
				// ncclDocaDevs[ncclNIbDevs].context = nullptr;
				ncclDocaDevs[ncclNIbDevs].pdRefs = 0;
				ncclDocaDevs[ncclNIbDevs].pd = pd;
				strncpy(ncclDocaDevs[ncclNIbDevs].devName, devname, DOCA_DEVINFO_IBDEV_NAME_SIZE);
				NCCLCHECK(ncclDocaGetPciPath(ncclDocaDevs[ncclNIbDevs].devName,
												&ncclDocaDevs[ncclNIbDevs].pciPath,
												&ncclDocaDevs[ncclNIbDevs].realPort));
				// Set const value
				ncclDocaDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
				ncclDocaDevs[ncclNIbDevs].mrCache.capacity = 0;
				ncclDocaDevs[ncclNIbDevs].mrCache.population = 0;
				ncclDocaDevs[ncclNIbDevs].mrCache.slots = NULL;
				ncclDocaDevs[ncclNIbDevs].max_qp_rd_atom = devAttr.max_qp_rd_atom;

				// Assume its doable. If later doca will not be able to create QP, an error will occurr
				ncclDocaDevs[ncclNIbDevs].peerMemEnabled = true;
				// Assumption: nccl calls cudaSetDevice() before initializing so this is guaranteed to return the correct device
				cudaGetDevice(&ncclDocaDevs[ncclNIbDevs].cuda_device);
				char pciBusId[1024];
				cudaDeviceGetPCIBusId(pciBusId, 1024, ncclDocaDevs[ncclNIbDevs].cuda_device);
				TRACE(NCCL_INIT|NCCL_NET,"NET/DOCA: NetDev %d CUDA device %d PCIe addr %s",
						ncclNIbDevs, ncclDocaDevs[ncclNIbDevs].cuda_device, pciBusId);

				// How do I get the gpuDev ? Should it be in the device or a separate istance?
				result = doca_gpu_create(pciBusId, &(ncclDocaDevs[ncclNIbDevs].gpuDev));
				if (result != DOCA_SUCCESS) {
					WARN("Function doca_gpu_create returned %s for PCIe %s ncclNIbDevs %d",
						doca_error_get_descr(result), pciBusId, ncclNIbDevs);
				}

				// pthread_create(&ncclIbgdaAsyncThread, NULL, ncclIbgdaAsyncThreadMain, context);
				// ncclSetThreadName(ncclIbgdaAsyncThread, "NCCL IbgdaAsync %2d", ncclNIbDevs);
				// pthread_detach(ncclIbgdaAsyncThread); // will not be pthread_join()'d
				ncclNIbDevs++;
			}

			doca_devinfo_destroy_list(dev_list);
		}

		if (ncclNIbDevs == 0) {
			WARN("NET/DOCA : No device found.");
		} else {
			char line[1024];
			line[0] = '\0';
			// Determine whether RELAXED_ORDERING is enabled and possible
			ncclDocaRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
			for (int d=0; d<ncclNIbDevs; d++)
				snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s", d, ncclDocaDevs[d].devName,
					ncclDocaDevs[d].port, ncclDocaDevs[d].link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
			line[1023] = '\0';

			char addrline[SOCKET_NAME_MAXLEN+1];
			INFO(NCCL_INIT|NCCL_NET, "NET/DOCA : Using%s %s; OOB %s:%s", line, ncclDocaRelaxedOrderingEnabled ? "[RO]" : "",
				ncclDocaIfName, ncclSocketToString(&ncclDocaIfAddr, addrline));
		}

		pthread_mutex_unlock(&ncclDocaLock);
	}

	INFO(NCCL_NET|NCCL_ENV, "NET/DOCA : Found %d devices", ncclNIbDevs);

	return ncclSuccess;
}

ncclResult_t ncclNDocaDevices(int* ndev) {
	*ndev = ncclNIbDevs;

	return ncclSuccess;
}

#define NCCL_NET_IB_MAX_RECVS 8

ncclResult_t ncclDocaGetProperties(int dev, ncclNetProperties_t* props) {

	INFO(NCCL_NET|NCCL_ENV, "NET/DOCA : Get properties for dev %d", dev);

	props->name = ncclDocaDevs[dev].devName;
	props->pciPath = ncclDocaDevs[dev].pciPath;
	props->guid = ncclDocaDevs[dev].guid;
	props->ptrSupport = NCCL_PTR_HOST;
	/*
	 * Assume this is always possible. Will support both dmabuf and nv-peermem.
	 * If mapping is not possible, will get an error later at mapping time.
	 * NCCL_PTR_CUDA == GDR support (via nv_peermem or dmabuf)
	 */
	props->ptrSupport |= NCCL_PTR_CUDA;

	props->speed = ncclDocaDevs[dev].speed;
	props->latency = 0; // Not set
	props->port = ncclDocaDevs[dev].port + ncclDocaDevs[dev].realPort;
	props->maxComms = ncclDocaDevs[dev].maxQp;
	props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
	props->netDeviceType    = NCCL_NET_DEVICE_DOCA;
	props->netDeviceVersion = NCCL_NET_DEVICE_DOCA_VERSION_P2P_OFFLOAD;
	props->regIsGlobal = 0; // Don't support user buffer registration for now
	// props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
	// props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;

	return ncclSuccess;
}

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

#define NCCL_IB_MAX_QPS 128

struct ncclDocaQpInfo {
	uint8_t connDetails[DOCA_MAX_CONN_DET];	/* Remote peer connection details */
	size_t connDetailsLen;		    /* Remote peer connection details data length */

	// FIFO RDMA info
	uint8_t fifoRdmaExport[DOCA_MAX_CONN_DET];	 /* RDMA export object to share with remote peer */
	size_t fifoRdmaExportLen;	    /* RDMA export object size */
	uintptr_t fifoAddr;
	uint64_t fifoSize;
	uint32_t fifoCount;
};

struct ncclDocaQp {
	bool dedicatedRecvQueue;
	bool deviceOffload;

	struct doca_rdma *rdma;		    /* DOCA RDMA instance */
	struct doca_gpu_dev_rdma *rdmaGpu; /* DOCA RDMA instance GPU handler */
	struct doca_ctx *rdmaCtx;	    /* DOCA context to be used with DOCA RDMA */
	
	uint32_t numRecvElements;
	uint32_t numSendElements;

	const void *connDetails;	/* Remote peer connection details */
	size_t connDetailsLen;		    /* Remote peer connection details data length */

	struct doca_rdma_connection *connection;
};

// Forward declare
typedef struct ncclDocaCommBase ncclDocaCommBase; 

struct ncclDocaCommBase {
	int dev;
	// struct ibv_pd* pd; // duplicate of ncclDocaDevs[dev].pd
	struct ncclDocaQp* qp;
	struct ncclSocket sock;
	int ready;

	struct ncclNetDeviceDocaFifo fifo;
	struct ncclDocaMem fifoMem;
	// struct ncclNetDeviceDocaFifo remFifo;
	struct ncclDocaMem remFifoMem;
	struct ncclNetDeviceDocaRemoteFifo remFifo;
};

enum ncclDocaCommState {
	ncclDocaCommStateStart = 0,
	ncclDocaCommStateSocketConnect = 1,
	ncclDocaCommStateAccept = 2,
	ncclDocaCommStateRecv = 3,
	ncclDocaCommStateConnected = 4,
	ncclDocaCommStateConnecting = 5,
	ncclDocaCommStateSend = 6,
	ncclDocaCommStatePendingReady = 7
};

struct ncclDocaCommStage {
	enum ncclDocaCommState state;
	int offset;
	void* buffer;
	void* comm;
	int fullOffload;
	
	const void *connDetails; /* Remote peer connection details */
	size_t connDetailsLen;		    /* Remote peer connection details data length */
};

struct ncclDocaHandle {
	union ncclSocketAddress connectAddr; // Filled by the target
	uint64_t magic; // random number to help debugging
	struct ncclDocaCommStage stage; // Used by the other side when connecting
};

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3

struct ncclDocaListenComm {
	int dev;
	struct ncclSocket sock;
	struct ncclDocaCommStage stage;
};

struct ncclDocaSendComm {
	struct ncclDocaCommBase base;
};

struct ncclDocaRecvComm {
	struct ncclDocaCommBase base;
	struct ibv_sge sge;
};

NCCL_PARAM(DocaQpsPerConn, "DOCA_QPS_PER_CONNECTION", 1);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(FifoDepth, "DOCA_FIFO_DEPTH", MAX_REQUESTS);

ncclResult_t ncclDocaGetDeviceHandle(ncclDocaCommBase* base, ncclNetDeviceHandle_t* handle) {
	if (!base->qp->deviceOffload) {
		handle->netDeviceType      = NCCL_NET_DEVICE_HOST;
		handle->needsProxyProgress = 1;
		WARN("DOCA Connection requested offload but was unable to initialize. Falling back to IBDEVX.");
		return ncclSuccess;
	}

	handle->needsProxyProgress = 0;
	handle->netDeviceType    = NCCL_NET_DEVICE_DOCA;
	handle->netDeviceVersion = NCCL_NET_DEVICE_DOCA_VERSION_P2P_OFFLOAD;

	// Populate handle, to be cudaMemCpy'd at the end of this function
	ncclNetDocaHandle objHandle = {0};
	// Dedicated Receive Queue. Assume it's alywas true
	// base->qp->dedicatedRecvQueue == true 

	// objHandle.localFifoAddr = base->fifoMem.hostPtr;
	objHandle.localFifoAddr = base->fifoMem.addr;
	objHandle.localFifoSize = base->fifoMem.elemSize;
	objHandle.localFifoElemsMask = (base->fifoMem.elemNum-1);
	objHandle.localFifoBufArrayGpu = base->fifoMem.bufArrayGpu;

	objHandle.remoteFifoAddr = base->remFifoMem.addr;
	// objHandle.remoteFifoAddr = base->remFifoMem.hostPtr;
	objHandle.remoteFifoSize = base->remFifoMem.elemSize;
	objHandle.remoteFifoElemsMask = (base->remFifoMem.elemNum - 1);
	objHandle.remoteFifoBufArrayGpu = base->remFifoMem.bufArrayGpu;

	objHandle.rdmaGpu = base->qp->rdmaGpu;
	objHandle.rdmaGpuR = nullptr;
	objHandle.numRecvElements = base->qp->numRecvElements;
	objHandle.numSendElements = base->qp->numSendElements;
	objHandle.rqpTail = 0;
	objHandle.sqpTail = 0;
	objHandle.phase = 0;

	base->fifo.position = 0;
	base->remFifo.position = 0;
	objHandle.localFifoPosition = 0;
	objHandle.remoteFifoPosition = 0;

	handle->size = sizeof(ncclNetDocaHandle);

#if 0
	INFO(NCCL_NET|NCCL_ENV, "DOCA Handle localFifoAddr %lx localFifoSize %d localFifoElems %d localFifoBufArrayGpu %lx",
			objHandle.localFifoAddr,
			objHandle.localFifoSize,
			objHandle.localFifoElemsMask+1,
			objHandle.localFifoBufArrayGpu);

	INFO(NCCL_NET|NCCL_ENV, "DOCA Handle remoteFifoAddr %lx remoteFifoSize %d remoteFifoElems %d remoteFifoBufArrayGpu %lx",
			objHandle.remoteFifoAddr,
			objHandle.remoteFifoSize,
			objHandle.remoteFifoElemsMask+1,
			objHandle.remoteFifoBufArrayGpu);

	INFO(NCCL_NET|NCCL_ENV, "DOCA Handle rdmaGpu %lx numRecvElements %d numSendElements %d",
			objHandle.rdmaGpu,
			objHandle.numRecvElements,
			objHandle.numSendElements);
#endif

	// Place opaque handle in device memory
	CUDACHECK(cudaMalloc(&handle->handle, handle->size));
	CUDACHECK(cudaMemcpy(handle->handle, &objHandle, handle->size, cudaMemcpyDefault));

	return ncclSuccess;
}

ncclResult_t ncclDocaInitBase(int dev, struct ibv_context* ctx, struct ncclDocaCommBase* base) {
	base->dev = dev;

	pthread_mutex_lock(&ncclDocaDevs[dev].lock);
	if (0 == ncclDocaDevs[dev].pdRefs++) {
		// 1. Allocate IB base protection domain
		// 2. Convert protection domain into index for DEVX
		// PDN is now stored in ncclDocaDevs[dev].dvpd.pdn
		// INFO(NCCL_NET, "dev=%u Allocated PDN=%u", dev, ncclDocaDevs[dev].dvpd.pdn);
		// Create shared receive queue to be shared by all send QPs owned by this process
		// Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
		// Query SRQN
		// SRQN is now stored in ncclDocaDevs[dev].dv_srq.srqn
		// Create host recv CQ for said CQ (ibverbs)
		// ncclDocaDevs[dev].srq_cq = ibv_create_cq(ctx, numSrqRequests, NULL, NULL, 0);
		// Query and store cqn (mlx5dv_init_obj())
		// CQN is now stored in ncclDocaDevs[dev].dv_srq_cq.srqn
		// INFO(NCCL_NET, "dev=%u Allocated shared receive CQN=%u", dev, ncclDocaDevs[dev].dv_srq_cq.cqn);
	}
	// base->pd = ncclDocaDevs[dev].pd;
	pthread_mutex_unlock(&ncclDocaDevs[dev].lock);

	return ncclSuccess;
}

ncclResult_t ncclDocaCommBaseDestroy(struct ncclDocaCommBase* base) {
	ncclResult_t res;
	pthread_mutex_lock(&ncclDocaDevs[base->dev].lock);

	if (0 == --ncclDocaDevs[base->dev].pdRefs) {
		TRACE(NCCL_NET, "Dealloc CommBase");
	}
	res = ncclSuccess;

returning:
	pthread_mutex_unlock(&ncclDocaDevs[base->dev].lock);
	return res;
}

ncclResult_t ncclDocaMemFree(ncclDocaMem* mem, bool device) {
	doca_error_t result;

	TRACE(NCCL_NET, "mem=%p size=%zu device=%u mem->alignment=%zu mem->addr=%p\n",
		mem, mem->size, device, mem->alignment, mem->addr);

	result = doca_mmap_destroy(mem->mmap);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to destroy mmap: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	if (device) {
		result = doca_gpu_mem_free(mem->gpuDev, (void *)mem->addr);
		if (result != DOCA_SUCCESS) {
			WARN("Failed to free gpu memory: %s", doca_error_get_descr(result));
			return ncclInternalError;
		}
	} else {
		free((void *)mem->addr);
	}

	mem->size = 0;

	return ncclSuccess;
}

ncclResult_t ncclDocaDestroyQp(struct ncclDocaQp* qp) {
	doca_error_t result = DOCA_SUCCESS;

	TRACE(NCCL_NET, "Destroying QP");

	result = doca_ctx_stop(qp->rdmaCtx);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to stop RDMA context: %s", doca_error_get_descr(result));
		return ncclSystemError;
	}

	/* Destroy DOCA RDMA */
	result = doca_rdma_destroy(qp->rdma);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to destroy DOCA RDMA: %s", doca_error_get_descr(result));
		return ncclSystemError;
	}

	free(qp);
	return ncclSuccess;
}

// Host-side reference: nvshmemt_ibdevx_mlx5_qp_create()
// dedicatedRecvQueue forces the following behavior:
// 1. Allocates dedicated receive space (else, use shared receive queue)
// 2. Allocates space in the dedicated CQ for recv queue entries
// 3. Forces the send to queue to not create completions on success (only used for CTS for receiver)
ncclResult_t ncclDocaCreateQp(ncclDocaDev* dev, struct ncclDocaCommBase* base, int access_flags, struct ncclDocaQp** qp, bool dedicatedRecvQueue, int deviceOffload) {

	doca_error_t result = DOCA_SUCCESS;
	uint32_t maxSendWqe;
	uint32_t maxRecvWqe;

	if (dedicatedRecvQueue == false) {
		WARN("DOCA supports only dedicated QP");
		return ncclInternalError;
	}

	// The plugin tends to set the proper cuda device before invoking this, although this behavior should be validated
	ncclDocaQp* newQp = (ncclDocaQp*) malloc(sizeof(ncclDocaQp));
	INFO(NCCL_NET|NCCL_ENV, "NET/DOCA : qp=%p newQp=%p deviceOffload=%u dev->peerMemEnabled=%d", qp, newQp, deviceOffload, dev->peerMemEnabled);
	// TRACE(NCCL_NET, "qp=%p newQp=%p deviceOffload=%u dev->peerMemEnabled=%d", qp, newQp, deviceOffload, dev->peerMemEnabled);
	newQp->deviceOffload = deviceOffload && dev->peerMemEnabled;

	// DOCA GPUNetIO supports only GPU communications. Will change in future
	if (newQp->deviceOffload == false) {
		WARN("DOCA GPUNetIO supports only full offload QP to GPU");
		return ncclInternalError;
	}

	result = doca_rdma_create(dev->docaDev, &(newQp->rdma));
	if (result != DOCA_SUCCESS) {
		WARN("Failed to create DOCA RDMA: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	newQp->rdmaCtx = doca_rdma_as_ctx(newQp->rdma);
	if (newQp->rdmaCtx == NULL) {
		result = DOCA_ERROR_UNEXPECTED;
		WARN("Failed to convert DOCA RDMA to DOCA context: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	/* Set permissions to DOCA RDMA */
	result = doca_rdma_set_permissions(newQp->rdma, access_flags);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to set permissions to DOCA RDMA: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	/* Set gid_index to DOCA RDMA */
	// RoCE, needed?
	result = doca_rdma_set_gid_index(newQp->rdma, ncclParamIbGidIndex());
	if (result != DOCA_SUCCESS) {
		WARN("Failed to set gid_index to DOCA RDMA: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_rdma_set_mtu(newQp->rdma, DOCA_MTU_SIZE_4K_BYTES);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to set mtu to DOCA RDMA: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	newQp->numSendElements = DOCA_QUEUE_DEPTH; //DOCA_ROUND_UP_POW2_OR_0(newQp->queue_depth);
	newQp->numRecvElements = DOCA_QUEUE_DEPTH; //DOCA_ROUND_UP_POW2_OR_0(newQp->queue_depth);

	INFO(NCCL_NET|NCCL_ENV, "qp=%p newQp=%p GID=%d numSendElements=%d numRecvElements=%d",
					qp, newQp, ncclParamIbGidIndex(), deviceOffload,
					newQp->numSendElements, newQp->numRecvElements);

	/* setup datapath of rdma ctx on gpu */
	result = doca_ctx_set_datapath_on_gpu(newQp->rdmaCtx, dev->gpuDev);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to set datapath on GPU: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_rdma_set_send_queue_size(newQp->rdma, newQp->numSendElements);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to set doca_rdma_set_send_queue_size on GPU: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_rdma_set_recv_queue_size(newQp->rdma, newQp->numRecvElements);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to set doca_rdma_set_recv_queue_size on GPU: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	// DOCA PE?

#if 0
	// RoCE, needed?
	result = doca_rdma_set_grh_enabled(newQp->rdma, true);
	if (result != DOCA_SUCCESS) {
		WARN("Unable to set ghr for RDMA: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}
#endif

	/* Start RDMA context */
	result = doca_ctx_start(newQp->rdmaCtx);
	if (result != DOCA_SUCCESS) {
		WARN("Failed to start RDMA context: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_rdma_get_gpu_handle(newQp->rdma, &(newQp->rdmaGpu));
	if (result != DOCA_SUCCESS) {
		WARN("Failed to get RDMA GPU handler: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	result = doca_rdma_export(newQp->rdma, &(newQp->connDetails), &(newQp->connDetailsLen), &(newQp->connection));
	if (result != DOCA_SUCCESS) {
		WARN("Failed to export RDMA with connection details");
		return ncclInternalError;
	}

	if (newQp->connDetailsLen > DOCA_MAX_CONN_DET) {
		WARN("QP connDetailsLen %d > DOCA_MAX_CONN_DET %d", newQp->connDetailsLen, DOCA_MAX_CONN_DET);
		return ncclInternalError;
	}

	base->fifo.position = 0;
	base->remFifo.position = 0;

	*qp = newQp;

	return ncclSuccess;
}

ncclResult_t ncclDocaListen(int dev, void* opaqueHandle, void** listenComm) {
	struct ncclDocaListenComm* comm;

	NCCLCHECK(ncclCalloc(&comm, 1));
	struct ncclDocaHandle* handle = (struct ncclDocaHandle*) opaqueHandle;
	static_assert(sizeof(struct ncclDocaHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclDocaHandle size too large");
	memset(handle, 0, sizeof(struct ncclDocaHandle));

	comm->dev = dev;
	handle->magic = NCCL_SOCKET_MAGIC;

	NCCLCHECK(ncclSocketInit(&comm->sock, &ncclDocaIfAddr, handle->magic, ncclSocketTypeNetDoca, NULL, 1));
	NCCLCHECK(ncclSocketListen(&comm->sock));
	NCCLCHECK(ncclSocketGetAddr(&comm->sock, &handle->connectAddr));

	*listenComm = comm;

	return ncclSuccess;
}

ncclResult_t ncclDocaSendConnect(int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
	struct ncclDocaHandle* handle = (struct ncclDocaHandle*) opaqueHandle;
	struct ncclDocaCommStage* stage = &handle->stage;
	struct ncclDocaSendComm* comm = (struct ncclDocaSendComm*) stage->comm;
	int ready;
	struct ncclDocaQpInfo qpInfo;
	doca_error_t result;
	ncclResult_t ncclResult;

	*sendComm = NULL;

	if (stage->state == ncclDocaCommStateSocketConnect) goto socket_connect_check;
	if (stage->state == ncclDocaCommStateSend)          goto socket_send_qp_details;
	if (stage->state == ncclDocaCommStateConnecting)    goto socket_recv_qp_details;
	if (stage->state == ncclDocaCommStateConnected)     goto socket_send_ready;
	if (stage->state != ncclDocaCommStateStart) {
		WARN("Error: trying to connect already connected sendComm");
		return ncclInternalError;
	}

	// Alloc container for internal metadata about this connection
	NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclDocaSendComm)));
	// Init out of band TCP socket to exchange info for connection
	NCCLCHECK(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetDoca, NULL, 1));
	stage->comm = comm;
	stage->state = ncclDocaCommStateSocketConnect;
	stage->fullOffload = (*sendDevComm != NULL);

	// Connect tcp socket
	NCCLCHECK(ncclSocketConnect(&comm->base.sock));

socket_connect_check:
	/* since ncclSocketConnect is async, we must check if connection is complete */
	NCCLCHECK(ncclSocketReady(&comm->base.sock, &ready));
	if (!ready) return ncclSuccess;

	// Initialize process-wide process singleton
	// [DOCA] we don't need this!
	INFO(NCCL_NET|NCCL_ENV, "NET/DOCA: ncclDocaSendConnect() creating Send QP dev=%u port=%u", dev, ncclDocaDevs[dev].port);
	NCCLCHECK(ncclDocaCreateQp(&ncclDocaDevs[dev], &comm->base,
								DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE,
								&comm->base.qp, true /* Dedicated recvCq */, stage->fullOffload));

	// Send my QP Info to receiver through the socket. Hope this won't block.

	/* ALLOC FIFO MEMORY IN GPU */
	{
		comm->base.fifo.count = ncclParamFifoDepth();
		comm->base.fifo.size = (uint64_t) sizeof(ncclNetDeviceDocaFifoElementPadded) * comm->base.fifo.count;
		comm->base.fifoMem.size = (size_t)comm->base.fifo.size;
		comm->base.fifoMem.elemNum = comm->base.fifo.count;
		comm->base.fifoMem.elemSize = sizeof(ncclNetDeviceDocaFifoElementPadded);

		if (stage->fullOffload) {
			comm->base.fifoMem.hostMem = false;
			comm->base.fifoMem.alignment = DOCA_PAGE_SIZE;
			comm->base.fifoMem.gpuDev = ncclDocaDevs[dev].gpuDev;

			result = doca_gpu_mem_alloc(ncclDocaDevs[dev].gpuDev,
						comm->base.fifoMem.size,
						comm->base.fifoMem.alignment,
						DOCA_GPU_MEM_TYPE_GPU,
						(void **)&comm->base.fifoMem.addr,
						nullptr);
			if (result != DOCA_SUCCESS) {
				WARN("Function doca_gpu_mem_alloc failed: %s", doca_error_get_descr(result));
				return ncclInternalError;
			}

			CUDACHECK(cudaMemset((void*)comm->base.fifoMem.addr, 0, comm->base.fifoMem.size));
			// Always allocate with my wrapper object to correctly release base pointers at cleanup
			comm->base.fifo.elems = (ncclNetDeviceDocaFifoElementPadded*) comm->base.fifoMem.addr;
		} else {
			comm->base.fifoMem.hostMem = true;
			comm->base.fifoMem.alignment = sysconf(_SC_PAGESIZE);
			ncclIbMalloc((void**)&comm->base.fifoMem.addr, comm->base.fifoMem.size);
			comm->base.fifo.elems = (ncclNetDeviceDocaFifoElementPadded*) comm->base.fifoMem.addr;
		}
	}

	/* CREATE FIFO MEMORY MMAP & BUF ARRAY */
	ncclResult = ncclDocaCreateMemory(comm->base.fifoMem.addr, comm->base.fifoMem.size, comm->base.fifoMem.alignment,
										comm->base.fifoMem.elemNum, comm->base.fifoMem.elemSize,
										ncclDocaDevs[dev].docaDev, ncclDocaDevs[dev].gpuDev,
										(ncclDocaRelaxedOrderingEnabled == true ? true : false),
										false /*dmabuf*/,
										&comm->base.fifoMem);
	if (ncclResult != ncclSuccess) {
		WARN("ncclDocaCreateMemory: %d", ncclResult);
		return ncclInternalError;
	}

	/* SendComm doesn't have remote fifo */
	comm->base.remFifo.remAddr = 0;
	comm->base.remFifo.size = 0;
	comm->base.remFifo.count = 0;
	comm->base.remFifo.position = 0;	

	if (comm->base.qp->connDetailsLen > DOCA_MAX_CONN_DET) {
		WARN("comm->base.qp->connDetailsLen %d > DOCA_MAX_CONN_DET %d",
				comm->base.qp->connDetailsLen, DOCA_MAX_CONN_DET);
		return ncclInternalError;
	}

	if (comm->base.fifoMem.rdmaExportLen > DOCA_MAX_CONN_DET) {
		WARN("comm->base.fifoMem.rdmaExportLen %d > DOCA_MAX_CONN_DET %d",
				comm->base.fifoMem.rdmaExportLen, DOCA_MAX_CONN_DET);
		return ncclInternalError;
	}

	qpInfo.connDetailsLen = comm->base.qp->connDetailsLen;
	memcpy(qpInfo.connDetails, comm->base.qp->connDetails, qpInfo.connDetailsLen);

	qpInfo.fifoRdmaExportLen = comm->base.fifoMem.rdmaExportLen;
	memcpy(qpInfo.fifoRdmaExport, comm->base.fifoMem.rdmaExport, qpInfo.fifoRdmaExportLen);

	qpInfo.fifoSize = comm->base.fifo.size;
	qpInfo.fifoCount = comm->base.fifo.count;
	qpInfo.fifoAddr = (uint64_t) comm->base.fifo.elems;

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaSendConnect() CREATE LOCAL FIFO fifo.size=%d fifo.count=%d fifo.addr=0x%lx connDetailsLen=%d",
			qpInfo.fifoSize, qpInfo.fifoCount, qpInfo.fifoAddr, qpInfo.connDetailsLen);

	// Prepare my fifo
	stage->state = ncclDocaCommStateSend;
	stage->offset = 0;
	
	NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(qpInfo)));
	memcpy(stage->buffer, &qpInfo, sizeof(qpInfo));

socket_send_qp_details:

	NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(qpInfo), &stage->offset));
	if (stage->offset != sizeof(qpInfo))
	return ncclSuccess;

	stage->state = ncclDocaCommStateConnecting;
	stage->offset = 0;
	// Clear the staging buffer for re-use
	memset(stage->buffer, 0, sizeof(qpInfo));

socket_recv_qp_details:
	struct ncclDocaQpInfo remQpInfo;

	NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(remQpInfo), &stage->offset));
	if (stage->offset != sizeof(remQpInfo)) {
		return ncclSuccess;
	}

	memcpy(&remQpInfo, stage->buffer, sizeof(ncclDocaQpInfo));

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaSendConnect() Connecting qp[%d]=%p detLen %zd fifoAddr %lx fifoSize %d fifoCount %d",
		0, comm->base.qp->rdma, (int)remQpInfo.connDetailsLen,
		remQpInfo.fifoAddr, remQpInfo.fifoSize, remQpInfo.fifoCount);

	result = doca_rdma_connect(comm->base.qp->rdma, remQpInfo.connDetails, remQpInfo.connDetailsLen, comm->base.qp->connection);
	if (result != DOCA_SUCCESS) {
		WARN("Function doca_rdma_connect failed: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	comm->base.ready = 1;
	stage->state = ncclDocaCommStateConnected;
	stage->offset = 0;

socket_send_ready:
	NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset));
	if (stage->offset != sizeof(int)) {
		return ncclSuccess;
	}

	free(stage->buffer);
	stage->state = ncclDocaCommStateStart;

	if (stage->fullOffload)
		NCCLCHECK(ncclDocaGetDeviceHandle(&comm->base, *sendDevComm));

	*sendComm = comm;

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaSendConnect() successfully exits!");

	return ncclSuccess;
}

ncclResult_t ncclDocaRecvConnect(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
	struct ncclDocaListenComm* lComm = (struct ncclDocaListenComm*)listenComm;
	struct ncclDocaCommStage* stage = &lComm->stage;
	struct ncclDocaRecvComm* rComm = (struct ncclDocaRecvComm*)stage->comm;
	int ready;
	doca_error_t result;
	ncclResult_t ncclResult;

	*recvComm = NULL;

	if (stage->state == ncclDocaCommStateAccept) goto socket_connect_check;
	if (stage->state == ncclDocaCommStateRecv) goto socket_recv_qp_details;
	if (stage->state == ncclDocaCommStateSend) goto socket_send_qp;
	if (stage->state == ncclDocaCommStatePendingReady) goto socket_recv_ready;
	if (stage->state != ncclDocaCommStateStart) {
		WARN("Listencomm in unknown state %d", stage->state);
		return ncclInternalError;
	}

	NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclDocaRecvComm)));
	stage->comm = rComm;
	stage->state = ncclDocaCommStateAccept;
	stage->fullOffload = (*recvDevComm != NULL);
	NCCLCHECK(ncclSocketInit(&rComm->base.sock));
	NCCLCHECK(ncclSocketAccept(&rComm->base.sock, &lComm->sock));

socket_connect_check:
	NCCLCHECK(ncclSocketReady(&rComm->base.sock, &ready));
	if (!ready) return ncclSuccess;

	struct ncclDocaQpInfo remQpInfo;
	stage->state = ncclDocaCommStateRecv;
	stage->offset = 0;
	NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remQpInfo)));

socket_recv_qp_details:

	NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remQpInfo), &stage->offset));
	if (stage->offset != sizeof(remQpInfo)) {
		if (stage->offset > 0)
			INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() stage->offset %d sizeof(remQpInfo) %zd",
					stage->offset, sizeof(remQpInfo));
		return ncclSuccess;
	}

	/* copy back the received info */
	memcpy(&remQpInfo, stage->buffer, sizeof(struct ncclDocaQpInfo));

	// QP Creation
	// [DOCA] we don't need this!
	INFO(NCCL_NET|NCCL_ENV, "Creating Recv QP dev=%u port=%u", lComm->dev, ncclDocaDevs[lComm->dev].port);
	NCCLCHECK(ncclDocaCreateQp(&ncclDocaDevs[lComm->dev], &rComm->base,
								DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE,
								&rComm->base.qp, true /* Dedicated recvCq */, stage->fullOffload));

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() Connecting qp[%d]=%p detLen %zd sizeof(remQpInfo) %zd fifoAddr %lx fifoSize %d fifoCount %d",
		0, rComm->base.qp->rdma, (int)remQpInfo.connDetailsLen, sizeof(remQpInfo),
		remQpInfo.fifoAddr, remQpInfo.fifoSize, remQpInfo.fifoCount);

	// Adjust the MTU
	// remQpInfo.mtu = (enum ibv_mtu)std::min(remQpInfo.mtu, portAttr.active_mtu);
	result = doca_rdma_connect(rComm->base.qp->rdma, remQpInfo.connDetails, remQpInfo.connDetailsLen, rComm->base.qp->connection);
	if (result != DOCA_SUCCESS) {
		WARN("Function doca_rdma_connect failed: %s", doca_error_get_descr(result));
		return ncclInternalError;
	}

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() CONNECTED!");

	// Retain remote fifo info and prepare my RDMA ops

	rComm->base.remFifo.count = remQpInfo.fifoCount;
	rComm->base.remFifo.size = remQpInfo.fifoSize;
	rComm->base.remFifoMem.elemNum = remQpInfo.fifoCount;
	rComm->base.remFifoMem.elemSize = sizeof(ncclNetDeviceDocaFifoElementPadded);
	// Right assumption?
	rComm->base.remFifoMem.alignment = DOCA_PAGE_SIZE;
	rComm->base.remFifoMem.addr = remQpInfo.fifoAddr;

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() Create Fifo from export fifoRdmaExportLen %d fifoAddr %lx fifoSize %d fifoCount %d",
		(int)remQpInfo.fifoRdmaExportLen, remQpInfo.fifoAddr, remQpInfo.fifoSize, remQpInfo.fifoCount);


	result = doca_mmap_create_from_export(NULL,
					      remQpInfo.fifoRdmaExport,
					      remQpInfo.fifoRdmaExportLen,
					      ncclDocaDevs[rComm->base.dev].docaDev,
					      &(rComm->base.remFifoMem.mmap));
	if (result != DOCA_SUCCESS) {
		WARN("Function doca_mmap_create_from_export failed: %s. fifoRdmaExportLen=%d dev=%d",
				doca_error_get_descr(result), remQpInfo.fifoRdmaExportLen, rComm->base.dev);
		return ncclInternalError;
	}

	/* CREATE REMOTE FIFO MEMORY BUF ARRAY */
	{
		result = doca_buf_arr_create(rComm->base.remFifoMem.elemNum, &(rComm->base.remFifoMem.bufArray));
		if (result != DOCA_SUCCESS)
			return ncclInternalError;

		result = doca_buf_arr_set_params(rComm->base.remFifoMem.bufArray, rComm->base.remFifoMem.mmap, rComm->base.remFifoMem.elemSize, 0);
		if (result != DOCA_SUCCESS)
			return ncclInternalError;

		result = doca_buf_arr_set_target_gpu(rComm->base.remFifoMem.bufArray, ncclDocaDevs[rComm->base.dev].gpuDev);
		if (result != DOCA_SUCCESS)
			return ncclInternalError;

		result = doca_buf_arr_start(rComm->base.remFifoMem.bufArray);
		if (result != DOCA_SUCCESS)
			return ncclInternalError;

		result = doca_buf_arr_get_gpu_handle(rComm->base.remFifoMem.bufArray, &(rComm->base.remFifoMem.bufArrayGpu));
		if (result != DOCA_SUCCESS)
			return ncclInternalError;
	}

	/* ALLOC FIFO MEMORY IN GPU */
	{
		rComm->base.fifo.count = remQpInfo.fifoCount;
		rComm->base.fifo.size  = sizeof(struct ncclNetDeviceDocaFifoElementPadded) * rComm->base.fifo.count;
		rComm->base.fifoMem.size = (size_t)rComm->base.fifo.size;
		rComm->base.fifoMem.elemNum = rComm->base.fifo.count;
		rComm->base.fifoMem.elemSize = sizeof(ncclNetDeviceDocaFifoElementPadded);

		if (stage->fullOffload) {
			rComm->base.fifoMem.hostMem = false;
			rComm->base.fifoMem.alignment = DOCA_PAGE_SIZE;
			rComm->base.fifoMem.gpuDev = ncclDocaDevs[rComm->base.dev].gpuDev;

			result = doca_gpu_mem_alloc(ncclDocaDevs[rComm->base.dev].gpuDev,
						rComm->base.fifoMem.size,
						rComm->base.fifoMem.alignment,
						DOCA_GPU_MEM_TYPE_GPU,
						(void **)&rComm->base.fifoMem.addr,
						nullptr);
			if (result != DOCA_SUCCESS) {
				WARN("Function doca_gpu_mem_alloc failed: %s", doca_error_get_descr(result));
				return ncclInternalError;
			}

			CUDACHECK(cudaMemset((void*)rComm->base.fifoMem.addr, 0, rComm->base.fifoMem.size));
			// Always allocate with my wrapper object to correctly release base pointers at cleanup
			rComm->base.fifo.elems = (ncclNetDeviceDocaFifoElementPadded*) rComm->base.fifoMem.addr;
		} else {
			rComm->base.fifoMem.hostMem = true;
			rComm->base.fifoMem.alignment = sysconf(_SC_PAGESIZE);
			ncclIbMalloc((void**)&rComm->base.fifoMem.addr, rComm->base.fifoMem.size);
			rComm->base.fifo.elems = (ncclNetDeviceDocaFifoElementPadded*) rComm->base.fifoMem.addr;
		}
	}

	/* CREATE FIFO MEMORY MMAP & BUF ARRAY */
	ncclResult = ncclDocaCreateMemory(rComm->base.fifoMem.addr, rComm->base.fifoMem.size, rComm->base.fifoMem.alignment,
										rComm->base.fifoMem.elemNum, rComm->base.fifoMem.elemSize,
										ncclDocaDevs[rComm->base.dev].docaDev, ncclDocaDevs[rComm->base.dev].gpuDev, 
										(ncclDocaRelaxedOrderingEnabled == true ? true : false),
										false, /* dmabuf */
										&rComm->base.fifoMem);
	if (ncclResult != ncclSuccess) {
		WARN("ncclDocaCreateMemory: %d", ncclResult);
		return ncclInternalError;
	}
	// rComm->sge.lkey = rComm->base.fifoMr->lkey;

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() initialized Receive Fifo: fifo.size=%d fifo.count=%d fifo.elems=%p",
		rComm->base.fifo.size, rComm->base.fifo.count, rComm->base.fifo.elems);

	// Fill Handle
	struct ncclDocaQpInfo qpInfo;
	qpInfo.connDetailsLen = rComm->base.qp->connDetailsLen;
	memcpy(qpInfo.connDetails, rComm->base.qp->connDetails, qpInfo.connDetailsLen);

	qpInfo.fifoRdmaExportLen = rComm->base.fifoMem.rdmaExportLen;
	memcpy(qpInfo.fifoRdmaExport, rComm->base.fifoMem.rdmaExport, qpInfo.fifoRdmaExportLen);

	qpInfo.fifoSize = rComm->base.fifo.size;
	qpInfo.fifoCount = rComm->base.fifo.count;
	qpInfo.fifoAddr = (uint64_t) rComm->base.fifo.elems;

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() sending fifo.size=%d fifo.count=%d fifo.addr=0x%lx connDetailsLen=%d",
			qpInfo.fifoSize, qpInfo.fifoCount, qpInfo.fifoAddr, qpInfo.connDetailsLen);

	stage->state = ncclDocaCommStateSend;
	stage->offset = 0;
	if (stage->buffer) free(stage->buffer);
	NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(qpInfo)));
	memcpy(stage->buffer, &qpInfo, sizeof(qpInfo));

socket_send_qp:

	NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(qpInfo), &stage->offset));
	if (stage->offset < sizeof(qpInfo)) return ncclSuccess;

	stage->offset = 0;
	stage->state = ncclDocaCommStatePendingReady;

socket_recv_ready:
	NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->base.sock, &rComm->base.ready, sizeof(int), &stage->offset));
	if (stage->offset != sizeof(int)) return ncclSuccess;
	free(stage->buffer);

	if (stage->fullOffload)
		NCCLCHECK(ncclDocaGetDeviceHandle(&rComm->base, *recvDevComm));

	*recvComm = rComm;

	/* reset lComm stage */
	stage->state = ncclDocaCommStateStart;
	stage->offset = 0;
	stage->comm = NULL;
	stage->buffer = NULL;

	INFO(NCCL_NET|NCCL_ENV, "ncclDocaRecvConnect() successfully exits!");

	return ncclSuccess;
}

ncclResult_t ncclDocaRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
	ncclResult_t ncclResult;
	doca_error_t result;
	static __thread uintptr_t pageSize = 0;

	static_assert(offsetof(struct ncclDocaSendComm, base) == offsetof(struct ncclDocaRecvComm, base), 
					"Send and recv comms must have base at the same offset");
	assert(size > 0);

	if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

	struct ncclDocaCommBase* base = (struct ncclDocaCommBase*)comm;
	struct ncclDocaMrCache* cache = &ncclDocaDevs[base->dev].mrCache;

	//Do I need this?
	uintptr_t addr = (uintptr_t)data;

	pthread_mutex_lock(&ncclDocaDevs[base->dev].lock);
	for (int slot=0; /*true*/; slot++) {
		if (slot == cache->population) { // didn't find in cache
			if (cache->population == cache->capacity) { // must grow cache
				cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
				NCCLCHECKGOTO(ncclRealloc(&cache->slots, cache->population, cache->capacity), ncclResult, returning);
			}

			/* CREATE FIFO MEMORY MMAP & BUF ARRAY */
			ncclResult = ncclDocaCreateMemory(addr, size, pageSize, 1, size,
												ncclDocaDevs[base->dev].docaDev, ncclDocaDevs[base->dev].gpuDev,
												(ncclDocaRelaxedOrderingEnabled == true ? true : false),
												false,
												&cache->slots[slot]);
			if (ncclResult != ncclSuccess) {
				WARN("ncclDocaCreateMemory: %d", ncclResult);
				goto returning;
			}

			//printf("ncclDocaRegMr addr=%p size=%lu\n", (void*)addr, size);

			cache->population += 1;
			cache->slots[slot].refs = 1;
			cache->slots[slot].addr = addr;
			*mhandle = (void*)(&cache->slots[slot]);

			ncclResult = ncclSuccess;
			goto returning;
		} else if (cache->slots[slot].addr == addr) {
			cache->slots[slot].refs += 1;
			*mhandle = (void*)(&cache->slots[slot]);

			ncclResult = ncclSuccess;
			goto returning;
		}
	}

returning:
	pthread_mutex_unlock(&ncclDocaDevs[base->dev].lock);
	return ncclResult;
}


ncclResult_t ncclDocaRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
	ncclResult_t ncclResult;
	doca_error_t result;
	static __thread uintptr_t pageSize = 0;

	static_assert(offsetof(struct ncclDocaSendComm, base) == offsetof(struct ncclDocaRecvComm, base), 
					"Send and recv comms must have base at the same offset");
	assert(size > 0);

	if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

	struct ncclDocaCommBase* base = (struct ncclDocaCommBase*)comm;
	struct ncclDocaMrCache* cache = &ncclDocaDevs[base->dev].mrCache;
	
	uintptr_t addr = (uintptr_t)data;	
	pthread_mutex_lock(&ncclDocaDevs[base->dev].lock);
	for (int slot=0; /*true*/; slot++) {
		if (slot == cache->population) { // didn't find in cache
			if (cache->population == cache->capacity) { // must grow cache
				cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
				NCCLCHECKGOTO(ncclRealloc(&cache->slots, cache->population, cache->capacity), ncclResult, returning);
			}

			ncclResult = ncclDocaCreateMemory(addr, size, pageSize, 1, size,
												ncclDocaDevs[base->dev].docaDev, ncclDocaDevs[base->dev].gpuDev,
												(ncclDocaRelaxedOrderingEnabled == true ? true : false),
												true,
												&cache->slots[slot]);
			if (ncclResult != ncclSuccess) {
				WARN("ncclDocaCreateMemory: %d", ncclResult);
				goto returning;
			}


			cache->population += 1;
			cache->slots[slot].refs = 1;
			cache->slots[slot].addr = addr;
			*mhandle = (void*)(&cache->slots[slot]);
			INFO(NCCL_NET, "ncclDocaRegMrDmabuf mem=%p addr=%p size=%lu\n",
					(void*)(&cache->slots[slot]), (void*)addr, size);
			ncclResult = ncclSuccess;
			goto returning;
		} else if (cache->slots[slot].addr == addr) {
			cache->slots[slot].refs += 1;
			*mhandle = (void*)(&cache->slots[slot]);

			ncclResult = ncclSuccess;
			goto returning;
		}
	}
returning:
	pthread_mutex_unlock(&ncclDocaDevs[base->dev].lock);
	return ncclResult;
}

ncclResult_t ncclDocaDeregMr(void* comm, void* mhandle) {
	struct ncclDocaCommBase* base = (struct ncclDocaCommBase*)comm;
	struct ncclDocaMrCache* cache = &ncclDocaDevs[base->dev].mrCache;
	ncclResult_t res;
	doca_error_t result;

	pthread_mutex_lock(&ncclDocaDevs[base->dev].lock);
	struct ncclDocaMem* mem = (struct ncclDocaMem*) mhandle;
	for (int i=0; i < cache->population; i++) {
		//WARN("NET/DOCA: mem=%p addr=%p", (void *)mem, (void*)mem->addr);
		if (mem == &(cache->slots[i])) {
			if (0 == --cache->slots[i].refs) {
				memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclDocaMem));
				if (cache->population == 0) {
					free(cache->slots);
					cache->slots = NULL;
					cache->capacity = 0;
				}
				
				WARN("NET/DOCA ncclDocaDeregMr mem=%p addr=%p\n", (void*)mem, (void*)mem->addr);
				//TODO: FILL THE DESTROY
				// result = doca_mmap_destroy(mmap);
				// if (result != DOCA_SUCCESS) {
				// 	WARN("Failed to destroy mmap: %s", doca_error_get_descr(result));
				// 	res = ncclSuccess;
				// 	goto returning;
				// }
			}

			res = ncclSuccess;
			goto returning;
		}
	}

	WARN("NET/DOCA: could not find mr %p inside cache of %d entries", mhandle, cache->population);
	res = ncclInternalError;

returning:
	pthread_mutex_unlock(&ncclDocaDevs[base->dev].lock);
	return res;
}

ncclResult_t ncclDocaGetDeviceMr(void* comm, void* mhandle, void** devMemHandle) {
	ncclNetDeviceDocaMr mr;
	mr.bufArrayGpu = ((struct ncclDocaMem*) mhandle)->bufArrayGpu;
	mr.elemNum = ((struct ncclDocaMem*) mhandle)->elemNum;
	mr.elemSize = ((struct ncclDocaMem*) mhandle)->elemSize;
	//TODO -- check if this could be hostPtr
	mr.addr = ((struct ncclDocaMem*) mhandle)->addr;
	mr.size = ((struct ncclDocaMem*) mhandle)->size;

	// cudaMalloc(devMemHandle, sizeof(ncclNetDeviceDocaMr));
	cudaMallocHost(devMemHandle, sizeof(ncclNetDeviceDocaMr));
	// *devMemHandle = malloc(sizeof(ncclNetDeviceDocaMr));
	cudaMemcpy(*devMemHandle, &mr, sizeof(ncclNetDeviceDocaMr), cudaMemcpyDefault);
	// memcpy(*devMemHandle, &mr, sizeof(ncclNetDeviceDocaMr));
	// *size = sizeof(ncclNetDeviceDocaMr);

	return ncclSuccess;
}

ncclResult_t ncclDocaCloseSend(void* sendComm) {
	struct ncclDocaSendComm* comm = (struct ncclDocaSendComm*)sendComm;
	if (comm) {
		NCCLCHECK(ncclSocketClose(&comm->base.sock));

		if (comm->base.fifoMem.size)
			NCCLCHECK(ncclDocaMemFree(&comm->base.fifoMem, comm->base.qp->deviceOffload));

		for (int q=0; q<1; q++) {
			if (comm->base.qp != NULL)
				NCCLCHECK(ncclDocaDestroyQp(comm->base.qp));
		}

		NCCLCHECK(ncclDocaCommBaseDestroy(&comm->base));
		free(comm);
	}
	return ncclSuccess;
}

ncclResult_t ncclDocaCloseRecv(void* recvComm) {
#if 0
	struct ncclDocaRecvComm* comm = (struct ncclDocaRecvComm*)recvComm;
	if (comm) {
		NCCLCHECK(ncclSocketClose(&comm->base.sock));
		int status = 0;

		// Free fifo
		if (comm->base.fifoMr != NULL) status = ibv_dereg_mr(comm->base.fifoMr);
		if (status != 0) {
			WARN("NET/DOCA: Failed ibv_dereg_mr(comm->base.fifoMr=%p, rkey=0x%x lkey=0x%x)",
			comm->base.fifoMr, comm->base.fifoMr->rkey, comm->base.fifoMr->lkey);
		}
		if (comm->base.fifoMem.size) NCCLCHECK(ncclDocaMemFree(&comm->base.fifoMem, comm->base.qp->deviceOffload));

		for (int q=0; q < 1; q++) {
			if (comm->base.qp != NULL) NCCLCHECK(ncclDocaDestroyQp(comm->base.qp));
		}

		NCCLCHECK(ncclDocaCommBaseDestroy(&comm->base));
		free(comm);
	}
#endif
	return ncclSuccess;
}

ncclResult_t ncclDocaCloseListen(void* listenComm) {
	struct ncclDocaListenComm* comm = (struct ncclDocaListenComm*)listenComm;
	if (comm) {
		NCCLCHECK(ncclSocketClose(&comm->sock));
		free(comm);
	}
	return ncclSuccess;
}

#define PLUGIN_NAME "DOCA"

volatile ncclNet_v8_t ncclNetPlugin_v8 = {
.name = PLUGIN_NAME,
ncclDocaInit,
ncclNDocaDevices,
ncclDocaGetProperties,
ncclDocaListen,
ncclDocaSendConnect,
ncclDocaRecvConnect,
ncclDocaRegMr,
ncclDocaRegMrDmaBuf,
ncclDocaDeregMr,
NULL, //ncclIbgdaISend,
NULL, //ncclIbgdaIrecv,
NULL, //ncclIbgdaIflush,
NULL, // ncclIbgdaTest,
ncclDocaCloseSend,
ncclDocaCloseRecv,
ncclDocaCloseListen,
ncclDocaGetDeviceMr,
NULL /*iRecvConsumed*/
};

