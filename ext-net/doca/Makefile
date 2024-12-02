#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME:=../../build/
CUDA_HOME ?= /usr/local/cuda
DOCA_HOME ?= /opt/mellanox/doca
DPDK_HOME ?= /opt/mellanox/dpdk
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -Icuda -Imlx5 -Iutils -Inccl -Idevice -I$(DOCA_HOME)/include -I$(DPDK_HOME)/include/dpdk/../x86_64-linux-gnu/dpdk -I${DPDK_HOME}/include/dpdk -include rte_config.h
PLUGIN_SO:=libnccl-net-doca.so
TRACE ?= 0
CXXFLAGS += -DALLOW_EXPERIMENTAL_API
CXXFLAGS += -DDOCA_ALLOW_EXPERIMENTAL_API

ifneq ($(TRACE), 0)
CXXFLAGS  += -DENABLE_TRACE
endif

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.cc utils/socket.cc utils/utils.cc cuda/cudawrap.cc
	$(CC) $(INC) -g -std=c++1z -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^ -lpthread -libverbs -lmlx5 \
				-L$(CUDA_HOME)/lib64 -lcudart \
				-Wl,--as-needed -L${DPDK_HOME}/lib/x86_64-linux-gnu -lrte_node -lrte_graph -lrte_flow_perf -lrte_pipeline -lrte_table -lrte_pdump -lrte_port -lrte_fib -lrte_ipsec -lrte_vhost -lrte_stack -lrte_security -lrte_sched -lrte_reorder -lrte_rib -lrte_dmadev -lrte_regexdev -lrte_rawdev -lrte_power -lrte_pcapng -lrte_member -lrte_lpm -lrte_latencystats -lrte_jobstats -lrte_gso -lrte_gro -lrte_gpudev -lrte_eventdev -lrte_efd -lrte_distributor -lrte_cryptodev -lrte_compressdev -lrte_cfgfile -lrte_bpf -lrte_bitratestats -lrte_bbdev -lrte_acl -lrte_timer -lrte_metrics -lrte_cmdline -lrte_pci -lrte_ethdev -lrte_ip_frag -lrte_hash -lrte_meter -lrte_net -lrte_mbuf -lrte_mempool -lrte_rcu -lrte_ring -lrte_eal -lrte_telemetry -lrte_kvargs \
				-L$(DOCA_HOME)/lib/x86_64-linux-gnu -ldoca_gpunetio -ldoca_rdma -ldoca_common -ldoca_dpdk_bridge $(CXXFLAGS)

clean:
	rm -f $(PLUGIN_SO)
