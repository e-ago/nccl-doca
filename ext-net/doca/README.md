# NCCL IBGDA External Network Plugin

## Dependancies for DOCA
DOCA requires:
 - libibverbs
 - libmlx5
 - nvidia-peermem or nv_peer_mem

For P2P offloading to work, the PeerMappingOverride registry must be set to 1:
`$cat /etc/modprobe.d/nvidia.conf`
`options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"`

If the display driver is r387 or newer, the CUDA Memory Operations API must be explicitly enabled by means of NVreg_EnableStreamMemOPs=1:
`$ cat /etc/modprobe.d/nvidia.conf`
`options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"`

After that, either reboot or manually reload the NVIDIA kernel module.

Note that PeerMappingOverride should be only enabled with careful consideration of system security. 

## Building the plugin
`make`

If the build fails, you probably don't have libibverbs-dev installed or it's not in your LD_LIBRARY_PATH. Try the following:
`export LD_LIBRARY_PATH=/usr/lib/:/usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/libibverbs:$LD_LIBRARY_PATH`

If that fails, then try installing libibverbs-dev:
`apt-get install libibverbs-dev`

Then make sure you've made symbolic links the build expects:
`ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so`
`ln -s /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav34.so /usr/lib/x86_64-linux-gnu/libmlx5.so`

## Running using the DOCA plugin
```export PLUGIN_PATH=$(pwd)
export NCCL_NET_PLUGIN=doca
export LD_LIBRARY_PATH=$PLUGIN_PATH:$LD_LIBRARY_PATH```

Please confirm when running that you see this line:
`Using NCCL_NET_DEVICE_DOCA_DEVX_VERSION_P2P_OFFLOAD net plugin version 8`
