/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#ifndef __AXCL_RT_TYPE_H__
#define __AXCL_RT_TYPE_H__

#include "axcl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct axclrtDeviceProperties {
    char swVersion[64];     /* software version */
    uint64_t uid;           /* unique id */
    uint32_t pciDomain;     /* pci domain */
    uint32_t pciBusID;      /* pci bus id */
    uint32_t pciDeviceID;   /* pci device id */
    int32_t  temperature;   /* Tj */
    uint32_t totalMemSize;  /* unit: KB */
    uint32_t freeMemSize;   /* unit: KB */
    uint32_t totalCmmSize;  /* unit: KB */
    uint32_t freeCmmSize;   /* unit: KB */
    uint32_t cpuLoading;    /* average cpu loading * 100% */
    uint32_t npuLoading;    /* average npu loading * 100% */
    uint32_t reserved[32];
} axclrtDeviceProperties;

typedef struct axclrtDeviceList {
    uint32_t num;
    int32_t devices[AXCL_MAX_DEVICE_COUNT];
} axclrtDeviceList;

typedef enum axclrtMemMallocPolicy {
    AXCL_MEM_MALLOC_HUGE_FIRST,
    AXCL_MEM_MALLOC_HUGE_ONLY,
    AXCL_MEM_MALLOC_NORMAL_ONLY,
    AXCL_MEM_MALLOC_SIZE_ALIGN
} axclrtMemMallocPolicy;

typedef enum axclrtMemcpyKind {
    AXCL_MEMCPY_HOST_TO_HOST,
    AXCL_MEMCPY_HOST_TO_DEVICE,     //!< host vir -> device phy
    AXCL_MEMCPY_DEVICE_TO_HOST,     //!< host vir <- device phy
    AXCL_MEMCPY_DEVICE_TO_DEVICE,
    AXCL_MEMCPY_HOST_PHY_TO_DEVICE, //!< host phy -> device phy
    AXCL_MEMCPY_DEVICE_TO_HOST_PHY, //!< host phy <- device phy
} axclrtMemcpyKind;

/**
 * @brief Device status callback
 * @param device Device ID
 * @param status Device status
 * @param userdata User data by from axclRegisterDeviceStatusCallback
 */
typedef enum {
    AXCL_DEVICE_STATUS_ONLINE  = 0,
    AXCL_DEVICE_STATUS_OFFLINE = 1,
    AXCL_DEVICE_STATUS_BUTT
} AXCL_DEVICE_STATUS_E;
typedef void (*axclrtDeviceStatusCallback)(int32_t device, AXCL_DEVICE_STATUS_E status, void *userdata);

typedef enum {
     FILE_TRANSFER_FROM_HOST_TO_DEVICE   = 0,
     FILE_TRANSFER_FROM_DEVICE_TO_HOST   = 1,
     FILE_TRANSFER_FROM_DEVICE_TO_DEVICE = 2,
} axclrtFileTransferPolicy;

#ifdef __cplusplus
}
#endif

#endif /* __AXCL_RT_TYPE_H__ */
