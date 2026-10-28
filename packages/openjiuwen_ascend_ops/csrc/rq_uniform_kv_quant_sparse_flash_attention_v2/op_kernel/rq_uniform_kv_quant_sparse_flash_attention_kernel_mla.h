/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_common.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_service_cube_mla.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_service_vector_mla.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    uint64_t actS1Size = 1ULL; // TND场景下当前Batch循环处理的S1轴的大小，非TND场景下不要用这个字段
    uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小

    int32_t nextTokensPerBatch = 0;
    uint32_t tndCoreStartKVSplitPos;
    uint32_t gS1Idx = 0U;
    bool tndIsS2SplitCore;
    bool curActSeqLenIsZero = false;
};

template <typename QSFAT>
class RqUniformKvQuantSparseFlashAttentionMla {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename QSFAT::queryType;
    using KV_T = typename QSFAT::kvType;
    using OUT_T = typename QSFAT::outputType;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline RqUniformKvQuantSparseFlashAttentionMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace,
                                const RqUniformKvQuantSparseFlashAttentionTilingDataMla *__restrict tiling,
                                __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t QSFA_PRELOAD_TASK_CACHE_SIZE = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = QSFAVectorService<QSFAT>::BYTE_BLOCK / sizeof(T);

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint32_t msdIterNum = 2U;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const RqUniformKvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfo kvSfaKernelConstInfo{};
    TempLoopInfo kvSfaKernelLoopInfo{};

    QSFAMatmulService<QSFAT> matmulService;
    QSFAVectorService<QSFAT> kvSfaKernelVectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_ROPE_T> qRopeGm;
    GlobalTensor<K_ROPE_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitWorkspaceGlobalTensor(__gm__ uint8_t *workspace);
    __aicore__ inline void InitVectorService();
    __aicore__ inline void InitMatmulService();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[QSFA_PRELOAD_TASK_CACHE_SIZE]);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &kvSfaKernelRunInfo);
    __aicore__ inline void CalcMSizeInfo(RunInfo &kvSfaKernelRunInfo);
    __aicore__ inline void CalcFirstTensorOffsets(RunInfo &kvSfaKernelRunInfo, uint64_t qsfaActualSeqQPrefixSum,
                                                  uint64_t actualSeqKVPrefixSum);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void GetPreNextTokensLeftUp();
    __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &kvSfaKernelRunInfo);
    // ================================Mm2==============================================
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void ComputeMm2(const RunInfo &kvSfaKernelRunInfo);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
};

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitTilingData()
{
    usedCoreNum = tilingData->singleCoreParams.usedCoreNum;
    kvSfaKernelConstInfo.splitKVNum = tilingData->splitKVParams.s2;
    kvSfaKernelConstInfo.mmResUbSize = tilingData->singleCoreTensorSize.mmResUbSize;
    kvSfaKernelConstInfo.bmm2ResUbSize = tilingData->singleCoreTensorSize.bmm2ResUbSize;
    kvSfaKernelConstInfo.vec1ResUbSize = kvSfaKernelConstInfo.mmResUbSize * msdIterNum;

    kvSfaKernelConstInfo.qHeadNum = kvSfaKernelConstInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    kvSfaKernelConstInfo.batchSize = tilingData->baseParams.batchSize;
    kvSfaKernelConstInfo.kvSeqSize = tilingData->baseParams.seqSize;
    kvSfaKernelConstInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    kvSfaKernelConstInfo.maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvSfaKernelConstInfo.kvCacheBlockSize = tilingData->baseParams.blockSize;
    kvSfaKernelConstInfo.outputLayout = static_cast<QSFA_LAYOUT>(tilingData->baseParams.outputLayout);
    kvSfaKernelConstInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
    kvSfaKernelConstInfo.s2BaseSize = tilingData->innerSplitParams.s2BaseSize;
    kvSfaKernelConstInfo.kvHeadNum = kvHeadNum;
    kvSfaKernelConstInfo.headDim = headDim;
    kvSfaKernelConstInfo.headDimRope = headDimRope;
    kvSfaKernelConstInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    kvSfaKernelConstInfo.sparseBlockCount = tilingData->baseParams.sparseBlockCount;
    kvSfaKernelConstInfo.sparseMode = tilingData->baseParams.sparseMode;
    kvSfaKernelConstInfo.quantScaleRepoMode = QUANT_SCALE_REPO_MODE::COMBINE;
    kvSfaKernelConstInfo.attentionMode = ATTENTION_MODE::MLA_ABSORB;
    // Schema A fused 388B slot (256B u4 + 128B rope + 4B scale/offset).
    // Discriminator is dSizeVInput; this op rejects 386/TQ4 at tiling.
    kvSfaKernelConstInfo.keyQuantMode =
        (tilingData->baseParams.dSizeVInput == static_cast<int64_t>(sfa_kv::SLOT_BYTES))
            ? QUANT_MODE::SFA_SCHEMA_A
            : QUANT_MODE::PER_TILE;
    kvSfaKernelConstInfo.valueQuantMode = kvSfaKernelConstInfo.keyQuantMode;
    kvSfaKernelConstInfo.combineHeadDim =
        (kvSfaKernelConstInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) ? headDim + headDimRope : headDim;

    // constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    kvSfaKernelConstInfo.preLoadNum = PRELOAD_NUM;
    kvSfaKernelConstInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    kvSfaKernelConstInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    kvSfaKernelConstInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    kvSfaKernelConstInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    kvSfaKernelConstInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    kvSfaKernelConstInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        kvSfaKernelVectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                                               __gm__ uint8_t *actualSeqLengths)
{
    kvSfaKernelConstInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    kvSfaKernelConstInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;
    if (kvSfaKernelConstInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, kvSfaKernelConstInfo.actualLenDimsQ);
    }
    if (kvSfaKernelConstInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths, kvSfaKernelConstInfo.actualLenDimsKV);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx,
                                                                                uint32_t n2Idx)
{
    if (kvSfaKernelConstInfo.outputLayout == QSFA_LAYOUT::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = kvSfaKernelLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * kvSfaKernelConstInfo.gSize * headDim + // T轴、s1轴偏移
                                  n2Idx * kvSfaKernelConstInfo.gSize * headDim;                        // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], kvSfaKernelConstInfo.gSize * headDim, 0);
    } else if (kvSfaKernelConstInfo.outputLayout == QSFA_LAYOUT::BSND) {
        uint64_t attenOutOffset =
            bIdx * kvSfaKernelConstInfo.qSeqSize * kvHeadNum * kvSfaKernelConstInfo.gSize * headDim +
            s1Idx * kvHeadNum * kvSfaKernelConstInfo.gSize * headDim + // B轴、S1轴偏移
            n2Idx * kvSfaKernelConstInfo.gSize * headDim;              // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], kvSfaKernelConstInfo.gSize * headDim, 0);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitOutputSingleCore()
{
    uint32_t qsfaCoreNum = GetBlockNum();
    if (qsfaCoreNum != 0) {
        uint64_t qsfaTotalOutputSize = kvSfaKernelConstInfo.batchSize * kvSfaKernelConstInfo.qHeadNum *
                                       kvSfaKernelConstInfo.qSeqSize * kvSfaKernelConstInfo.headDim;
        // 2 means c:v = 1:2
        uint64_t qsfaSingleCoreSize = (qsfaTotalOutputSize + (2 * qsfaCoreNum) - 1) / (2 * qsfaCoreNum);
        uint64_t qsfaTailSize = qsfaTotalOutputSize - tmpBlockIdx * qsfaSingleCoreSize;
        uint64_t qsfaSingleInitOutputSize = qsfaTailSize < qsfaSingleCoreSize ? qsfaTailSize : qsfaSingleCoreSize;
        if (tmpBlockIdx * qsfaSingleCoreSize < qsfaTotalOutputSize && qsfaSingleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * qsfaSingleCoreSize], qsfaSingleInitOutputSize, 0);
        }
        SyncAll();
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    kvSfaKernelLoopInfo.curActualSeqLenOri = GetActualSeqLenKV(bIdx);
    kvSfaKernelLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx,
                                                                                    uint32_t n2Idx)
{
    if (kvSfaKernelLoopInfo.nextTokensPerBatch < 0 && s1Idx < (-kvSfaKernelLoopInfo.nextTokensPerBatch)) { // 存在行无效
        kvSfaKernelLoopInfo.curActualSeqLen = 0;
        return;
    }
    int64_t threshold = kvSfaKernelLoopInfo.curActualSeqLenOri;
    if (kvSfaKernelConstInfo.sparseMode == 3) {
        threshold = static_cast<int64_t>(kvSfaKernelLoopInfo.nextTokensPerBatch) + s1Idx + 1;
    }
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        kvSfaKernelLoopInfo.curActualSeqLen =
            (kvSfaKernelConstInfo.sparseBlockCount * kvSfaKernelConstInfo.sparseBlockSize > threshold) ?
                threshold :
                kvSfaKernelConstInfo.sparseBlockCount * kvSfaKernelConstInfo.sparseBlockSize;
    } else {
        uint64_t topKBaseOffset = 0;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) { // B,S1,N2 K
            topKBaseOffset = bIdx * kvSfaKernelConstInfo.qSeqSize * kvHeadNum * kvSfaKernelConstInfo.sparseBlockCount +
                             s1Idx * kvHeadNum * kvSfaKernelConstInfo.sparseBlockCount +
                             n2Idx * kvSfaKernelConstInfo.sparseBlockCount;
        } else if (LAYOUT_T == QSFA_LAYOUT::TND) { // T N2 K
            uint64_t actualSeqQPrefixSum = (bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
            topKBaseOffset = actualSeqQPrefixSum * kvHeadNum * kvSfaKernelConstInfo.sparseBlockCount +
                             s1Idx * kvHeadNum * kvSfaKernelConstInfo.sparseBlockCount +
                             n2Idx * kvSfaKernelConstInfo.sparseBlockCount;
        } else { // B N2 S1 K
            topKBaseOffset = bIdx * kvHeadNum * kvSfaKernelConstInfo.qSeqSize * kvSfaKernelConstInfo.sparseBlockCount +
                             n2Idx * kvSfaKernelConstInfo.qSeqSize * kvSfaKernelConstInfo.sparseBlockCount +
                             s1Idx * kvSfaKernelConstInfo.sparseBlockCount;
        }

        uint64_t sparseLen = 0;

        for (uint64_t topkIdx = 0; topkIdx < kvSfaKernelConstInfo.sparseBlockCount; topkIdx++) {
            int32_t sparseIndices = topKGm.GetValue(topKBaseOffset + topkIdx);
            uint64_t blockBegin = sparseIndices * kvSfaKernelConstInfo.sparseBlockSize;
            if (blockBegin >= threshold) {
                continue;
            }
            uint64_t blockEnd =
                (blockBegin + kvSfaKernelConstInfo.sparseBlockSize > kvSfaKernelLoopInfo.curActualSeqLenOri) ?
                    kvSfaKernelLoopInfo.curActualSeqLenOri :
                    blockBegin + kvSfaKernelConstInfo.sparseBlockSize;
            uint64_t blockLen = (blockEnd <= threshold) ? blockEnd - blockBegin : threshold - blockBegin;
            sparseLen += blockLen;
        }
        kvSfaKernelLoopInfo.curActualSeqLen = sparseLen;
    }
}

template <typename QSFAT>
__aicore__ inline uint32_t RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (KV_LAYOUT_T == QSFA_LAYOUT::TND) {
        if (bIdx > 0) {
            int32_t curActualSeqLen = actualSeqLengthsKVGm.GetValue(bIdx);
            int32_t prevActualSeqLen = actualSeqLengthsKVGm.GetValue(bIdx - 1);
            return (curActualSeqLen >= prevActualSeqLen) ? static_cast<uint32_t>(curActualSeqLen - prevActualSeqLen) :
                                                           0U;
        } else if (bIdx == 0) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (kvSfaKernelConstInfo.actualLenDimsKV == 0) {
            return kvSfaKernelConstInfo.kvSeqSize;
        } else if (kvSfaKernelConstInfo.actualLenDimsKV == 1) {
            return actualSeqLengthsKVGm.GetValue(0);
        } else {
            return actualSeqLengthsKVGm.GetValue(bIdx);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1Idx,
                                                                                  uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, s1Idx, n2Idx);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetPreNextTokensLeftUp()
{
    if (kvSfaKernelConstInfo.sparseMode == 3) {
        kvSfaKernelLoopInfo.nextTokensPerBatch = static_cast<int32_t>(kvSfaKernelLoopInfo.curActualSeqLenOri) -
                                                 static_cast<int32_t>(kvSfaKernelLoopInfo.actS1Size);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::UpdateInnerLoopCond()
{
    if ((kvSfaKernelLoopInfo.curActualSeqLen == 0) || (kvSfaKernelLoopInfo.actS1Size == 0)) {
        kvSfaKernelLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    kvSfaKernelLoopInfo.curActSeqLenIsZero = false;
    kvSfaKernelLoopInfo.s2BasicSizeTail = kvSfaKernelLoopInfo.curActualSeqLen % kvSfaKernelConstInfo.s2BaseSize;
    kvSfaKernelLoopInfo.s2BasicSizeTail = (kvSfaKernelLoopInfo.s2BasicSizeTail == 0) ?
                                              kvSfaKernelConstInfo.s2BaseSize :
                                              kvSfaKernelLoopInfo.s2BasicSizeTail;
    kvSfaKernelLoopInfo.mBasicSizeTail =
        (kvSfaKernelLoopInfo.actS1Size * kvSfaKernelConstInfo.gSize) % kvSfaKernelConstInfo.mBaseSize;
    kvSfaKernelLoopInfo.mBasicSizeTail =
        (kvSfaKernelLoopInfo.mBasicSizeTail == 0) ? kvSfaKernelConstInfo.mBaseSize : kvSfaKernelLoopInfo.mBasicSizeTail;
    kvSfaKernelLoopInfo.s2LoopTimes = 0;
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::UpdateInner(uint32_t &s2End, uint32_t &curS2End,
                                                                          uint32_t s1Idx, bool isEnd)
{
    uint32_t s1BaseSize = 1;
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2LastToken =
        Min(s1Offset + kvSfaKernelLoopInfo.nextTokensPerBatch + s1BaseSize, kvSfaKernelLoopInfo.curActualSeqLenOri);
    s2LastToken = Min(kvSfaKernelConstInfo.sparseBlockSize * kvSfaKernelConstInfo.sparseBlockCount, s2LastToken);
    curS2End = (s2LastToken + kvSfaKernelConstInfo.s2BaseSize - 1) / kvSfaKernelConstInfo.s2BaseSize;
    kvSfaKernelLoopInfo.s2LoopTimes = isEnd ? kvSfaKernelConstInfo.s2End + 1 : curS2End;
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *valueScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const RqUniformKvQuantSparseFlashAttentionTilingDataMla *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIC {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    } else {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 初始化计算参数
    InitCalcParamsEach();
    keyPtr = key;
    valuePtr = value;
    pipe = tPipe;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)keyPtr);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)valuePtr);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    if ASCEND_IS_AIV {
        if (kvSfaKernelConstInfo.needInit && LAYOUT_T != QSFA_LAYOUT::TND) {
            InitOutputSingleCore();
        }
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);

    InitWorkspaceGlobalTensor(workspace);

    InitVectorService();
    InitMatmulService();

    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitWorkspaceGlobalTensor(__gm__ uint8_t *workspace)
{
    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t qsfaOffset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + qsfaOffset +
                             aiCoreIdx * dbWorkspaceRatio * kvSfaKernelConstInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * kvSfaKernelConstInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ K_ROPE_T *)(workspace + qsfaOffset +
                            aiCoreIdx * dbWorkspaceRatio * kvSfaKernelConstInfo.mmResUbSize * sizeof(K_ROPE_T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * kvSfaKernelConstInfo.mmResUbSize * sizeof(K_ROPE_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + qsfaOffset +
                             aiCoreIdx * dbWorkspaceRatio * kvSfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * kvSfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    vec2ResGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + qsfaOffset +
                     aiCoreIdx * dbWorkspaceRatio * kvSfaKernelConstInfo.bmm2ResUbSize * sizeof(T)));
    qsfaOffset += GetBlockNum() * dbWorkspaceRatio * kvSfaKernelConstInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // s2  d+rope bufNum
        uint64_t kvMergeBytesPerCore = 512 * kvSfaKernelConstInfo.combineHeadDim * 4 * sizeof(K_ROPE_T);
        kvMergeGm_.SetGlobalBuffer((__gm__ K_ROPE_T *)(workspace + qsfaOffset + aiCoreIdx * kvMergeBytesPerCore));
        qsfaOffset += GetBlockNum() * kvMergeBytesPerCore;

        // 每核 4 份：2 份 AIV 有效 size 分区 + 等大的 TQ4 逐 token FP16 scale 区
        kvValidSizeGm_.SetGlobalBuffer(
            (__gm__ int32_t *)(workspace + qsfaOffset + (aiCoreIdx * 4) * 128 * 4 * sizeof(int32_t)));
    }

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + qsfaOffset));
        qsfaOffset = qsfaOffset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + qsfaOffset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + qsfaOffset) +
                                   tilingData->splitKVParams.logSumExpSize / 2);
        qsfaOffset = qsfaOffset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitVectorService()
{
    if ASCEND_IS_AIV {
        kvSfaKernelVectorService.InitParams(kvSfaKernelConstInfo, tilingData);
        kvSfaKernelVectorService.InitMm2ResInt32GmGlobalTensor(mm2ResInt32Gm);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            kvSfaKernelVectorService.InitVec0GlobalTensor(kvValidSizeGm_, kvMergeGm_, kRopeGm, keyGm, blockTableGm);
        }
        kvSfaKernelVectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm, actualSeqLengthsKVGm,
                                                      lseMaxFdGm, lseSumFdGm, topKGm);
        kvSfaKernelVectorService.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitMatmulService()
{
    if ASCEND_IS_AIC {
        matmulService.InitParams(kvSfaKernelConstInfo);
        matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        matmulService.InitPageAttentionInfo(kvMergeGm_, blockTableGm, topKGm, kvSfaKernelConstInfo.kvCacheBlockSize,
                                            kvSfaKernelConstInfo.maxBlockNumPerBatch);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::InitCalcParamsEach()
{
    // 计算总的基本块
    uint32_t qsfaTotalBaseNum = 0;
    uint32_t qsfaS1GBaseSize = kvSfaKernelConstInfo.gSize;
    uint32_t qsfaActBatchS2 = 1;
    uint32_t qsfaCoreNum = GetBlockNum();
    uint32_t qsfaActBatchS1 = 1;
    uint32_t qsfaCurrCoreIdx = aiCoreIdx;
    for (uint32_t qsfaBIdx = 0; qsfaBIdx < kvSfaKernelConstInfo.batchSize; qsfaBIdx++) {
        uint32_t qsfaActBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBIdx);
        if (qsfaActBatchS1 < kvSfaKernelConstInfo.qSeqSize) {
            kvSfaKernelConstInfo.needInit = true;
        }
        qsfaTotalBaseNum += qsfaActBatchS1 * qsfaActBatchS2;
    }
    uint32_t qsfaAvgBaseNum = 1;
    if (qsfaTotalBaseNum > qsfaCoreNum) {
        qsfaAvgBaseNum = (qsfaTotalBaseNum + qsfaCoreNum - 1) / qsfaCoreNum;
    } else {
        usedCoreNum = qsfaTotalBaseNum;
    }
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
    // 计算当前核的基本块
    uint32_t qsfaAccumBaseNum = 0; // 当前累积的基本块数
    uint32_t qsfaTargetBaseNum = 0;
    uint32_t qsfaLastValidBIdx = 0;
    uint32_t qsfaLastValidactBatchS1 = 0;
    bool qsfaSetStart = false;
    qsfaTargetBaseNum = (qsfaCurrCoreIdx + 1) * qsfaAvgBaseNum; // 计算当前的目标权重
    uint32_t qsfaTargetStartBaseNum = qsfaTargetBaseNum - qsfaAvgBaseNum;
    for (uint32_t qsfaBN2Idx = 0; qsfaBN2Idx < kvSfaKernelConstInfo.batchSize * kvSfaKernelConstInfo.kvHeadNum;
         qsfaBN2Idx++) {
        uint32_t qsfaBIdx = qsfaBN2Idx / kvSfaKernelConstInfo.kvHeadNum;
        qsfaActBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBIdx);
        for (uint32_t qsfaS1GIdx = 0; qsfaS1GIdx < qsfaActBatchS1; qsfaS1GIdx++) {
            qsfaAccumBaseNum += 1;
            if (!qsfaSetStart && qsfaAccumBaseNum >= qsfaTargetStartBaseNum) {
                kvSfaKernelConstInfo.bN2Start = qsfaBN2Idx;
                kvSfaKernelConstInfo.gS1Start = qsfaS1GIdx;
                qsfaSetStart = true;
            }
            if (qsfaAccumBaseNum >= qsfaTargetBaseNum) {
                // 更新当前核的End分核信息
                kvSfaKernelConstInfo.bN2End = qsfaBN2Idx;
                kvSfaKernelConstInfo.gS1End = qsfaS1GIdx;
                kvSfaKernelConstInfo.coreStartKVSplitPos = 0;
                kvSfaKernelConstInfo.s2End = 0;
                if (aiCoreIdx != 0) {
                    GetAxisStartIdx(kvSfaKernelConstInfo.bN2Start, kvSfaKernelConstInfo.gS1Start, 0);
                }
                return;
            }
        }
        if ((qsfaActBatchS1 > 0) && (qsfaActBatchS2 > 0)) {
            qsfaLastValidactBatchS1 = qsfaActBatchS1;
            qsfaLastValidBIdx = qsfaBIdx;
        }
    }
    if (!qsfaSetStart) {
        kvSfaKernelConstInfo.bN2Start = qsfaLastValidBIdx;
        kvSfaKernelConstInfo.gS1Start = qsfaLastValidactBatchS1 - 1;
    }
    if (qsfaAccumBaseNum < qsfaTargetBaseNum) {
        // 更新最后一个核的End分核信息
        kvSfaKernelConstInfo.bN2End = qsfaLastValidBIdx;
        kvSfaKernelConstInfo.gS1End = qsfaLastValidactBatchS1 - 1;
        kvSfaKernelConstInfo.s2End = 0;
        kvSfaKernelConstInfo.coreStartKVSplitPos = 0;
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(kvSfaKernelConstInfo.bN2Start, kvSfaKernelConstInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset,
                                                                              LocalTensor<OUT_T> &attenOutUb,
                                                                              uint32_t startRow, uint32_t dealRowCount,
                                                                              uint32_t columnCount,
                                                                              uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride =
        (columnCount - actualColumnCount) / (QSFAVectorService<QSFAT>::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::CalcParams(uint32_t loop, uint64_t s2Start,
                                                                         uint32_t s2LoopIdx,
                                                                         RunInfo &kvSfaKernelRunInfo)
{
    kvSfaKernelRunInfo.loop = loop;
    kvSfaKernelRunInfo.bIdx = kvSfaKernelLoopInfo.bIdx;
    kvSfaKernelRunInfo.gS1Idx = kvSfaKernelLoopInfo.gS1Idx;
    kvSfaKernelRunInfo.s2Idx = s2LoopIdx;
    kvSfaKernelRunInfo.curSInnerLoopTimes = kvSfaKernelLoopInfo.s2LoopTimes;

    kvSfaKernelRunInfo.isBmm2Output = false;
    kvSfaKernelRunInfo.tndIsS2SplitCore = kvSfaKernelLoopInfo.tndIsS2SplitCore;
    kvSfaKernelRunInfo.tndCoreStartKVSplitPos = kvSfaKernelLoopInfo.tndCoreStartKVSplitPos;

    kvSfaKernelRunInfo.actS1Size = kvSfaKernelLoopInfo.actS1Size;
    kvSfaKernelRunInfo.actS2Size = kvSfaKernelLoopInfo.curActualSeqLen;

    kvSfaKernelRunInfo.actMBaseSize = kvSfaKernelConstInfo.mBaseSize;
    uint32_t qsfaRemainedGS1Size =
        kvSfaKernelLoopInfo.actS1Size * kvSfaKernelConstInfo.gSize - kvSfaKernelLoopInfo.gS1Idx;
    if (qsfaRemainedGS1Size <= kvSfaKernelConstInfo.mBaseSize && qsfaRemainedGS1Size > 0) {
        kvSfaKernelRunInfo.actMBaseSize = kvSfaKernelLoopInfo.mBasicSizeTail;
    }

    kvSfaKernelRunInfo.isValid = s2LoopIdx < kvSfaKernelLoopInfo.s2LoopTimes;
    CalcMSizeInfo(kvSfaKernelRunInfo);

    kvSfaKernelRunInfo.isChangeBatch = false;

    kvSfaKernelRunInfo.isFirstSInnerLoop = (s2LoopIdx == s2Start);
    if (kvSfaKernelRunInfo.isFirstSInnerLoop) {
        kvSfaKernelLoopInfo.bn2IdxInCurCore++;
    }
    kvSfaKernelRunInfo.isLastS2Loop = (s2LoopIdx == kvSfaKernelLoopInfo.s2LoopTimes - 1);
    kvSfaKernelRunInfo.bn2IdxInCurCore = kvSfaKernelLoopInfo.bn2IdxInCurCore - 1;
    uint64_t qsfaActualSeqQPrefixSum;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        qsfaActualSeqQPrefixSum =
            (kvSfaKernelRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(kvSfaKernelRunInfo.bIdx - 1);
    } else {
        qsfaActualSeqQPrefixSum =
            (kvSfaKernelRunInfo.bIdx <= 0) ? 0 : kvSfaKernelRunInfo.bIdx * kvSfaKernelConstInfo.qSeqSize;
    }
    kvSfaKernelRunInfo.tndBIdxOffsetForQ =
        qsfaActualSeqQPrefixSum * kvSfaKernelConstInfo.qHeadNum * kvSfaKernelConstInfo.combineHeadDim;

    uint64_t actualSeqKVPrefixSum;
    if constexpr (KV_LAYOUT_T == QSFA_LAYOUT::TND) {
        actualSeqKVPrefixSum =
            (kvSfaKernelRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsKVGm.GetValue(kvSfaKernelRunInfo.bIdx - 1);
    } else {
        actualSeqKVPrefixSum =
            (kvSfaKernelRunInfo.bIdx <= 0) ? 0 : kvSfaKernelRunInfo.bIdx * kvSfaKernelConstInfo.kvSeqSize;
    }
    kvSfaKernelRunInfo.tndBIdxOffsetForKV =
        actualSeqKVPrefixSum * kvSfaKernelConstInfo.kvHeadNum * kvSfaKernelConstInfo.combineHeadDim;

    CalcFirstTensorOffsets(kvSfaKernelRunInfo, qsfaActualSeqQPrefixSum, actualSeqKVPrefixSum);

    uint64_t sInnerOffsetDataSize = kvSfaKernelRunInfo.s2Idx * kvSfaKernelConstInfo.s2BaseSize;
    kvSfaKernelRunInfo.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;

    kvSfaKernelRunInfo.curActualSeqLenOri = kvSfaKernelLoopInfo.curActualSeqLenOri;
    if (kvSfaKernelLoopInfo.curActualSeqLen > sInnerOffsetDataSize) {
        kvSfaKernelRunInfo.actualSingleProcessSInnerSize = kvSfaKernelLoopInfo.curActualSeqLen - sInnerOffsetDataSize;
        kvSfaKernelRunInfo.actualSingleProcessSInnerSize =
            kvSfaKernelRunInfo.actualSingleProcessSInnerSize > kvSfaKernelConstInfo.s2BaseSize ?
                kvSfaKernelConstInfo.s2BaseSize :
                kvSfaKernelRunInfo.actualSingleProcessSInnerSize;
    } else {
        kvSfaKernelRunInfo.actualSingleProcessSInnerSize = 0;
    }
    kvSfaKernelRunInfo.actualSingleProcessSInnerSizeAlign = QSFAAlign(
        (uint32_t)kvSfaKernelRunInfo.actualSingleProcessSInnerSize, (uint32_t)QSFAVectorService<QSFAT>::BYTE_BLOCK);
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::CalcMSizeInfo(RunInfo &kvSfaKernelRunInfo)
{
    if ASCEND_IS_AIV {
        kvSfaKernelRunInfo.mSize = kvSfaKernelRunInfo.actMBaseSize;
        kvSfaKernelRunInfo.mSizeV = (kvSfaKernelRunInfo.mSize <= 16) ?
                                        kvSfaKernelRunInfo.mSize :
                                        (((kvSfaKernelRunInfo.mSize + 15) / 16 + 1) / 2 * 16);
        kvSfaKernelRunInfo.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            kvSfaKernelRunInfo.mSizeVStart = kvSfaKernelRunInfo.mSizeV;
            kvSfaKernelRunInfo.mSizeV = kvSfaKernelRunInfo.mSize - kvSfaKernelRunInfo.mSizeV;
        }
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::CalcFirstTensorOffsets(RunInfo &kvSfaKernelRunInfo,
                                                                                     uint64_t qsfaActualSeqQPrefixSum,
                                                                                     uint64_t actualSeqKVPrefixSum)
{
    if (kvSfaKernelRunInfo.isFirstSInnerLoop) {
        tensorACoreOffset =
            kvSfaKernelRunInfo.tndBIdxOffsetForQ + kvSfaKernelRunInfo.gS1Idx * kvSfaKernelConstInfo.combineHeadDim;
        tensorBCoreOffset =
            kvSfaKernelRunInfo.tndBIdxOffsetForKV + kvSfaKernelRunInfo.n2Idx * kvSfaKernelConstInfo.combineHeadDim;
        if (kvSfaKernelConstInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            attenOutOffset =
                (qsfaActualSeqQPrefixSum * kvSfaKernelConstInfo.qHeadNum + kvSfaKernelRunInfo.gS1Idx) * headDim;
        } else {
            uint64_t tndBIdxRopeOffsetForQ = qsfaActualSeqQPrefixSum * kvSfaKernelConstInfo.qHeadNum * headDimRope;
            tensorARopeCoreOffset = tndBIdxRopeOffsetForQ + kvSfaKernelRunInfo.gS1Idx * headDimRope;
            uint64_t tndBIdxRopeOffsetForK = actualSeqKVPrefixSum * kvSfaKernelConstInfo.kvHeadNum * headDimRope;
            tensorBRopeCoreOffset = tndBIdxRopeOffsetForK + kvSfaKernelRunInfo.n2Idx * headDimRope;
            attenOutOffset = tensorACoreOffset;
        }
        if (kvSfaKernelConstInfo.sparseMode == 3) {
            threshold = static_cast<int64_t>(kvSfaKernelLoopInfo.nextTokensPerBatch) +
                        kvSfaKernelRunInfo.gS1Idx / kvSfaKernelConstInfo.gSize + 1;
        } else {
            threshold = kvSfaKernelLoopInfo.curActualSeqLenOri;
        }
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
            topKBaseOffset = kvSfaKernelRunInfo.bIdx * kvSfaKernelConstInfo.qSeqSize * kvSfaKernelConstInfo.kvHeadNum *
                                 kvSfaKernelConstInfo.sparseBlockCount +
                             kvSfaKernelRunInfo.gS1Idx / kvSfaKernelConstInfo.gSize * kvSfaKernelConstInfo.kvHeadNum *
                                 kvSfaKernelConstInfo.sparseBlockCount +
                             kvSfaKernelRunInfo.n2Idx * kvSfaKernelConstInfo.sparseBlockCount;
        } else if (LAYOUT_T == QSFA_LAYOUT::TND) {
            topKBaseOffset = kvSfaKernelRunInfo.tndBIdxOffsetForQ / kvSfaKernelConstInfo.gSize /
                                 kvSfaKernelConstInfo.combineHeadDim * kvSfaKernelConstInfo.kvHeadNum *
                                 kvSfaKernelConstInfo.sparseBlockCount +
                             kvSfaKernelRunInfo.n2Idx * kvSfaKernelConstInfo.sparseBlockCount +
                             kvSfaKernelRunInfo.gS1Idx / kvSfaKernelConstInfo.gSize * kvSfaKernelConstInfo.kvHeadNum *
                                 kvSfaKernelConstInfo.sparseBlockCount;
        } else {
            topKBaseOffset =
                kvSfaKernelRunInfo.bIdx * kvSfaKernelConstInfo.kvHeadNum * kvSfaKernelConstInfo.qSeqSize *
                    kvSfaKernelConstInfo.sparseBlockCount +
                kvSfaKernelRunInfo.n2Idx * kvSfaKernelConstInfo.qSeqSize * kvSfaKernelConstInfo.sparseBlockCount +
                kvSfaKernelRunInfo.gS1Idx / kvSfaKernelConstInfo.gSize * kvSfaKernelConstInfo.sparseBlockCount;
        }
    }
    kvSfaKernelRunInfo.topKBaseOffset = topKBaseOffset;
    kvSfaKernelRunInfo.threshold = threshold;
    kvSfaKernelRunInfo.tensorAOffset = tensorACoreOffset;
    kvSfaKernelRunInfo.tensorARopeOffset = tensorARopeCoreOffset;
    kvSfaKernelRunInfo.tensorBOffset = tensorBCoreOffset;
    kvSfaKernelRunInfo.tensorBRopeOffset = tensorBRopeCoreOffset;
    kvSfaKernelRunInfo.attenOutOffset = attenOutOffset;
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::ComputeMm1(const RunInfo &kvSfaKernelRunInfo)
{
    uint32_t nBufferLoopTimes = (kvSfaKernelRunInfo.actMBaseSize + kvSfaKernelConstInfo.nBufferMBaseSize - 1) /
                                kvSfaKernelConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail =
        kvSfaKernelRunInfo.actMBaseSize - (nBufferLoopTimes - 1) * kvSfaKernelConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo kvSfaKernelSplitInfo;
        kvSfaKernelSplitInfo.nBufferStartM = i * kvSfaKernelConstInfo.nBufferMBaseSize;
        kvSfaKernelSplitInfo.nBufferDealM =
            (i + 1 != nBufferLoopTimes) ? kvSfaKernelConstInfo.nBufferMBaseSize : nBufferTail;
        matmulService.ComputeMm1(kvSfaKernelRunInfo, kvSfaKernelSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(kvSfaKernelConstInfo.syncC1V1);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::ComputeMm2(const RunInfo &kvSfaKernelRunInfo)
{
    uint32_t nBufferLoopTimes = (kvSfaKernelRunInfo.actMBaseSize + kvSfaKernelConstInfo.nBufferMBaseSize - 1) /
                                kvSfaKernelConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail =
        kvSfaKernelRunInfo.actMBaseSize - (nBufferLoopTimes - 1) * kvSfaKernelConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        // CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
        MSplitInfo kvSfaKernelSplitInfo;
        kvSfaKernelSplitInfo.nBufferStartM = i * kvSfaKernelConstInfo.nBufferMBaseSize;
        kvSfaKernelSplitInfo.nBufferDealM =
            (i + 1 != nBufferLoopTimes) ? kvSfaKernelConstInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(kvSfaKernelConstInfo.syncV1C2);
        matmulService.ComputeMm2(kvSfaKernelRunInfo, kvSfaKernelSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(kvSfaKernelConstInfo.syncC2V2);
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIC {
            matmulService.AllocEventID();
        } else {
            kvSfaKernelVectorService.AllocEventID();
            kvSfaKernelVectorService.InitSoftmaxDefaultBuffer();
        }
        ProcessBalance();

        if ASCEND_IS_AIC {
            matmulService.FreeEventID();
        } else {
            kvSfaKernelVectorService.FreeEventID();
        }
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx,
                                                                        uint32_t &n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::ProcessBalance()
{
    RunInfo extraInfo[QSFA_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        // CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    for (uint32_t qsfaBN2LoopIdx = kvSfaKernelConstInfo.bN2Start; qsfaBN2LoopIdx <= kvSfaKernelConstInfo.bN2End;
         qsfaBN2LoopIdx++) {
        GetBN2Idx(qsfaBN2LoopIdx, kvSfaKernelLoopInfo.bIdx, kvSfaKernelLoopInfo.n2Idx);
        GetActualSeqLen(kvSfaKernelLoopInfo.bIdx); // 获取actualSeqLength及ActualSeqLengthKV
        GetPreNextTokensLeftUp();
        if (kvSfaKernelLoopInfo.actS1Size == 0) {
            continue;
        }
        int gS1SplitNum =
            (kvSfaKernelLoopInfo.actS1Size * kvSfaKernelConstInfo.gSize + kvSfaKernelConstInfo.mBaseSize - 1) /
            kvSfaKernelConstInfo.mBaseSize;
        gS1LoopEnd = (qsfaBN2LoopIdx == kvSfaKernelConstInfo.bN2End) ? kvSfaKernelConstInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t qsfaGS1LoopIdx = kvSfaKernelConstInfo.gS1Start; qsfaGS1LoopIdx <= gS1LoopEnd; qsfaGS1LoopIdx++) {
            kvSfaKernelLoopInfo.gS1Idx = qsfaGS1LoopIdx * kvSfaKernelConstInfo.mBaseSize;
            // TopK值sparse完后的ActualSeqLengthKV
            GetSparseActualSeqLen(kvSfaKernelLoopInfo.bIdx, qsfaGS1LoopIdx, kvSfaKernelLoopInfo.n2Idx);
            UpdateInnerLoopCond();

            if (kvSfaKernelLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(kvSfaKernelLoopInfo.bIdx, qsfaGS1LoopIdx, kvSfaKernelLoopInfo.n2Idx);
            }
            int s2SplitNum = (kvSfaKernelLoopInfo.curActualSeqLen + kvSfaKernelConstInfo.s2BaseSize - 1) /
                             kvSfaKernelConstInfo.s2BaseSize; // S2切分份数
            bool qsfaIsEnd =
                (qsfaBN2LoopIdx == kvSfaKernelConstInfo.bN2End) && (qsfaGS1LoopIdx == kvSfaKernelConstInfo.gS1End);
            kvSfaKernelLoopInfo.s2LoopTimes = s2SplitNum;
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            kvSfaKernelLoopInfo.tndIsS2SplitCore =
                ((kvSfaKernelConstInfo.s2Start == 0) && (kvSfaKernelLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            kvSfaKernelLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? kvSfaKernelConstInfo.coreStartKVSplitPos : 0;
            uint32_t qsfaExtraLoop = qsfaIsEnd ? 2 : 0;
            for (int s2LoopIdx = kvSfaKernelConstInfo.s2Start;
                 s2LoopIdx < (kvSfaKernelLoopInfo.s2LoopTimes + qsfaExtraLoop); s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, kvSfaKernelConstInfo.s2Start, s2LoopIdx, extraInfo);
                ++gloop;
            }
            globalLoopStart = false;
            kvSfaKernelConstInfo.s2Start = 0;
        }
        kvSfaKernelConstInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
            CrossCoreWaitFlag(3);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::PreloadPipeline(
    uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx, RunInfo extraInfo[QSFA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % QSFA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % QSFA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % QSFA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(kvSfaKernelConstInfo.syncV0C1);
            }
            ComputeMm1(extraInfo0);
        } else {
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreWaitFlag(3);
                kvSfaKernelVectorService.MergeKv(extraInfo0);
                CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(kvSfaKernelConstInfo.syncV0C1);
            }
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            kvSfaKernelVectorService.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
            if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE2>(3);
            }
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            kvSfaKernelVectorService.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

template <typename QSFAT>
__aicore__ inline uint64_t RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetBalanceActualSeqLengths(
    GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx)
{
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        if (bIdx > 0) {
            int32_t curActualSeqLen = actualSeqLengths.GetValue(bIdx);
            int32_t prevActualSeqLen = actualSeqLengths.GetValue(bIdx - 1);
            return (curActualSeqLen >= prevActualSeqLen) ? static_cast<uint64_t>(curActualSeqLen - prevActualSeqLen) :
                                                           0ULL;
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (kvSfaKernelConstInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else if (kvSfaKernelConstInfo.actualLenDimsQ == 0) {
            return kvSfaKernelConstInfo.qSeqSize;
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void RqUniformKvQuantSparseFlashAttentionMla<QSFAT>::GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev,
                                                                              uint32_t s2EndPrev)
{
    uint32_t qsfaBEndPrev = bN2EndPrev / kvHeadNum;
    uint32_t qsfaActualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, qsfaBEndPrev);
    uint32_t qsfaS1GPrevBaseNum =
        (qsfaActualSeqQPrev * kvSfaKernelConstInfo.gSize + kvSfaKernelConstInfo.mBaseSize - 1) /
        kvSfaKernelConstInfo.mBaseSize;
    kvSfaKernelConstInfo.bN2Start = bN2EndPrev;
    kvSfaKernelConstInfo.gS1Start = s1GEndPrev;

    kvSfaKernelConstInfo.s2Start = 0;
    if (s1GEndPrev >= qsfaS1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        kvSfaKernelConstInfo.gS1Start = 0;
        kvSfaKernelConstInfo.bN2Start++;
    } else {
        kvSfaKernelConstInfo.gS1Start++;
    }
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
