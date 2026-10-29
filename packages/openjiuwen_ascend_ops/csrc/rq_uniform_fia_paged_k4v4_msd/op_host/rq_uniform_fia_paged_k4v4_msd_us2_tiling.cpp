/*!
 * \file rq_uniform_fia_paged_k4v4_msd_us2_tiling.cpp
 * \brief
 */

/*
 * Host tiling for RqUniformFiaPagedK4v4Msd Us2 path (P0-op).
 *
 * Decode-only subset of IFA FillTiling (ops-transformer incre_flash_attention_tiling.cpp):
 * PA + GQA + MSD (msdIterNum=2). PACK_GATHER=0: Us2-native PA int8, PER_CHANNEL.
 * PACK_GATHER=1: PER_TOKEN flow (Mode1 QueryPreProcessPerToken) + Scheme A
 * pack meta; antiqSeqSize / per-head / paged-meta flags match Mode5 PA gather.
 */

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "rq_uniform_fia_paged_k4v4_msd_us2_tiling.h"
#include "vendored/us2/incre_flash_attention_tilingkey.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace optiling {
namespace {

constexpr uint32_t kUbBlockBytes = 32;
constexpr uint32_t kMsdIters = 2;
constexpr uint32_t kAntiqSInnerGqa = 1024;
constexpr uint32_t kCoreCap = 50;
constexpr uint32_t kMmBaseCap = 512;
constexpr uint32_t kMmBaseMCap = 128;
constexpr uint32_t kL0cBytes = 128 * 1024;
constexpr uint32_t kL0bBytes = 64 * 1024;
constexpr uint32_t kPaCbBytes = 64;
constexpr uint32_t kLibApiWsFallback = 16 * 1024 * 1024;
constexpr uint32_t kFdKvMinSplit = 512;
constexpr uint32_t kPaNoFdKvMax = 1024;
constexpr uint32_t kFdGqaKvMin = 2048;
constexpr float kFdBnRatioTh = 0.4F;
#if BR_MSD_US2_PACK_GATHER
constexpr uint64_t kKeyPaged =
    QF16_KVINT4_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V2_TILING;
constexpr uint64_t kKeyPagedC1V1 =
    QF16_KVINT4_OUTF16_ANTIPERTOKEN_PAGEDCACHE_C1V1_TILING;
constexpr uint64_t kKeyPagedFd =
    QF16_KVINT4_OUTF16_ANTIPERTOKEN_PAGEDCACHE_FLASHDECODING_C1V2_TILING;
#else
constexpr uint64_t kKeyPaged =
    QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V2_TILING;
constexpr uint64_t kKeyPagedC1V1 =
    QF16_KVINT4_OUTF16_ANTIPERTOKEN_C1V1_TILING;
constexpr uint64_t kKeyPagedFd =
    QF16_KVINT4_OUTF16_ANTIPERTOKEN_FLASHDECODING_C1V2_TILING;
#endif

bool SetupBmm2Tiling(const platform_ascendc::PlatformAscendC& platform, uint32_t m, uint32_t sInner,
                    uint32_t sInnerAlign, uint32_t seqSize, uint32_t headDim, uint32_t blockSize,
                    IncreFlashAttentionTilingData& tiling)
{
    matmul_tiling::MatmulApiTiling bmm2(platform);
#if BR_MSD_US2_PACK_GATHER
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT4, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT4, false);
#else
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT8, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT8, false);
#endif
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                  matmul_tiling::DataType::DT_INT32);
    bmm2.SetShape(m, headDim, sInner);
    bmm2.SetOrgShape(m, headDim, sInnerAlign, seqSize);
    bmm2.SetBias(false);
    if (bmm2.SetFixSplit(std::min(RoundUpU32(m, 16U), kMmBaseMCap)) == -1) {
        return false;
    }
    if (bmm2.GetTiling(tiling.bmm2TilingData) == -1) {
        return false;
    }
    uint32_t targetBaseK = 128U;
    if (targetBaseK < blockSize) {
        while ((blockSize % targetBaseK != 0U) ||
               (targetBaseK * tiling.bmm2TilingData.get_baseN() * sizeof(float) > kL0bBytes)) {
            targetBaseK /= 2U;
            if (targetBaseK < 16U) {
                break;
            }
        }
    } else {
        uint32_t tmpBaseK = GreatestCommonDiv(targetBaseK, blockSize);
        while (tmpBaseK * tiling.bmm2TilingData.get_baseN() * sizeof(float) > kL0bBytes &&
               tmpBaseK >= 16U) {
            tmpBaseK /= 2U;
        }
        targetBaseK = tmpBaseK;
    }
    tiling.bmm2TilingData.set_baseK(targetBaseK);
    return true;
}
bool SetupBmmPiTiling(const platform_ascendc::PlatformAscendC& platform, uint32_t m,
                     uint32_t headDim, matmul_tiling::DataType qDtype,
                     IncreFlashAttentionTilingDataV2& tilingV2)
{
    matmul_tiling::MatmulApiTiling bmmPi(platform);
    bmmPi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   qDtype, false);
    bmmPi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   qDtype, false);
    bmmPi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_FLOAT);
    bmmPi.SetShape(m, headDim, headDim);
    bmmPi.SetOrgShape(m, headDim, headDim);
    bmmPi.SetBias(false);
    const uint32_t baseM = std::min(RoundUpU32(m, 16U), kMmBaseMCap);
    const uint32_t baseN = RoundUpU32(headDim, 16U);
    if (bmmPi.SetFixSplit(baseM, baseN) == -1) {
        return false;
    }
    if (bmmPi.GetTiling(tilingV2.bmmPiTilingData) == -1) {
        return false;
    }
    return true;
}

static ge::graphStatus RqUniformFiaPagedK4v4MsdUs2TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (attrs == nullptr || platformInfoPtr == nullptr) {
        OPS_LOG_E(nodeName, "attrs or platformInfo is null");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (aicNum == 0U || aivNum == 0U) {
        OPS_LOG_E(nodeName, "invalid core nums aic=%u aiv=%u", aicNum, aivNum);
        return ge::GRAPH_FAILED;
    }
    const uint32_t cvRatio = aivNum / aicNum;
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (libapiSize == 0) {
        libapiSize = kLibApiWsFallback;
    }

    const int64_t* numHeadsPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* headSizePtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(3);
    const float* scaleValuePtr = attrs->GetAttrPointer<float>(4);
    if (numHeadsPtr == nullptr || numKvHeadsPtr == nullptr || headSizePtr == nullptr ||
        blockSizePtr == nullptr || scaleValuePtr == nullptr) {
        OPS_LOG_E(nodeName, "required attrs are null");
        return ge::GRAPH_FAILED;
    }
    const int64_t numHeads = *numHeadsPtr;
    const int64_t numKvHeads = *numKvHeadsPtr;
    const int64_t headSize = *headSizePtr;
    const int64_t blockSize = *blockSizePtr;
    if (headSize != 128) {
        OPS_LOG_E(nodeName, "Us2 P0 only supports head_size=128, got %ld", headSize);
        return ge::GRAPH_FAILED;
    }
    if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0 || blockSize <= 0 ||
        (blockSize % 16) != 0) {
        OPS_LOG_E(nodeName, "invalid head/block attrs");
        return ge::GRAPH_FAILED;
    }
    const uint32_t gSize = static_cast<uint32_t>(numHeads / numKvHeads);
    const uint32_t headDim = static_cast<uint32_t>(headSize);
    const uint32_t headDimAlign = RoundUpU32(headDim, 16U);
    const uint32_t blockSizeU = static_cast<uint32_t>(blockSize);

    auto qShapePtr = context->GetInputShape(0);
    auto keyCacheShapePtr = context->GetInputShape(1);
    auto btShapePtr = context->GetInputShape(3);
    auto seqQTensor = context->GetInputTensor(4);
    auto seqKvTensor = context->GetInputTensor(5);
    if (qShapePtr == nullptr || keyCacheShapePtr == nullptr || btShapePtr == nullptr ||
        seqQTensor == nullptr || seqKvTensor == nullptr) {
        OPS_LOG_E(nodeName, "required inputs missing");
        return ge::GRAPH_FAILED;
    }

    // Query dtype selects the compiled kernel variant (ORIG_DTYPE_QUERY);
    // bf16/bf16 pack meta + rotation Π are read as Q_T there. bmmPi Cube
    // tiling must declare the same dtype as the Q_T Matmul instantiation.
    const ge::DataType qGeDtype = context->GetInputDesc(0)->GetDataType();
    matmul_tiling::DataType qDtype;
    if (qGeDtype == ge::DT_FLOAT16) {
        qDtype = matmul_tiling::DataType::DT_FLOAT16;
    } else if (qGeDtype == ge::DT_BF16) {
        qDtype = matmul_tiling::DataType::DT_BF16;
    } else {
        OPS_LOG_E(nodeName, "Us2 query dtype must be fp16 or bf16, got %d",
                  static_cast<int>(qGeDtype));
        return ge::GRAPH_FAILED;
    }
    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape keyCacheShape = keyCacheShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (btShape.GetDimNum() < 2 || keyCacheShape.GetDimNum() < 2) {
        OPS_LOG_E(nodeName, "unexpected ranks query=%zu key=%zu block_table=%zu", qShape.GetDimNum(),
                  keyCacheShape.GetDimNum(), btShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = btShape.GetDim(0);
    const int64_t maxBlocksPerSeq = btShape.GetDim(1);
    const int64_t totalBlockNum = keyCacheShape.GetDim(0);
    if (batchSize <= 0 || maxBlocksPerSeq <= 0 || totalBlockNum <= 0) {
        OPS_LOG_E(nodeName, "invalid batch / block_table / cache shape");
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens = qShape.GetDim(0);
    const int64_t* seqQHost = seqQTensor->GetData<int64_t>();
    const int64_t* seqKvHost = seqKvTensor->GetData<int64_t>();
    if (seqQHost == nullptr || seqKvHost == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len tensors must be host-value-dependent");
        return ge::GRAPH_FAILED;
    }

    int64_t maxKvSeq = 0;
    uint32_t qSeqSize = 0U;
    const bool accumQ = (seqQHost[batchSize - 1] == numTokens);
    for (int64_t b = 0; b < batchSize; ++b) {
        maxKvSeq = std::max(maxKvSeq, seqKvHost[b]);
        int64_t tmpS1 = 0;
        if (accumQ) {
            tmpS1 = (b == 0) ? seqQHost[0] : (seqQHost[b] - seqQHost[b - 1]);
        } else {
            tmpS1 = seqQHost[b];
        }
        if (tmpS1 <= 0) {
            OPS_LOG_E(nodeName, "invalid actual_seq_len_q[%ld]=%ld (accum=%d)", b, tmpS1,
                      accumQ ? 1 : 0);
            return ge::GRAPH_FAILED;
        }
        qSeqSize = std::max(qSeqSize, static_cast<uint32_t>(tmpS1));
    }
    if (maxKvSeq <= 0) {
        OPS_LOG_E(nodeName, "invalid max kv seq");
        return ge::GRAPH_FAILED;
    }
    // Decode P0: one query token per sequence (TND/BNSD S=1).
    if (qSeqSize != 1U) {
        OPS_LOG_E(nodeName, "Us2 P0 Decode-only, expected actual_seq_len_q=1, got %u", qSeqSize);
        return ge::GRAPH_FAILED;
    }

    const uint32_t seqSize = static_cast<uint32_t>(maxKvSeq);
    const uint32_t bn = static_cast<uint32_t>(batchSize) * static_cast<uint32_t>(numKvHeads);

    // FD decision stays AIC-oriented (open-source IsFlashDecode uses aicNum).
    bool enableFd = false;
    if (!(seqSize <= kPaNoFdKvMax) &&
        (static_cast<float>(bn) < kFdBnRatioTh * static_cast<float>(aicNum)) &&
        (gSize == 1U || seqSize >= kFdGqaKvMin)) {
        enableFd = true;
    }
    // Open-source NORMAL: coreNum_ = aivNum_ (both AIVs enter Process).
    // FD / cube-view keep aicNum. Cap at kCoreCap-1 (coreSidxEnd[tmp+1]).
    const uint32_t coreNum =
        enableFd ? aicNum : std::min(aivNum, kCoreCap - 1U);

    uint32_t usedCoreNum = 0;
    uint32_t formerCoreNum = 0;
    uint32_t blockSplitBn2Range = 1;
    uint32_t tailSplitedBatchRange = 1;
    uint32_t kvSplitPart = 0;
    uint32_t coreSidxEnd[kCoreCap];
    for (uint32_t i = 0; i < kCoreCap; ++i) {
        coreSidxEnd[i] = bn;
    }

    uint32_t innerSeq = seqSize;
    if (enableFd) {
        kvSplitPart = coreNum / std::max(bn, 1U);
        if (kvSplitPart == 0U) {
            kvSplitPart = 1U;
        }
        while (((seqSize / kvSplitPart) < kFdKvMinSplit) && (kvSplitPart > 1U)) {
            kvSplitPart--;
        }
        usedCoreNum = bn * kvSplitPart;
        innerSeq = RoundUpU32((seqSize + kvSplitPart - 1U) / kvSplitPart, blockSizeU);
        formerCoreNum = 0U;
        blockSplitBn2Range = 1U;
        tailSplitedBatchRange = 1U;
    } else {
        formerCoreNum = bn % coreNum;
        if (formerCoreNum == 0U) {
            blockSplitBn2Range = bn / coreNum;
            tailSplitedBatchRange = blockSplitBn2Range;
        } else {
            blockSplitBn2Range = bn / coreNum + 1U;
            tailSplitedBatchRange = blockSplitBn2Range - 1U;
        }
        usedCoreNum = bn > coreNum ? coreNum : bn;
        for (uint32_t i = 0; i < formerCoreNum && i < kCoreCap; ++i) {
            coreSidxEnd[i] = blockSplitBn2Range * i;
        }
        const uint32_t formerBase = formerCoreNum * blockSplitBn2Range;
        for (uint32_t i = formerCoreNum; i < usedCoreNum && i < kCoreCap; ++i) {
            coreSidxEnd[i] = formerBase + tailSplitedBatchRange * (i - formerCoreNum);
        }
        if (usedCoreNum < kCoreCap) {
            coreSidxEnd[usedCoreNum] = bn;
        }
    }

    // True CV 1:1: bn*2 <= aivNum, not FlashDecode. Matches open-source
    // EnableC1V1. PACK_GATHER product defaults MIX 1:2; set
    // VLLM_ASCEND_BR_MSD_FORCE_C1V2=0 to A/B natural C1V1.
    bool forceC1V2 = false;
#if BR_MSD_US2_PACK_GATHER
    forceC1V2 = true;
#endif
    {
        const char *forceEnv = std::getenv("VLLM_ASCEND_BR_MSD_FORCE_C1V2");
        if (forceEnv != nullptr) {
            if (forceEnv[0] == '1' && forceEnv[1] == '\0') {
                forceC1V2 = true;
            } else if (forceEnv[0] == '0' && forceEnv[1] == '\0') {
                forceC1V2 = false;
            }
        }
    }
    const bool enableC1V1 = !forceC1V2 && !enableFd && (bn * 2U <= aivNum);

    // MIX 1:2 CrossCoreWaitFlag mode-2 needs both AIV of a cube. bn=1 launches
    // one cube and the kernel writes all-zero (WaitFlag never completes).
    // Keep an idle second cube so the even-AIV / dummy-odd-AIV pair exists.
    // C1V1 launches aiv=aic=usedCoreNum; do not pad.
    if (!enableC1V1 && !enableFd && usedCoreNum == 1U && usedCoreNum + 1U < kCoreCap) {
        usedCoreNum = 2U;
        coreSidxEnd[0] = 0U;
        coreSidxEnd[1] = bn;
        coreSidxEnd[2] = bn;
        formerCoreNum = 1U;
        blockSplitBn2Range = bn;
        tailSplitedBatchRange = 0U;
    }

    // Open-source antiquant: sInner=1024, floor to blockSize so Cube baseN can
    // equal one PA page. Across pages is callback / N-iterate, not softmax=16.
    uint32_t sInnerSize = kAntiqSInnerGqa;
    if (blockSizeU != 0U && sInnerSize % blockSizeU != 0U) {
        sInnerSize = (sInnerSize / blockSizeU) * blockSizeU;
    }
#if BR_MSD_US2_PACK_GATHER
    // Native int4 V callback walks PA pages like BMM1. The old sInner=64 cap
    // was for unpack UB (64-row MTE3 vs Cube row-0 race). Int8 K NZ C0=32
    // still cannot fuse pages when blockSize % 32 != 0 (e.g. 16).
    if (blockSizeU != 0U && (blockSizeU % 32U) != 0U && sInnerSize > blockSizeU) {
        sInnerSize = blockSizeU;
    }
#endif
    if (sInnerSize == 0U) {
        sInnerSize = (blockSizeU != 0U) ? blockSizeU : kAntiqSInnerGqa;
    }
    if (sInnerSize > innerSeq && !enableFd) {
        sInnerSize = innerSeq;
    }
    uint32_t sInnerLoopTimes = 1U;
    uint32_t sInnerSizeTail = innerSeq;
    if (sInnerSize > 0U && innerSeq > sInnerSize) {
        sInnerLoopTimes = (innerSeq + sInnerSize - 1U) / sInnerSize;
        sInnerSizeTail = innerSeq - (sInnerLoopTimes - 1U) * sInnerSize;
    }
    const uint32_t sInnerSizeAlign =
#if BR_MSD_US2_PACK_GATHER
        RoundUpU32(sInnerSize, 64U);
#else
        RoundUpU32(sInnerSize, kUbBlockBytes);
#endif

    const uint32_t mmM = kMsdIters * gSize;
    const uint32_t cubeMAlign = RoundUpU32(mmM, 16U);
    const uint32_t mmResUbSize = cubeMAlign * sInnerSizeAlign;
    const uint32_t bmm2ResUbSize = cubeMAlign * headDimAlign;

    IncreFlashAttentionTilingDataV2* tilingV2Ptr = new IncreFlashAttentionTilingDataV2();
    IncreFlashAttentionTilingDataV2& tilingV2 = *tilingV2Ptr;
    auto& tiling = tilingV2.tilingBase;

    if (!SetupBmm1Tiling(ascendcPlatform, mmM, sInnerSize, seqSize, headDim, headDimAlign, blockSizeU,
                        tiling) ||
        !SetupBmm2Tiling(ascendcPlatform, mmM, sInnerSize, sInnerSizeAlign, seqSize, headDim,
                        blockSizeU, tiling) ||
        !SetupBmmPiTiling(ascendcPlatform, blockSplitBn2Range * gSize, headDim, qDtype, tilingV2)) {
        OPS_LOG_E(nodeName, "Us2 bmm1/bmm2/bmmPi GetTiling failed");
        delete tilingV2Ptr;
        return ge::GRAPH_FAILED;
    }

    // Softmax tiling is recomputed on-device (SoftMaxFlashV2TilingFunc in Us2 kernel).

    tiling.baseParams.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.baseParams.set_seqSize(seqSize);
    tiling.baseParams.set_qSeqSize(qSeqSize);
    tiling.baseParams.set_headSize(headDim);
    tiling.baseParams.set_headSizeV(headDim);
    tiling.baseParams.set_blockSize(blockSizeU);
    tiling.baseParams.set_maxBlockNumPerBatch(static_cast<uint32_t>(maxBlocksPerSeq));
    tiling.baseParams.set_maxBlockNumPerSeq(static_cast<uint32_t>(maxBlocksPerSeq));
    tiling.baseParams.set_scaleValue(*scaleValuePtr);
    tiling.baseParams.set_kvHeadNum(static_cast<uint32_t>(numKvHeads));
    tiling.baseParams.set_headNumRatio(gSize);
    tiling.baseParams.set_qHeadNum(static_cast<uint32_t>(numHeads));
    tiling.baseParams.set_nNumOfQInOneGroup(gSize);
    tiling.baseParams.set_batchContinuousFlag(1U);
    tiling.baseParams.set_actualLenQDims(static_cast<uint32_t>(batchSize));
    tiling.baseParams.set_actualLenDims(static_cast<uint32_t>(batchSize));
    tiling.baseParams.set_msdIterNum(kMsdIters);
    tiling.baseParams.set_antiquantPerTensorFlag(0U);
#if BR_MSD_US2_PACK_GATHER
    // Pack meta is per token × kv-head; same offset as open Mode5 PA gather.
    tiling.baseParams.set_antiquantPerHeadFlag(1U);
#else
    tiling.baseParams.set_antiquantPerHeadFlag(0U);
#endif
    tiling.baseParams.set_attenMaskFlag(0U);
    tiling.baseParams.set_totalBlockNum(static_cast<uint32_t>(totalBlockNum));
    tiling.baseParams.set_antiqSeqSize(seqSize);
    tiling.baseParams.set_preToken(2147483647);
#if BR_MSD_US2_PACK_GATHER
    // Pack-inline meta only (same tensor as codes). Sibling paged-meta IO removed.
    tiling.baseParams.set_antiquantParamsInPagedAttentionFlag(0U);
    // Rotate CrossCore flag ids so a leftover Set from the previous kernel
    // cannot satisfy this launch's Wait (in-process 128-row unpack flake).
    static uint32_t brUnpackEpoch = 0;
    brUnpackEpoch = (brUnpackEpoch + 1U) & 1U;
    tiling.baseParams.set_nextToken(brUnpackEpoch);
#else
    tiling.baseParams.set_nextToken(2147483647);
#endif

    if (enableFd) {
        tiling.splitKVParams.set_s2(kvSplitPart);
        uint32_t sInnerLoopSize = RoundUpU32((seqSize + kvSplitPart - 1U) / kvSplitPart, blockSizeU);
        tiling.splitKVParams.set_sInnerLoopSize(sInnerLoopSize);
        tiling.splitKVParams.set_accumOutSize(static_cast<uint32_t>(batchSize) *
                                              static_cast<uint32_t>(numHeads) * kvSplitPart *
                                              headDimAlign);
        tiling.splitKVParams.set_logSumExpSize(2U * static_cast<uint32_t>(batchSize) *
                                               static_cast<uint32_t>(numHeads) * kvSplitPart *
                                               (kUbBlockBytes / 4U));
    } else {
        tiling.splitKVParams.set_s2(0);
    }

    uint32_t* coreStartIdx = tiling.increFlashAttentionCoreParams.get_coreSidxEnd();
    std::memcpy(coreStartIdx, coreSidxEnd, kCoreCap * sizeof(uint32_t));

    tiling.increFlashAttentionSingleCoreParams.set_sInnerLoopTimes(sInnerLoopTimes);
    tiling.increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerSize);
    tiling.increFlashAttentionSingleCoreParams.set_singleProcessSInnerSizeTail(sInnerSizeTail);
    tiling.increFlashAttentionSingleCoreParams.set_usedCoreNum(usedCoreNum);
    tiling.increFlashAttentionSingleCoreParams.set_formerCoreNum(formerCoreNum);
    tiling.increFlashAttentionSingleCoreParams.set_blockSplitBn2Range(blockSplitBn2Range);
    tiling.increFlashAttentionSingleCoreParams.set_tailSplitedBatchRange(tailSplitedBatchRange);
    tiling.increFlashAttentionSingleCoreParams.set_groupSplitSize(gSize);
    tiling.increFlashAttentionSingleCoreParams.set_s1SplitSize(qSeqSize);

    tiling.increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(mmResUbSize);
    tiling.increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize);

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tilingV2.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer too small need=%zu cap=%zu", tilingV2.GetDataSize(),
                  rawTiling == nullptr ? 0UL : rawTiling->GetCapacity());
        delete tilingV2Ptr;
        return ge::GRAPH_FAILED;
    }
    tilingV2.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tilingV2.GetDataSize());

    const uint32_t actCoreNum = aivNum;
    const uint32_t preLoadNum = 1U;
    uint64_t workspaceSize = static_cast<uint64_t>(libapiSize);
    workspaceSize += preLoadNum * static_cast<uint64_t>(mmResUbSize) * actCoreNum * 4ULL;
#if BR_MSD_US2_PACK_GATHER
    workspaceSize += preLoadNum * static_cast<uint64_t>(mmResUbSize) * actCoreNum / 2ULL;  // int4 P
#else
    workspaceSize += preLoadNum * static_cast<uint64_t>(mmResUbSize) * actCoreNum * 1ULL;
#endif
    workspaceSize += preLoadNum * static_cast<uint64_t>(bmm2ResUbSize) * actCoreNum * 4ULL;
    workspaceSize += preLoadNum * static_cast<uint64_t>(bmm2ResUbSize) * actCoreNum * 4ULL;
    workspaceSize += preLoadNum * static_cast<uint64_t>(bmm2ResUbSize) * actCoreNum * 1ULL;
    // Match kernel Init: mm* then FD then PA (no unpack WS).
    if (enableFd) {
        workspaceSize += static_cast<uint64_t>(tiling.splitKVParams.get_accumOutSize()) * 4ULL;
        workspaceSize += static_cast<uint64_t>(tiling.splitKVParams.get_logSumExpSize()) * 4ULL;
    }
    workspaceSize += static_cast<uint64_t>(actCoreNum) * kPaCbBytes * 2ULL;
    // Cube Π: small Q_T A (fp16/bf16, 2B; FD per-head ApplyPiCube) + fp32 C/qRot
    // (batched prologue/epilogue, M = max BN per core * G).
    const uint32_t piAAlign = RoundUpU32(gSize, 16U);
    const uint32_t piMAlign = RoundUpU32(blockSplitBn2Range * gSize, 16U);
    workspaceSize += static_cast<uint64_t>(actCoreNum) * piAAlign * headDimAlign * 2ULL;
    workspaceSize += static_cast<uint64_t>(actCoreNum) * piMAlign * headDimAlign * 4ULL;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        delete tilingV2Ptr;
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(workspaceSize);

    // Match open-source IFATiling::CalcBlockDim exactly:
    //   CalcTschBlockDim(aivLaunch, aicLaunch, aivLaunch) — 3rd arg is the
    //   *launched* aiv count, NOT full-chip aivNum. Wrong 3rd arg desyncs
    //   GetBlockNum()/GetTaskRation() so odd remap skips while usedCoreNum
    //   still splits work across aivNum slots → missing BN heads → cosine~0.
    uint64_t tilingKey = kKeyPaged;
    if (enableFd) {
        tilingKey = kKeyPagedFd;
    } else if (enableC1V1) {
        tilingKey = kKeyPagedC1V1;
    }
    uint32_t aivLaunch = usedCoreNum;
    uint32_t aicLaunch = usedCoreNum;
    if (enableC1V1) {
        // CV 1:1
    } else if (enableFd) {
        aivLaunch = std::max(cvRatio * usedCoreNum, cvRatio);
        aicLaunch = usedCoreNum;
    } else {
        // NORMAL: usedCoreNum is AIV-counted
        aivLaunch = (usedCoreNum + 1U) & ~1U;
        aicLaunch = (aivLaunch + 1U) / 2U;
    }
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivLaunch, aicLaunch, aivLaunch));
    context->SetTilingKey(tilingKey);

    OPS_LOG_I(nodeName,
              "Us2 tiling: B=%ld N2=%ld G=%u S2=%u cores=%u fd=%d c1v1=%d sInner=%u ws=%zu key=%lu",
              batchSize, numKvHeads, gSize, seqSize, usedCoreNum, enableFd ? 1 : 0,
              enableC1V1 ? 1 : 0, sInnerSize, workspaces[0],
              static_cast<unsigned long>(tilingKey));
    delete tilingV2Ptr;
    return ge::GRAPH_SUCCESS;
}

inline uint32_t RoundUpU32(uint32_t a, uint32_t align)
{
    return align == 0 ? a : ((a + align - 1) / align) * align;
}

// Haar Q@Π / O@Π: Q_T[M,D] × Q_T[D,D] → fp32[M,D] (Q_T = query dtype,
// fp16/bf16; both 2B so shapes/align are identical — only the declared
// Cube dtype must match the Q_T Matmul instantiation).
// M = max BN per core * G (batched prologue/epilogue). Always fill so
// REGIST_MATMUL_OBJ(mmPi) is valid when APPLY_PI=0.

inline uint32_t GreatestCommonDiv(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        const uint32_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

}  // namespace

bool SetupBmm1Tiling(const platform_ascendc::PlatformAscendC& platform, uint32_t m, uint32_t sInner,
                    uint32_t seqSize, uint32_t headDim, uint32_t headDimAlign, uint32_t blockSize,
                    IncreFlashAttentionTilingData& tiling)
{
    matmul_tiling::MatmulApiTiling bmm1(platform);
    bmm1.SetShape(m, sInner, headDim);
#if BR_MSD_US2_PACK_GATHER
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT4, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT4, true);
#else
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT8, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT8, true);
#endif
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                  matmul_tiling::DataType::DT_INT32);
    bmm1.SetOrgShape(m, seqSize, headDimAlign, headDim);
    bmm1.SetBias(false);

    uint32_t bmm1BaseN = std::min(RoundUpU32(sInner, 16U), kMmBaseCap);
    if (bmm1BaseN < blockSize) {
        while (blockSize % bmm1BaseN != 0U && bmm1BaseN >= 16U) {
            bmm1BaseN /= 2U;
        }
    } else if (bmm1BaseN > blockSize) {
        bmm1BaseN = GreatestCommonDiv(bmm1BaseN, blockSize);
    }
    if (bmm1BaseN == 0U) {
        bmm1BaseN = 16U;
    }
    // Int8 MDL SplitB: if baseN%32!=0 && stepN>1, axisL1Offset uses
    // CeilAlign(baseN,32)*C0, but PA callback writes one fused [useN,K]
    // NZ and L1 B is sized to useN*K (kv=32/bs=16: 4096, not 2*4096).
    // Callback rewrite of independent tiles OOB'd L1 (cosine 0.48/0.81).
    // Prefer baseN=sInner (stepN=1) instead of GreatestCommonDiv-down to 16. Product
    // blockSize=128 already %32==0, this branch does not fire.
    constexpr uint32_t kInt8NzC0 = 32U;
    if ((bmm1BaseN % kInt8NzC0) != 0U) {
        const uint32_t singleN = std::min(RoundUpU32(sInner, 16U), kMmBaseCap);
        if (singleN > 0U) {
            bmm1BaseN = singleN;
        }
    }
    const uint32_t bmm1MaxBaseM =
        RoundUpU32(static_cast<uint32_t>(kL0cBytes / sizeof(float) / bmm1BaseN) - 16U, 16U);
    if (bmm1.SetFixSplit(std::min(RoundUpU32(m, 16U), bmm1MaxBaseM), bmm1BaseN) == -1) {
        return false;
    }
    if (bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) == -1) {
        return false;
    }
    if (bmm1.GetTiling(tiling.bmm1TilingData) == -1) {
        return false;
    }
    // Do not poke stepN/baseK after GetTiling — that desyncs C dest offsets
    // (sInner=128 then only the last 16-token N-tile survives).
    return true;
}

IMPL_OP_OPTILING(RqUniformFiaPagedK4v4Msd)
    .Tiling(RqUniformFiaPagedK4v4MsdUs2TilingFunc);

}  // namespace optiling
