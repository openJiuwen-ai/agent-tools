/*
 * Host tiling for TurboQuant FIA MSE-8bit (TND + PA + GQA).
 *
 * Uses ops-transformer SplitCore for outer split + FlashDecode tables.
 * Tiling keys: 0 = no FD, 1 = FlashDecode (numOfFdHead > 0).
 */

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "rq_uniform_fia_paged_k8v4_tiling.h"
#include "split_core.h"  // shared TU in csrc/common_fia_split (sole provider of optiling::SplitCore)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace optiling {
namespace {

constexpr uint32_t kPreLoadNum = 2;
// P29: BR_S2_BASICSIZE_IS_1024=1 → s2Base=1024 (FD 4→2 @ kv=2000).
// Must match Cube CFG::S2_BASICSIZE_IS_1024 (same compile define).
#ifndef BR_S2_BASICSIZE_IS_1024
#define BR_S2_BASICSIZE_IS_1024 1
#endif
#if BR_S2_BASICSIZE_IS_1024
constexpr uint32_t kS2BaseSize = 1024;
#else
constexpr uint32_t kS2BaseSize = 512;
#endif
// Align with FIA TND CalcMBaseSize (M_BASE_SIZE_512), not the non-TND 256 tier.
constexpr uint32_t kMBaseSize = 512;
constexpr uint32_t kFdGS1BaseSize = 8;
constexpr uint32_t kByteBlock = 32;
constexpr uint32_t kBlockTableElemByte = 4;
constexpr uint32_t kLibApiWorkspaceFallback = 16 * 1024 * 1024;
constexpr uint64_t kTilingKeyNoFd = 0;
constexpr uint64_t kTilingKeyFd = 1;
constexpr int64_t kSparseModeIntMax = 2147483647;
constexpr int32_t kSparseModeNoMask = 0;
constexpr int32_t kSparseModeAllMask = 1;
constexpr int32_t kSparseModeLeftUp = 2;
constexpr int32_t kSparseModeRightDown = 3;
constexpr int32_t kSparseModeBand = 4;
constexpr uint32_t kCompressMaskStride = 2048;
constexpr uint32_t kSInnerSizeCompressCap = 1024;
constexpr size_t kAttrIdxPreTokens = 5;
constexpr size_t kAttrIdxNextTokens = 6;
constexpr size_t kAttrIdxSparseMode = 7;
constexpr size_t kInputIdxAttenMask = 6;
constexpr uint32_t kBrBlockRows = 16;

inline uint32_t TqKeyTileBytes(int64_t headSize)
{
    return kBrBlockRows * static_cast<uint32_t>(headSize + 4);
}

inline uint32_t TqValTileBytes(int64_t headSize)
{
    return kBrBlockRows * static_cast<uint32_t>(headSize / 2 + 4);
}

inline uint32_t AlignUp(uint32_t a, uint32_t align)
{
    return align == 0 ? a : ((a + align - 1) / align) * align;
}

inline int64_t ClampSparseToken(int64_t v)
{
    if (v > kSparseModeIntMax) {
        return kSparseModeIntMax;
    }
    if (v < -kSparseModeIntMax) {
        return -kSparseModeIntMax;
    }
    return v;
}

// Mirror FiaInfoParser::GetPreNextToken for causal / all-mask modes.
void ApplySparsePreNextTokens(int32_t sparseMode, int64_t& preToken, int64_t& nextToken)
{
    if (sparseMode == kSparseModeAllMask) {
        preToken = kSparseModeIntMax;
        nextToken = kSparseModeIntMax;
    } else if (sparseMode == kSparseModeLeftUp || sparseMode == kSparseModeRightDown) {
        preToken = kSparseModeIntMax;
        nextToken = 0;
    }
    preToken = ClampSparseToken(preToken);
    nextToken = ClampSparseToken(nextToken);
}

// Mirror FiaInfoParser::GetAttenMaskInfo for stride / batchStride.
void CalcAttenMaskStrides(const gert::Shape& maskShape, int64_t batchSize, uint32_t s1Size,
                          int32_t sparseMode, uint32_t& batchStride, uint32_t& maskStride)
{
    const size_t dimNum = maskShape.GetDimNum();
    batchStride = 0U;
    if (dimNum == 2U && s1Size == 1U && maskShape.GetDim(0) != 1) {
        batchStride = static_cast<uint32_t>(maskShape.GetDim(dimNum - 1));
    } else if ((dimNum == 3U || dimNum == 4U) && maskShape.GetDim(0) == batchSize && batchSize != 1) {
        batchStride = static_cast<uint32_t>(maskShape.GetDim(dimNum - 1) * maskShape.GetDim(dimNum - 2));
    }

    if (sparseMode == kSparseModeNoMask || sparseMode == kSparseModeAllMask) {
        maskStride = static_cast<uint32_t>(maskShape.GetDim(dimNum - 1));
    } else {
        maskStride = kCompressMaskStride;  // compressed causal/band mask
    }
}

void GetSafeActToken(SparseMode mode, int64_t actSeqQ, int64_t actSeqKv, int64_t& safePre,
                     int64_t& safeNext)
{
    if (mode == SparseMode::DEFAULT_MASK) {
        safePre = std::max(-actSeqKv, safePre);
        safePre = std::min(safePre, actSeqQ);
        safeNext = std::max(-actSeqQ, safeNext);
        safeNext = std::min(safeNext, actSeqKv);
    } else if (mode == SparseMode::BAND) {
        safePre = std::max(-actSeqQ, safePre);
        safePre = std::min(safePre, actSeqKv);
        safeNext = std::max(-actSeqKv, safeNext);
        safeNext = std::min(safeNext, actSeqQ);
    }
}

// Mirror FiaTilingNonQuant::IsExistRowInvalid.
bool IsExistRowInvalid(const BaseInfo& baseInfo)
{
    if (!baseInfo.attenMaskFlag) {
        return false;
    }
    const auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::LEFT_UP_CAUSAL) {
        return false;
    }
    if (mode == SparseMode::ALL_MASK) {
        return true;
    }
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; ++bIdx) {
        const int32_t s1 = static_cast<int32_t>(GetS1SeqSize(bIdx, baseInfo));
        const int32_t s2 = static_cast<int32_t>(GetS2SeqSize(bIdx, baseInfo));
        if (s1 == 0 || s2 == 0) {
            continue;
        }
        int64_t safePre = baseInfo.preToken;
        int64_t safeNext = baseInfo.nextToken;
        GetSafeActToken(mode, s1, s2, safePre, safeNext);
        int64_t preLeftUp = 0;
        int64_t nextLeftUp = 0;
        if (mode == SparseMode::BAND) {
            preLeftUp = safePre;
            nextLeftUp = static_cast<int64_t>(s2) - static_cast<int64_t>(s1) + safeNext;
        } else if (mode == SparseMode::DEFAULT_MASK) {
            preLeftUp = static_cast<int64_t>(s2) - static_cast<int64_t>(s1) + safePre;
            nextLeftUp = safeNext;
        } else {
            preLeftUp = 0;
            nextLeftUp = static_cast<int64_t>(s2) - static_cast<int64_t>(s1);
        }
        if (preLeftUp < 0 || nextLeftUp < 0) {
            return true;
        }
    }
    return false;
}

void ApplySplitResult(RqUniformFiaPagedK8v4TilingData& tiling, const SplitResult& res,
                      uint32_t aicNum, uint32_t cvRatio)
{
    uint32_t bN2End[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1End[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2End[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t bN2IdxOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1IdxOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2SplitNumOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2SplitStartIdxOfCore[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1SplitNumOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1LastPartSizeOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1IdxEndOfFdHead[TQ_FIA_MAX_AIC_CORE_NUM * 2] = {0};
    uint32_t gS1IdxEndOfFdHeadSplit[TQ_FIA_MAX_AIC_CORE_NUM * 2] = {0};

    const uint32_t n = std::min(res.usedCoreNum, TQ_FIA_MAX_AIC_CORE_NUM);
    for (uint32_t i = 0; i < n; ++i) {
        bN2End[i] = res.bN2End[i];
        gS1End[i] = res.gS1End[i];
        s2End[i] = res.s2End[i];
        bN2IdxOfFdHead[i] = res.fdRes.bN2IdxOfFdHead[i];
        gS1IdxOfFdHead[i] = res.fdRes.gS1IdxOfFdHead[i];
        s2SplitNumOfFdHead[i] = res.fdRes.s2SplitNumOfFdHead[i];
        s2SplitStartIdxOfCore[i] = res.fdRes.s2SplitStartIdxOfCore[i];
        gS1SplitNumOfFdHead[i] = res.fdRes.gS1SplitNumOfFdHead[i];
        gS1LastPartSizeOfFdHead[i] = res.fdRes.gS1LastPartSizeOfFdHead[i];
    }
    const uint32_t nVec = std::min(res.usedCoreNum * cvRatio, TQ_FIA_MAX_AIC_CORE_NUM * 2U);
    for (uint32_t i = 0; i < nVec; ++i) {
        gS1IdxEndOfFdHead[i] = res.fdRes.gS1IdxEndOfFdHead[i];
        gS1IdxEndOfFdHeadSplit[i] = res.fdRes.gS1IdxEndOfFdHeadSplit[i];
    }

    tiling.outerSplitParams.set_bN2End(bN2End);
    tiling.outerSplitParams.set_gS1End(gS1End);
    tiling.outerSplitParams.set_s2End(s2End);

    tiling.fdParams.set_numOfFdHead(res.numOfFdHead);
    tiling.fdParams.set_reserved(0U);
    tiling.fdParams.set_gS1BaseSizeOfFd(kFdGS1BaseSize);
    tiling.fdParams.set_usedVecNumOfFd(res.numOfFdHead > 0U ? res.usedVecNumOfFd : 0U);
    tiling.fdParams.set_bN2IdxOfFdHead(bN2IdxOfFdHead);
    tiling.fdParams.set_gS1IdxOfFdHead(gS1IdxOfFdHead);
    tiling.fdParams.set_s2SplitNumOfFdHead(s2SplitNumOfFdHead);
    tiling.fdParams.set_s2SplitStartIdxOfCore(s2SplitStartIdxOfCore);
    tiling.fdParams.set_gS1SplitNumOfFdHead(gS1SplitNumOfFdHead);
    tiling.fdParams.set_gS1LastPartSizeOfFdHead(gS1LastPartSizeOfFdHead);
    tiling.fdParams.set_gS1IdxEndOfFdHead(gS1IdxEndOfFdHead);
    tiling.fdParams.set_gS1IdxEndOfFdHeadSplit(gS1IdxEndOfFdHeadSplit);

    (void)aicNum;
}

}  // namespace

static ge::graphStatus RqUniformFiaPagedK8v4TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(nodeName, "attrs is null");
        return ge::GRAPH_FAILED;
    }

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OPS_LOG_E(nodeName, "platformInfo is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    const uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (aicNum == 0 || aivNum == 0) {
        OPS_LOG_E(nodeName, "invalid core nums aic=%u aiv=%u", aicNum, aivNum);
        return ge::GRAPH_FAILED;
    }
    const uint32_t cvRatio = aivNum / aicNum;
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (libapiSize == 0) {
        libapiSize = kLibApiWorkspaceFallback;
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

    // Optional FIA-compatible sparse attrs (defaults match OpDef).
    const int64_t* preTokensPtr = attrs->GetAttrPointer<int64_t>(kAttrIdxPreTokens);
    const int64_t* nextTokensPtr = attrs->GetAttrPointer<int64_t>(kAttrIdxNextTokens);
    const int64_t* sparseModePtr = attrs->GetAttrPointer<int64_t>(kAttrIdxSparseMode);
    int64_t preToken = preTokensPtr != nullptr ? *preTokensPtr : kSparseModeIntMax;
    int64_t nextToken = nextTokensPtr != nullptr ? *nextTokensPtr : kSparseModeIntMax;
    const int32_t sparseMode =
        sparseModePtr != nullptr ? static_cast<int32_t>(*sparseModePtr) : kSparseModeNoMask;
    if (sparseMode < kSparseModeNoMask || sparseMode > kSparseModeBand) {
        OPS_LOG_E(nodeName, "invalid sparse_mode=%d (expect 0..4)", sparseMode);
        return ge::GRAPH_FAILED;
    }

    const int64_t numHeads = *numHeadsPtr;
    const int64_t numKvHeads = *numKvHeadsPtr;
    const int64_t headSize = *headSizePtr;
    const int64_t blockSize = *blockSizePtr;
    if (headSize != 128 && headSize != 256) {
        OPS_LOG_E(nodeName, "only head_size=128 or 256 supported, got %ld", headSize);
        return ge::GRAPH_FAILED;
    }
    if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0 || blockSize <= 0) {
        OPS_LOG_E(nodeName, "invalid head/block attrs");
        return ge::GRAPH_FAILED;
    }
    if (blockSize % kBrBlockRows != 0) {
        OPS_LOG_E(nodeName, "block_size must be a multiple of %u, got %ld", kBrBlockRows, blockSize);
        return ge::GRAPH_FAILED;
    }
    const uint32_t gSize = static_cast<uint32_t>(numHeads / numKvHeads);

    auto qShapePtr = context->GetInputShape(0);
    auto keyCacheShapePtr = context->GetInputShape(1);
    auto valCacheShapePtr = context->GetInputShape(2);
    auto btShapePtr = context->GetInputShape(3);
    auto seqQTensor = context->GetInputTensor(4);
    auto seqKvTensor = context->GetInputTensor(5);
    if (qShapePtr == nullptr || keyCacheShapePtr == nullptr || valCacheShapePtr == nullptr ||
        btShapePtr == nullptr || seqQTensor == nullptr || seqKvTensor == nullptr) {
        OPS_LOG_E(nodeName, "required inputs missing");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape keyCacheShape = keyCacheShapePtr->GetStorageShape();
    const gert::Shape valCacheShape = valCacheShapePtr->GetStorageShape();
    const gert::Shape btShape = btShapePtr->GetStorageShape();
    if (qShape.GetDimNum() < 3 || keyCacheShape.GetDimNum() < 3 || valCacheShape.GetDimNum() < 3 ||
        btShape.GetDimNum() < 2) {
        OPS_LOG_E(nodeName, "unexpected ranks query=%zu key=%zu value=%zu block_table=%zu",
                  qShape.GetDimNum(), keyCacheShape.GetDimNum(), valCacheShape.GetDimNum(),
                  btShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const uint32_t subBlocksPerBlock = static_cast<uint32_t>(blockSize) / kBrBlockRows;
    const uint32_t keyPackedBytes = subBlocksPerBlock * TqKeyTileBytes(headSize);
    const uint32_t valuePackedBytes = subBlocksPerBlock * TqValTileBytes(headSize);
    if (keyCacheShape.GetDim(1) != numKvHeads ||
        keyCacheShape.GetDim(2) != static_cast<int64_t>(keyPackedBytes)) {
        OPS_LOG_E(nodeName,
                  "key_cache shape must be [num_blocks, num_kv_heads, (block_size/%u)*%u], "
                  "got [%ld, %ld, %ld]",
                  kBrBlockRows, TqKeyTileBytes(headSize), keyCacheShape.GetDim(0), keyCacheShape.GetDim(1),
                  keyCacheShape.GetDim(2));
        return ge::GRAPH_FAILED;
    }
    if (valCacheShape.GetDim(1) != numKvHeads ||
        valCacheShape.GetDim(2) != static_cast<int64_t>(valuePackedBytes)) {
        OPS_LOG_E(nodeName,
                  "value_cache shape must be [num_blocks, num_kv_heads, (block_size/%u)*%u], "
                  "got [%ld, %ld, %ld]",
                  kBrBlockRows, TqValTileBytes(headSize), valCacheShape.GetDim(0), valCacheShape.GetDim(1),
                  valCacheShape.GetDim(2));
        return ge::GRAPH_FAILED;
    }

    const int64_t numTokens = qShape.GetDim(0);
    const int64_t batchSize = btShape.GetDim(0);
    const int64_t maxBlocksPerSeq = btShape.GetDim(1);
    if (numTokens <= 0 || batchSize <= 0 || maxBlocksPerSeq <= 0) {
        OPS_LOG_E(nodeName, "invalid query/block_table shape");
        return ge::GRAPH_FAILED;
    }

    // atten_mask is OPTIONAL; present + non-empty enables mask path (FIA GetMaskFlag).
    const gert::StorageShape* maskShapePtr = context->GetOptionalInputShape(kInputIdxAttenMask);
    const bool attenMaskFlag =
        (maskShapePtr != nullptr) && (maskShapePtr->GetStorageShape().GetShapeSize() != 0);
    if (!attenMaskFlag && sparseMode != kSparseModeNoMask) {
        OPS_LOG_E(nodeName, "sparse_mode=%d requires non-empty atten_mask", sparseMode);
        return ge::GRAPH_FAILED;
    }
    ApplySparsePreNextTokens(sparseMode, preToken, nextToken);

    const int64_t* seqQHost = seqQTensor->GetData<int64_t>();
    const int64_t* seqKvHost = seqKvTensor->GetData<int64_t>();
    if (seqQHost == nullptr || seqKvHost == nullptr) {
        OPS_LOG_E(nodeName, "actual_seq_len tensors must be host-value-dependent");
        return ge::GRAPH_FAILED;
    }

    int64_t maxKvSeq = 0;
    for (int64_t b = 0; b < batchSize; ++b) {
        maxKvSeq = std::max(maxKvSeq, seqKvHost[b]);
    }
    if (maxKvSeq <= 0) {
        OPS_LOG_E(nodeName, "invalid max kv seq");
        return ge::GRAPH_FAILED;
    }

    const bool accumQ = (seqQHost[batchSize - 1] == numTokens);
    // TND: s1Size must be max per-batch S1 (FIA GetS1Size), NOT 1 when accum.
    // qSeqSize==1 forces mask layout S1_EQUAL1 and breaks multi-Q causal.
    uint32_t s1Size = 0U;
    for (int64_t b = 0; b < batchSize; ++b) {
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
        s1Size = std::max(s1Size, static_cast<uint32_t>(tmpS1));
    }
    const uint32_t s2Size = static_cast<uint32_t>(maxKvSeq);
    uint32_t attenMaskBatchStride = 0;
    uint32_t attenMaskStride = 0;
    if (attenMaskFlag && maskShapePtr != nullptr) {
        CalcAttenMaskStrides(maskShapePtr->GetStorageShape(), batchSize, s1Size, sparseMode,
                             attenMaskBatchStride, attenMaskStride);
    }

    uint32_t s2BaseSize = kS2BaseSize;
    // FIA: compress mask (sparse 2/3/4) caps sInner at 1024; TND base is already 512.
    if (attenMaskFlag && (sparseMode == kSparseModeLeftUp || sparseMode == kSparseModeRightDown ||
                          sparseMode == kSparseModeBand)) {
        s2BaseSize = std::min(s2BaseSize, kSInnerSizeCompressCap);
    }
    // PA: keep s2Base a multiple of blockSize when possible.
    if (blockSize > 0 && s2BaseSize > static_cast<uint32_t>(blockSize) &&
        (s2BaseSize % static_cast<uint32_t>(blockSize) != 0U)) {
        s2BaseSize = (s2BaseSize / static_cast<uint32_t>(blockSize)) *
                     static_cast<uint32_t>(blockSize);
    }
    // Must match device actualSingleProcessSInnerSizeAlign =
    // Align(actualS2, fa_base_vector::BYTE_BLOCK) with BYTE_BLOCK=32.
    // Aligning only to 16 under-sizes mm1/vec1 workspace whenever S2%32!=0
    // (e.g. 144→host 144 vs device stride 160) → Fixpipe OOB → nondeterministic NaN.
    const uint32_t sInnerSizeAlign = AlignUp(std::min(s2Size, s2BaseSize), kByteBlock);
    const uint32_t headDimAlign = AlignUp(static_cast<uint32_t>(headSize), 16U);

    BaseInfo baseInfo {};
    SplitParam splitParam {};
    baseInfo.bSize = static_cast<uint32_t>(batchSize);
    baseInfo.n2Size = static_cast<uint32_t>(numKvHeads);
    baseInfo.gSize = gSize;
    baseInfo.s1Size = s1Size;
    baseInfo.s2Size = s2Size;
    baseInfo.isS1G = true;  // TND
    baseInfo.isAccumSeqS1 = accumQ;
    baseInfo.isAccumSeqS2 = false;
    baseInfo.actualLenQDims = static_cast<uint32_t>(batchSize);
    baseInfo.actualLenKvDims = static_cast<uint32_t>(batchSize);
    baseInfo.attenMaskFlag = attenMaskFlag;
    baseInfo.sparseMode = sparseMode;
    baseInfo.preToken = preToken;
    baseInfo.nextToken = nextToken;
    baseInfo.actualSeqS1Size.assign(seqQHost, seqQHost + batchSize);
    baseInfo.actualSeqS2Size.assign(seqKvHost, seqKvHost + batchSize);

    splitParam.mBaseSize = kMBaseSize;
    splitParam.s2BaseSize = s2BaseSize;
    splitParam.gS1BaseSizeOfFd = kFdGS1BaseSize;

    SplitResult splitRes {aicNum, cvRatio};
    SplitCore(aicNum, baseInfo, splitParam, splitRes);
    if (splitRes.usedCoreNum == 0U) {
        OPS_LOG_E(nodeName, "SplitCore returned usedCoreNum=0");
        return ge::GRAPH_FAILED;
    }
    if (splitRes.numOfFdHead > aicNum || splitRes.usedCoreNum > aicNum ||
        splitRes.maxS2SplitNum > aicNum + 1U) {
        OPS_LOG_E(nodeName,
                  "SplitCore out of range used=%u fdHeads=%u maxS2Split=%u aic=%u",
                  splitRes.usedCoreNum, splitRes.numOfFdHead, splitRes.maxS2SplitNum, aicNum);
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum = splitRes.usedCoreNum;
    const bool enableFd = (splitRes.numOfFdHead > 0U);
    const bool existRowInvalid = IsExistRowInvalid(baseInfo);

    RqUniformFiaPagedK8v4TilingData tiling {};
    auto& base = tiling.baseParams;
    base.set_bSize(static_cast<uint32_t>(batchSize));
    base.set_n2Size(static_cast<uint32_t>(numKvHeads));
    base.set_gSize(gSize);
    base.set_s1Size(s1Size);
    base.set_s2Size(s2Size);
    base.set_headDim(static_cast<uint32_t>(headSize));
    base.set_headDimRope(0);
    base.set_actualSeqS1Dims(static_cast<uint32_t>(batchSize));
    base.set_actualSeqS2Dims(static_cast<uint32_t>(batchSize));
    base.set_accumQSeqFlag(accumQ ? 1U : 0U);
    base.set_accumKVSeqFlag(0U);
    base.set_scaleValue(*scaleValuePtr);
    base.set_usedCoreNum(usedCoreNum);
    base.set_outputLayout(3U);  // FIA_LAYOUT::TND
    base.set_batchContinuous(1U);
    base.set_softmaxLseFlag(0U);
    base.set_needInit(0U);
    base.set_slidingFlag(0U);
    base.set_l2CacheOffFlag(0U);
    base.set_isLegacyIfa(0U);

    tiling.pageAttenParams.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.pageAttenParams.set_maxBlockNumPerBatch(static_cast<uint32_t>(maxBlocksPerSeq));

    tiling.maskParams.set_attenMaskFlag(attenMaskFlag ? 1U : 0U);
    tiling.maskParams.set_attenMaskBatchStride(attenMaskBatchStride);
    tiling.maskParams.set_attenMaskStride(attenMaskStride);
    tiling.maskParams.set_preToken(static_cast<int32_t>(preToken));
    tiling.maskParams.set_nextToken(static_cast<int32_t>(nextToken));
    tiling.maskParams.set_isRowInvalid(0U);  // no inner_precise attr yet (FIA: innerPrecise>>1)
    tiling.maskParams.set_isExistRowInvalid(existRowInvalid ? 1U : 0U);
    tiling.maskParams.set_sparseMode(static_cast<uint32_t>(sparseMode));

    const uint32_t mm1ResSize = kMBaseSize * sInnerSizeAlign;
    const uint32_t mm2ResSize = kMBaseSize * headDimAlign;
    tiling.workspaceParams.set_mm1ResSize(mm1ResSize);
    tiling.workspaceParams.set_mm2ResSize(mm2ResSize);

    // S2-indexed dequant cache: only useful when the same (b,n2,s2) is revisited
    // across gS1 tiles (Prefill with gS1LoopTimes>=2). Decode (and short Prefill
    // with a single gS1 tile) keeps classic 2-slot loop%2 ping-pong — no skip
    // benefit, and no extra WS / s2Idx indexing tax.
    const uint32_t s2LoopTimes =
        s2BaseSize == 0U ? 1U : (s2Size + s2BaseSize - 1U) / s2BaseSize;
    const uint64_t gS1Size =
        static_cast<uint64_t>(s1Size) * static_cast<uint64_t>(gSize);
    const uint32_t gS1LoopTimes = kMBaseSize == 0U
                                      ? 1U
                                      : static_cast<uint32_t>((gS1Size + kMBaseSize - 1U) /
                                                             kMBaseSize);
    uint32_t maxDeqSlots = 8U;
    if (const char *envMax = std::getenv("TQ_FIA_DEQUANT_S2_CACHE_MAX_SLOTS")) {
        char *end = nullptr;
        unsigned long v = std::strtoul(envMax, &end, 10);
        if (end != envMax && v >= 2UL && v <= 64UL) {
            maxDeqSlots = static_cast<uint32_t>(v);
        }
    }
    bool wantS2Cache = true;
    if (const char *envCache = std::getenv("TQ_FIA_DEQUANT_S2_CACHE")) {
        wantS2Cache = !(envCache[0] == '0' && envCache[1] == '\0');
    }
    uint32_t dequantWsSlots = kPreLoadNum;
    uint32_t dequantS2Cache = 0U;
    if (wantS2Cache && gS1LoopTimes >= 2U && s2LoopTimes >= 1U &&
        s2LoopTimes <= maxDeqSlots) {
        dequantWsSlots = std::max(kPreLoadNum, s2LoopTimes);
        dequantS2Cache = 1U;
    }
    tiling.workspaceParams.set_dequantWsSlots(dequantWsSlots);
    tiling.workspaceParams.set_dequantS2Cache(dequantS2Cache);

    tiling.innerSplitParams.set_mBaseSize(kMBaseSize);
    tiling.innerSplitParams.set_s2BaseSize(s2BaseSize);

    ApplySplitResult(tiling, splitRes, aicNum, cvRatio);

    // FD workspace element counts (mirror FiaTilingNonQuant::FillTilingWorkspaceParams).
    const uint64_t fdAccumOutSize =
        enableFd ? (static_cast<uint64_t>(aicNum) * 2ULL * kMBaseSize * headDimAlign) : 0ULL;
    const uint64_t fdLogSumExpSize =
        enableFd ? (2ULL * aicNum * 2ULL * kMBaseSize * (kByteBlock / kBlockTableElemByte)) : 0ULL;
    tiling.workspaceParams.set_fdAccumOutSize(static_cast<uint32_t>(fdAccumOutSize));
    tiling.workspaceParams.set_fdLogSumExpSize(static_cast<uint32_t>(fdLogSumExpSize));

    tiling.prefixParams.set_prefixMaxLen(0);
    tiling.prefixParams.set_prefixLen(0);
    tiling.prefixParams.set_prefixFlag(false);
    tiling.pseParams.set_pseShiftFlag(0U);
    tiling.pseParams.set_pseShiftByBatch(0U);
    tiling.pseParams.set_pseShiftS1(0U);
    tiling.pseParams.set_pseShiftS2(0U);
    tiling.leftPaddingParams.set_qPaddingFlag(0U);
    tiling.leftPaddingParams.set_kvPaddingFlag(0U);

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < tiling.GetDataSize()) {
        OPS_LOG_E(nodeName, "raw tiling buffer is null or too small need=%zu cap=%zu",
                  tiling.GetDataSize(),
                  rawTiling == nullptr ? 0UL : rawTiling->GetCapacity());
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    const uint64_t normalWs =
        static_cast<uint64_t>(kPreLoadNum) * usedCoreNum *
        (static_cast<uint64_t>(mm1ResSize) * 4ULL +
         static_cast<uint64_t>(mm1ResSize) * 2ULL +
         static_cast<uint64_t>(mm2ResSize) * 4ULL +
         static_cast<uint64_t>(mm2ResSize) * 4ULL);
    // FD bytes: (accumOut + logSumExp) * sizeof(float) with paramNums = aic*2*mBase
    const uint64_t fdParamNums = static_cast<uint64_t>(aicNum) * 2ULL * kMBaseSize;
    const uint64_t fdWs =
        enableFd ? ((fdParamNums * headDimAlign +
                     2ULL * fdParamNums * (kByteBlock / kBlockTableElemByte)) *
                    static_cast<uint64_t>(kBlockTableElemByte))
                 : 0ULL;
    // Dequant workspace stride on device uses s2BaseSize (BYTE_BLOCK-aligned), NOT
    // sInnerSizeAlign=min(s2,s2Base). Multi-core GQA indexes aiCoreIdx*perCoreDequantSize
    // with that stride; allocating only sInnerSizeAlign makes core>=1 collide/OOB.
    // Slot count = dequantWsSlots (2 ping-pong or s2LoopTimes for S2 cache).
    const uint32_t dequantS2Align = AlignUp(s2BaseSize, kByteBlock);
    const uint64_t tqWs =
        static_cast<uint64_t>(dequantWsSlots) * usedCoreNum * 2ULL *
        static_cast<uint64_t>(dequantS2Align) * headDimAlign * 2ULL;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OPS_LOG_E(nodeName, "workspace size buffer is null");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(libapiSize + normalWs + fdWs + tqWs);

    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(usedCoreNum * cvRatio, usedCoreNum,
                                                               usedCoreNum * cvRatio);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(enableFd ? kTilingKeyFd : kTilingKeyNoFd);

    OPS_LOG_I(nodeName,
              "BR-FIA tiling: B=%ld N2=%ld G=%u S1=%u S2=%u gS1Loops=%u cores=%u fd=%d "
              "fdHeads=%u maxS2Split=%u mask=%d sparse=%d pre=%ld next=%ld "
              "deqSlots=%u s2Cache=%u ws=%zu key=%lu",
              batchSize, numKvHeads, gSize, s1Size, s2Size, gS1LoopTimes, usedCoreNum,
              enableFd ? 1 : 0, splitRes.numOfFdHead, splitRes.maxS2SplitNum,
              attenMaskFlag ? 1 : 0, sparseMode, preToken, nextToken, dequantWsSlots,
              dequantS2Cache, workspaces[0],
              static_cast<unsigned long>(enableFd ? kTilingKeyFd : kTilingKeyNoFd));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RqUniformFiaPagedK8v4)
    .Tiling(RqUniformFiaPagedK8v4TilingFunc);

}  // namespace optiling
