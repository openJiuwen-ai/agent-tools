/*!
 * \file split_core.h
 * \brief Shared FIA paged-attention core-splitting (host tiling). Single canonical
 *        copy for all FIA-family ops (oscar_fia / oscar_dequant / rq_uniform_fia /
 *        turboquant_fia_mse8bit). Vendoring a second copy into a per-op vendored/
 *        tree causes a multiple-definition link error on optiling::SplitCore; every
 *        FIA op must include this header and let the host op's CMakeLists be the
 *        sole provider of split_core.cpp.
 */

#ifndef SPLIT_CORE_H
#define SPLIT_CORE_H

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace optiling {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

template<class T>
using Range = std::pair<T, T>;

template<class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template<typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template<typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// 分核功能模块输入：输入case的基本信息
struct BaseInfo {
    uint32_t bSize { 0U };
    uint32_t n2Size { 0U };
    uint32_t gSize { 0U };
    uint32_t qTokLen { 0U };
    uint32_t kvTokLen { 0U };
    bool isS1G { true };
    bool isAccumSeqS1 { false };
    bool isAccumSeqS2 { false };
    std::vector<int64_t> actualSeqS1Size {};
    std::vector<int64_t> actualSeqS2Size {};
    uint32_t actualLenQDims { 0U };
    uint32_t actualLenKvDims { 0U };
    bool attenMaskFlag { false };
    int32_t sparseMode { 0U };
    int64_t preToken { 0 };
    int64_t nextToken { 0 };
    int64_t actualSeqPrefixSize { 0 };
};

// 分核功能模块输入：切分属性，预留接口，可作为切分方案的参数入口
struct SplitParam {
    uint32_t mBaseSize { 1U };
    uint32_t s2BaseSize { 1U };
    uint32_t gS1BaseSizeOfFd { 8U };                // FD阶段分核，m轴切分基本块大小
};


// 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
struct FlashDecodeResult {
    // 1、归约任务的索引信息
    std::vector<uint32_t> bN2IdxOfFdHead {};           // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> gS1IdxOfFdHead {};           // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> s2SplitNumOfFdHead {};       // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    // 2、FD负载均衡阶段，归约任务的分核（vec）信息
    std::vector<uint32_t> gS1SplitNumOfFdHead {};      // 每个归约任务m轴切分份数，脚标为归约任务的序号
    std::vector<uint32_t> gS1LastPartSizeOfFdHead {};  // 每个归约任务m轴切分的最后一份的大小，脚标为归约任务的序号
    std::vector<uint32_t> gS1IdxEndOfFdHead {};        // FD负载均衡阶段，每个vector的一级索引，脚标为vector ID，值为归约任务的ID
    std::vector<uint32_t> gS1IdxEndOfFdHeadSplit {};   // FD负载均衡阶段，每个vector的二级索引，脚标为vector ID，值为归约任务的m轴切分ID
    // 3、每个core处理的第1个归约任务的数据应存放的workspace位置
    std::vector<uint32_t> s2SplitStartIdxOfCore {};

    FlashDecodeResult(uint32_t aicBudget, uint32_t vecCubeRatio) :
        bN2IdxOfFdHead(aicBudget),
        gS1IdxOfFdHead(aicBudget),
        s2SplitNumOfFdHead(aicBudget),
        gS1SplitNumOfFdHead(aicBudget),
        gS1LastPartSizeOfFdHead(aicBudget),
        gS1IdxEndOfFdHead(aicBudget * vecCubeRatio),
        gS1IdxEndOfFdHeadSplit(aicBudget * vecCubeRatio),
        s2SplitStartIdxOfCore(aicBudget) {}
};

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum { 0U };        // 使用的核数量
    uint32_t vecCubeRatio { 0U };        // vec 与 cube 核数比例
    std::vector<uint32_t> bN2End {};    // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End {};    // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End {};     // 每个核处理数据的S2结束点
    int64_t maxCost { 0 };            // 慢核开销
    uint32_t numOfFdHead { 0U };        // 归约任务数量
    uint32_t maxS2SplitNum { 0U };      // 单个归约任务最大分核数量
    uint32_t usedVecNumOfFd { 0U };     // 归约过程使用的vector数量
    FlashDecodeResult fdRes { 0U, 0U };     // FD信息

    SplitResult(uint32_t aicBudget, uint32_t ratio) :
        vecCubeRatio(ratio),
        bN2End(aicBudget),
        gS1End(aicBudget),
        s2End(aicBudget),
        fdRes(aicBudget, ratio) {};
};


// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum {};                   // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum {};                    // S2方向，切了多少个基本块
    std::vector<uint32_t> rowTailExtent {};                  // S1G方向，尾块size
    std::vector<uint32_t> colTailExtent {};                   // S2方向，尾块size
    bool isKvSeqAllZero { true };

    explicit SplitInfo(uint32_t batchSize) :
        s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        rowTailExtent(batchSize),
        colTailExtent(batchSize) {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch {};           // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch {};          // 整个batch的开销
    std::vector<int64_t> bN2LastBlockCostOfEachBatch {};  // batch最后一块的开销
    uint32_t totalBlockNum { 0U };
    int64_t totalCost { 0 };

    explicit CostInfo(uint32_t batchSize) :
        bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
// References must be bound in the ctor (no default brace-init of references).
struct SplitContext {
    const BaseInfo &baseMeta;
    const SplitParam &tileSpec;
    SplitInfo splitInfo;
    CostInfo costBag;

    explicit SplitContext(const BaseInfo &info, const SplitParam &spec) :
        baseMeta(info),
        tileSpec(spec),
        splitInfo(info.bSize),
        costBag(info.bSize) {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t batchSlot { 0U };
    uint32_t qTokLen { 0U };
    uint32_t kvTokLen { 0U };
    int64_t preTokenLeftUp { 0 };
    int64_t nextTokenLeftUp { 0 };
    BlockCost<int64_t> typeCost {};
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t batchSlot { 0U };
    uint32_t rowTileIdx { 0U };
    uint32_t s2Start { 0U };
    uint32_t s2End { 0U };
    int64_t s1GCost { 0 };
    int64_t s1GLastBlockCost { 0 };
    uint32_t s1GBlock { 0U };
    int64_t s1GNormalBlockCost { 0 };
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costCap { 0 };  // 负载上限
    int64_t cost { 0 };       // 已分配负载
    uint32_t block { 0U };      // 已分配块数
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint32_t curCoreIdx { 0U };
    int64_t unassignedCost { 0 };
    uint32_t usedCoreNum { 0U };
    uint32_t curKvSplitPart { 1U };

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    BatchCache batchScratch {};
    S1GCache rowScratch {};
    CoreCache coreCache {};
};

// util
uint32_t GetS1SeqSize(uint32_t batchSlot, const BaseInfo &baseMeta);
uint32_t GetS2SeqSize(uint32_t batchSlot, const BaseInfo &baseMeta);
int64_t CalcPreTokenLeftUp(uint32_t qTokLen, uint32_t kvTokLen, const BaseInfo &baseMeta);
int64_t CalcNextTokenLeftUp(uint32_t qTokLen, uint32_t kvTokLen, const BaseInfo &baseMeta);
Range<uint32_t> CalcS2Range(uint32_t rowTileIdx, const BaseInfo &baseMeta, const SplitParam &tileSpec,
    const BatchCache &batchScratch);
int64_t CalcCost(uint32_t tileRows, uint32_t tileCols);
BlockCost<int64_t> CalcCostTable(uint32_t rowNormExtent, uint32_t colNormExtent, uint32_t rowTailExtent,
    uint32_t colTailExtent);

// cache calculation
void CalcBatchCache(uint32_t batchSlot, const SplitContext &planCtx, BatchCache &batchScratch);
void CalcS1GCache(uint32_t rowTileIdx, const SplitContext &planCtx, const BatchCache &batchScratch, S1GCache &rowScratch);
void CopyTmpResult(SplitResult &scratchRes, SplitResult &outPlan);
void ClearTmpResult(SplitResult &scratchPlan);

// preprocess
void CalcSplitInfo(SplitContext &planCtx);
void CalcBatchCost(uint32_t batchSlot, const SplitContext &planCtx, CostInfo &costBag);
void CalcCostInfo(SplitContext &planCtx);

// assign
void UpdateCursor(const SplitContext &planCtx, AssignContext &assignState);
void AssignByBatch(const SplitContext &planCtx, AssignContext &assignState);
void AssignByRow(const SplitContext &planCtx, AssignContext &assignState);
void AssignByBlock(const SplitContext &planCtx, AssignContext &assignState);
void ForceAssign(const SplitContext &planCtx, AssignContext &assignState);

// FD
bool IsNeedRecordFDInfo(const AssignContext &assignState, const SplitResult &outPlan);
void RecordFDInfo(const SplitContext &planCtx, const AssignContext &assignState, SplitResult &planOut);

// main
void SplitFD(SplitResult &planOut);
void CalcSplitPlan(uint32_t aicBudget, int64_t costCap, const SplitContext &planCtx, SplitResult &planOut);
void SplitCore(uint32_t aicBudget, const BaseInfo &baseMeta, const SplitParam &tileSpec, SplitResult &planOut);
}
#endif
