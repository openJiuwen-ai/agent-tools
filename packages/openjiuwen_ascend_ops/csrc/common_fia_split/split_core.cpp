/*!
 * \file split_core.cpp
 * \brief Host tiling helper that partitions FIA work across AIC cores (RQ-uniform ops).
 *        Sole TU providing optiling::SplitCore — linked once via the FIA host CMakeLists.
 */

#include "split_core.h"
#include <cstdio>
#include <math.h>


namespace optiling {

uint32_t GetS1SeqSize(uint32_t batchSlot, const BaseInfo &baseMeta)
{
    const auto &lens = baseMeta.actualSeqS1Size;
    if (lens.empty()) {
        return baseMeta.qTokLen;
    }
    if (baseMeta.actualLenQDims == 1U) {
        return static_cast<uint32_t>(lens[0]);
    }
    if (baseMeta.isAccumSeqS1) {
        if (batchSlot == 0U) {
            return static_cast<uint32_t>(lens[batchSlot]);
        }
        return static_cast<uint32_t>(lens[batchSlot] - lens[batchSlot - 1U]);
    }
    return static_cast<uint32_t>(lens[batchSlot]);
}

uint32_t GetS2SeqSize(uint32_t batchSlot, const BaseInfo &baseMeta)
{
    const uint32_t prefix = static_cast<uint32_t>(baseMeta.actualSeqPrefixSize);
    const auto &lens = baseMeta.actualSeqS2Size;
    if (lens.empty()) {
        return prefix + baseMeta.kvTokLen;
    }
    if (baseMeta.actualLenKvDims == 1U) {
        return prefix + static_cast<uint32_t>(lens[0]);
    }
    if (baseMeta.isAccumSeqS2) {
        if (batchSlot == 0U) {
            return prefix + static_cast<uint32_t>(lens[batchSlot]);
        }
        return prefix + static_cast<uint32_t>(lens[batchSlot] - lens[batchSlot - 1U]);
    }
    return prefix + static_cast<uint32_t>(lens[batchSlot]);
}

int64_t CalcPreTokenLeftUp(uint32_t qTokLen, uint32_t kvTokLen, const BaseInfo &baseMeta)
{
    auto mode = static_cast<SparseMode>(baseMeta.sparseMode);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(qTokLen) - static_cast<int64_t>(kvTokLen) + baseMeta.preToken;
    }
    return baseMeta.preToken;
}

int64_t CalcNextTokenLeftUp(uint32_t qTokLen, uint32_t kvTokLen, const BaseInfo &baseMeta)
{
    auto mode = static_cast<SparseMode>(baseMeta.sparseMode);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return baseMeta.nextToken;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(kvTokLen) - static_cast<int64_t>(qTokLen);
        case SparseMode::BAND:
            return static_cast<int64_t>(kvTokLen) - static_cast<int64_t>(qTokLen) + baseMeta.nextToken;
        default:
            return baseMeta.nextToken;
    }
}

int64_t CalcCost(uint32_t tileRows, uint32_t tileCols)
{
    uint32_t mAlignUnit = 16U;
    uint32_t s2AlignUnit = 64U;
    uint32_t alignedTileM = (tileRows + mAlignUnit - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignedTileS2 = (tileCols + s2AlignUnit - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignedTileM + 10U * alignedTileS2);                 // 6：M轴系数，10：S2轴系数
}

BlockCost<int64_t> CalcCostTable(uint32_t rowNormExtent, uint32_t colNormExtent, uint32_t rowTailExtent,
    uint32_t colTailExtent)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(rowNormExtent, colNormExtent);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (rowTailExtent == 0U) ? 0U : CalcCost(rowTailExtent, colNormExtent);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (colTailExtent == 0U) ? 0U : CalcCost(rowNormExtent, colTailExtent);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (rowTailExtent == 0U || colTailExtent == 0U) ? 0U : CalcCost(rowTailExtent, colTailExtent);
    return typeCost;
}

Range<uint32_t> CalcS2Range(uint32_t rowTileIdx, const BaseInfo &baseMeta, const SplitParam &tileSpec,
    const BatchCache &batchScratch)
{
    if (batchScratch.qTokLen == 0U || batchScratch.kvTokLen == 0U) {
        return {0U, 0U};
    }

    if (!baseMeta.attenMaskFlag) {
        const uint32_t tiles =
            (batchScratch.kvTokLen + tileSpec.s2BaseSize - 1U) / tileSpec.s2BaseSize;
        return {0U, tiles};
    }

    const int64_t mBase = static_cast<int64_t>(tileSpec.mBaseSize);
    const int64_t g = static_cast<int64_t>(baseMeta.gSize);
    const int64_t qLen = static_cast<int64_t>(batchScratch.qTokLen);
    const int64_t begin = static_cast<int64_t>(rowTileIdx) * mBase;
    const int64_t end = std::min(begin + mBase, qLen * g) - 1;

    int64_t qFirst = 0;
    int64_t qLast = 0;
    if (baseMeta.isS1G) {
        qFirst = begin / g;
        qLast = end / g;
    } else if (begin / qLen == end / qLen) {
        qFirst = begin % qLen;
        qLast = end % qLen;
    } else {
        qFirst = 0;
        qLast = qLen;
    }

    int64_t kvLo = qFirst - batchScratch.preTokenLeftUp;
    int64_t kvHi = qLast + batchScratch.nextTokenLeftUp;
    if (kvLo >= static_cast<int64_t>(batchScratch.kvTokLen) || kvHi < 0 || kvHi < kvLo) {
        return {0U, 0U};
    }

    const int64_t kvMax = static_cast<int64_t>(batchScratch.kvTokLen) - 1;
    kvLo = Clip(kvLo, int64_t{0}, kvMax);
    kvHi = Clip(kvHi, int64_t{0}, kvMax);
    const uint32_t loTile = static_cast<uint32_t>(kvLo) / tileSpec.s2BaseSize;
    const uint32_t hiTile = static_cast<uint32_t>(kvHi) / tileSpec.s2BaseSize + 1U;
    return {loTile, hiTile};
}

void CalcSplitInfo(SplitContext &planCtx)
{
    const BaseInfo &baseMeta = planCtx.baseMeta;
    const SplitParam &tileSpec = planCtx.tileSpec;

    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = planCtx.splitInfo;
    for (uint32_t batchSlot = 0; batchSlot < baseMeta.bSize; batchSlot++) {
        uint32_t qTokLen = GetS1SeqSize(batchSlot, baseMeta);
        uint32_t kvTokLen = GetS2SeqSize(batchSlot, baseMeta);

        splitInfo.s1GBaseNum[batchSlot] = (qTokLen * baseMeta.gSize + (tileSpec.mBaseSize - 1U)) / tileSpec.mBaseSize;
        splitInfo.rowTailExtent[batchSlot] = (qTokLen * baseMeta.gSize) % tileSpec.mBaseSize;
        splitInfo.s2BaseNum[batchSlot] = (kvTokLen + tileSpec.s2BaseSize - 1U) / tileSpec.s2BaseSize;
        splitInfo.colTailExtent[batchSlot] = kvTokLen % tileSpec.s2BaseSize;
        if (splitInfo.s1GBaseNum[batchSlot] != 0U && splitInfo.s2BaseNum[batchSlot] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

void CalcBatchCache(uint32_t batchSlot, const SplitContext &planCtx, BatchCache &batchScratch)
{
    const BaseInfo &caseInfo = planCtx.baseMeta;
    const SplitParam &tileCfg = planCtx.tileSpec;
    const SplitInfo &tileMeta = planCtx.splitInfo;

    batchScratch.batchSlot = batchSlot;
    const uint32_t qLen = GetS1SeqSize(batchSlot, caseInfo);
    const uint32_t kvLen = GetS2SeqSize(batchSlot, caseInfo);
    batchScratch.qTokLen = qLen;
    batchScratch.kvTokLen = kvLen;
    batchScratch.preTokenLeftUp = CalcPreTokenLeftUp(qLen, kvLen, caseInfo);
    batchScratch.nextTokenLeftUp = CalcNextTokenLeftUp(qLen, kvLen, caseInfo);
    batchScratch.typeCost = CalcCostTable(tileCfg.mBaseSize, tileCfg.s2BaseSize, tileMeta.rowTailExtent[batchSlot],
        tileMeta.colTailExtent[batchSlot]);
}

void CalcS1GCache(uint32_t rowTileIdx, const SplitContext &planCtx, const BatchCache &batchScratch, S1GCache &rowScratch)
{
    const BaseInfo &caseInfo = planCtx.baseMeta;
    const SplitParam &tileCfg = planCtx.tileSpec;
    const SplitInfo &tileMeta = planCtx.splitInfo;
    const uint32_t batchSlot = batchScratch.batchSlot;

    rowScratch.batchSlot = batchSlot;
    rowScratch.rowTileIdx = rowTileIdx;

    const auto kvSpan = CalcS2Range(rowTileIdx, caseInfo, tileCfg, batchScratch);
    rowScratch.s2Start = kvSpan.first;
    rowScratch.s2End = kvSpan.second;

    const bool emptySpan = (rowScratch.s2Start >= rowScratch.s2End);
    if (emptySpan) {
        rowScratch.s1GBlock = 0U;
        rowScratch.s1GCost = 0;
        rowScratch.s1GLastBlockCost = 0;
        rowScratch.s1GNormalBlockCost = 0;
        return;
    }

    rowScratch.s1GBlock = rowScratch.s2End - rowScratch.s2Start;
    const bool touchesTail =
        (tileMeta.colTailExtent[batchSlot] != 0U) && (rowScratch.s2End == tileMeta.s2BaseNum[batchSlot]);
    const uint32_t tailTiles = touchesTail ? 1U : 0U;
    const uint32_t fullTiles = rowScratch.s1GBlock - tailTiles;
    const uint32_t rowTiles = tileMeta.s1GBaseNum[batchSlot];
    const bool isLastRowTail =
        (rowTiles != 0U) && (rowTileIdx == (rowTiles - 1U)) && (tileMeta.rowTailExtent[batchSlot] != 0U);

    if (rowTiles == 0U) {
        rowScratch.s1GCost = 0;
        rowScratch.s1GLastBlockCost = 0;
        rowScratch.s1GNormalBlockCost = 0;
        return;
    }

    const auto &costMat = batchScratch.typeCost;
    if (isLastRowTail) {
        const int64_t fullCost = costMat[TAIL_BLOCK][NORMAL_BLOCK];
        const int64_t tailCost = costMat[TAIL_BLOCK][TAIL_BLOCK];
        rowScratch.s1GNormalBlockCost = fullCost;
        rowScratch.s1GLastBlockCost = (tailTiles > 0U) ? tailCost : fullCost;
        rowScratch.s1GCost = fullCost * static_cast<int64_t>(fullTiles) + tailCost * static_cast<int64_t>(tailTiles);
        return;
    }

    const int64_t fullCost = costMat[NORMAL_BLOCK][NORMAL_BLOCK];
    const int64_t tailCost = costMat[NORMAL_BLOCK][TAIL_BLOCK];
    rowScratch.s1GNormalBlockCost = fullCost;
    rowScratch.s1GLastBlockCost = (tailTiles > 0U) ? tailCost : fullCost;
    rowScratch.s1GCost = fullCost * static_cast<int64_t>(fullTiles) + tailCost * static_cast<int64_t>(tailTiles);
}

void CalcBatchCost(uint32_t batchSlot, const SplitContext &planCtx, CostInfo &costBag)
{
    const BaseInfo &baseMeta = planCtx.baseMeta;
    const SplitInfo &splitInfo = planCtx.splitInfo;

    costBag.bN2CostOfEachBatch[batchSlot] = 0;
    costBag.bN2BlockOfEachBatch[batchSlot] = 0U;
    costBag.bN2LastBlockCostOfEachBatch[batchSlot] = 0U;

    if (GetS1SeqSize(batchSlot, baseMeta) == 0U || GetS2SeqSize(batchSlot, baseMeta) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache rowScratch;
    CalcBatchCache(batchSlot, planCtx, bCache);
    for (uint32_t rowTileIdx = 0; rowTileIdx < splitInfo.s1GBaseNum[batchSlot]; rowTileIdx++) {
        CalcS1GCache(rowTileIdx, planCtx, bCache, rowScratch);
        costBag.bN2CostOfEachBatch[batchSlot] += rowScratch.s1GCost;
        costBag.bN2BlockOfEachBatch[batchSlot] += rowScratch.s1GBlock;

        if (rowScratch.s1GBlock > 0) {
            costBag.bN2LastBlockCostOfEachBatch[batchSlot] = rowScratch.s1GLastBlockCost;
        }
    }
}

void CalcCostInfo(SplitContext &planCtx)
{
    CostInfo &costBag = planCtx.costBag;
    if (planCtx.splitInfo.isKvSeqAllZero) {
        costBag.totalCost = 0;
        costBag.totalBlockNum = 0U;
        return;
    }

    const uint32_t batches = planCtx.baseMeta.bSize;
    const uint32_t n2 = planCtx.baseMeta.n2Size;
    costBag.totalCost = 0;
    costBag.totalBlockNum = 0U;
    uint32_t batch = 0U;
    while (batch < batches) {
        CalcBatchCost(batch, planCtx, costBag);
        costBag.totalCost += costBag.bN2CostOfEachBatch[batch] * static_cast<int64_t>(n2);
        costBag.totalBlockNum += costBag.bN2BlockOfEachBatch[batch] * n2;
        ++batch;
    }
}

void CopyTmpResult(SplitResult &scratchRes, SplitResult &outPlan)
{
    const size_t coreSlots = scratchRes.bN2End.size();
    outPlan.usedCoreNum = scratchRes.usedCoreNum;
    outPlan.maxCost = scratchRes.maxCost;
    outPlan.numOfFdHead = scratchRes.numOfFdHead;
    outPlan.maxS2SplitNum = scratchRes.maxS2SplitNum;

    size_t slot = 0;
    while (slot < coreSlots) {
        outPlan.bN2End[slot] = scratchRes.bN2End[slot];
        outPlan.gS1End[slot] = scratchRes.gS1End[slot];
        outPlan.s2End[slot] = scratchRes.s2End[slot];
        outPlan.fdRes.bN2IdxOfFdHead[slot] = scratchRes.fdRes.bN2IdxOfFdHead[slot];
        outPlan.fdRes.gS1IdxOfFdHead[slot] = scratchRes.fdRes.gS1IdxOfFdHead[slot];
        outPlan.fdRes.s2SplitNumOfFdHead[slot] = scratchRes.fdRes.s2SplitNumOfFdHead[slot];
        outPlan.fdRes.s2SplitStartIdxOfCore[slot] = scratchRes.fdRes.s2SplitStartIdxOfCore[slot];
        outPlan.fdRes.gS1SplitNumOfFdHead[slot] = scratchRes.fdRes.gS1SplitNumOfFdHead[slot];
        outPlan.fdRes.gS1LastPartSizeOfFdHead[slot] = scratchRes.fdRes.gS1LastPartSizeOfFdHead[slot];
        ++slot;
    }
}

void ClearTmpResult(SplitResult &scratchPlan)
{
    const size_t coreSlots = scratchPlan.bN2End.size();
    scratchPlan.usedCoreNum = 0U;
    scratchPlan.maxCost = 0;
    scratchPlan.numOfFdHead = 0U;
    scratchPlan.maxS2SplitNum = 0U;
    scratchPlan.usedVecNumOfFd = 0U;

    size_t slot = 0;
    while (slot < coreSlots) {
        scratchPlan.bN2End[slot] = 0U;
        scratchPlan.gS1End[slot] = 0U;
        scratchPlan.s2End[slot] = 0U;
        scratchPlan.fdRes.bN2IdxOfFdHead[slot] = 0U;
        scratchPlan.fdRes.gS1IdxOfFdHead[slot] = 0U;
        scratchPlan.fdRes.s2SplitNumOfFdHead[slot] = 0U;
        scratchPlan.fdRes.s2SplitStartIdxOfCore[slot] = 0U;
        scratchPlan.fdRes.gS1SplitNumOfFdHead[slot] = 0U;
        scratchPlan.fdRes.gS1LastPartSizeOfFdHead[slot] = 0U;
        ++slot;
    }
}

void UpdateCursor(const SplitContext &planCtx, AssignContext &assignState)
{
    const BaseInfo &baseMeta = planCtx.baseMeta;
    const SplitInfo &splitInfo = planCtx.splitInfo;
    const CostInfo &costBag = planCtx.costBag;

    enum class CursorBump : uint8_t { kNone = 0, kRow = 1, kBatch = 2 };
    CursorBump bump = CursorBump::kNone;

    // S2 open-interval exhausted → advance row tile.
    if (assignState.curS2Idx >= assignState.rowScratch.s2End) {
        assignState.curS2Idx = 0U;
        assignState.curS1GIdx += 1U;
        bump = CursorBump::kRow;
    }

    // Row tiles exhausted for this BN2 → advance BN2 cursor.
    if (assignState.curS1GIdx >= splitInfo.s1GBaseNum[assignState.curBIdx]) {
        assignState.curS1GIdx = 0U;
        assignState.curBN2Idx += 1U;
    }

    const uint32_t bn2Limit = baseMeta.bSize * baseMeta.n2Size;
    if (assignState.curBN2Idx == bn2Limit) {
        assignState.curS1GIdx = 0U;
        assignState.curS2Idx = 0U;
        assignState.isFinished = true;
        return;
    }

    const uint32_t nextBatch = assignState.curBN2Idx / baseMeta.n2Size;
    if (nextBatch != assignState.curBIdx) {
        assignState.curBIdx = nextBatch;
        assignState.curS1GIdx = 0U;
        bump = CursorBump::kBatch;
    }

    switch (bump) {
        case CursorBump::kBatch:
            CalcBatchCache(assignState.curBIdx, planCtx, assignState.batchScratch);
            assignState.bN2Cost = costBag.bN2CostOfEachBatch[assignState.curBIdx];
            assignState.bN2Block = costBag.bN2BlockOfEachBatch[assignState.curBIdx];
            CalcS1GCache(assignState.curS1GIdx, planCtx, assignState.batchScratch, assignState.rowScratch);
            assignState.curS2Idx = assignState.rowScratch.s2Start;
            break;
        case CursorBump::kRow:
            CalcS1GCache(assignState.curS1GIdx, planCtx, assignState.batchScratch, assignState.rowScratch);
            assignState.curS2Idx = assignState.rowScratch.s2Start;
            break;
        case CursorBump::kNone:
        default:
            break;
    }
}

void AssignByBatch(const SplitContext &planCtx, AssignContext &assignState)
{
    if (assignState.isFinished) {
        return;
    }

    const BaseInfo &baseMeta = planCtx.baseMeta;
    const CostInfo &costBag = planCtx.costBag;
    const uint32_t bn2Limit = baseMeta.bSize * baseMeta.n2Size;

    for (;;) {
        const int64_t slack = costBag.bN2LastBlockCostOfEachBatch[assignState.curBIdx] / FA_TOLERANCE_RATIO;
        const bool emptyBatch = (assignState.bN2Cost == 0);
        const bool fits = IsWithinTolerance(assignState.coreCache.costCap, slack,
            assignState.coreCache.cost + assignState.bN2Cost);
        if (!emptyBatch && !fits) {
            break;
        }

        assignState.coreCache.cost += assignState.bN2Cost;
        assignState.coreCache.block += assignState.bN2Block;
        assignState.curBN2Idx += 1U;

        if (assignState.curBN2Idx == bn2Limit) {
            assignState.curS1GIdx = 0U;
            assignState.curS2Idx = 0U;
            assignState.isFinished = true;
            return;
        }

        const uint32_t batchNow = assignState.curBN2Idx / baseMeta.n2Size;
        if (batchNow != assignState.curBIdx) {
            assignState.curBIdx = batchNow;
            CalcBatchCache(assignState.curBIdx, planCtx, assignState.batchScratch);
        }

        assignState.bN2Cost = costBag.bN2CostOfEachBatch[assignState.curBIdx];
        assignState.bN2Block = costBag.bN2BlockOfEachBatch[assignState.curBIdx];
        assignState.curS1GIdx = 0U;
        CalcS1GCache(0U, planCtx, assignState.batchScratch, assignState.rowScratch);
        assignState.curS2Idx = assignState.rowScratch.s2Start;
    }
}

void AssignByRow(const SplitContext &planCtx, AssignContext &assignState)
{
    if (assignState.isFinished) {
        return;
    }

    auto takeWholeRow = [&]() -> bool {
        const int64_t rowSlack = assignState.rowScratch.s1GLastBlockCost / FA_TOLERANCE_RATIO;
        return IsWithinTolerance(assignState.coreCache.costCap, rowSlack,
            assignState.coreCache.cost + assignState.rowScratch.s1GCost);
    };

    while (takeWholeRow()) {
        const int64_t rowCost = assignState.rowScratch.s1GCost;
        const uint32_t rowBlocks = assignState.rowScratch.s1GBlock;
        assignState.coreCache.cost += rowCost;
        assignState.coreCache.block += rowBlocks;

        assignState.bN2Cost = (assignState.bN2Cost > rowCost) ? (assignState.bN2Cost - rowCost) : 0;
        assignState.bN2Block = (assignState.bN2Block > rowBlocks) ? (assignState.bN2Block - rowBlocks) : 0U;

        // Skip empty row tiles until a non-empty S1G cache is materialised.
        do {
            assignState.curS1GIdx += 1U;
            CalcS1GCache(assignState.curS1GIdx, planCtx, assignState.batchScratch, assignState.rowScratch);
        } while (assignState.rowScratch.s1GBlock == 0U);
        assignState.curS2Idx = assignState.rowScratch.s2Start;
    }
}

void AssignByBlock(const SplitContext &planCtx, AssignContext &assignState)
{
    (void)planCtx;
    if (assignState.isFinished) {
        return;
    }

    // Cost is latched once at entry (matches historical FIA SplitCore): subsequent
    // blocks reuse the same piece value even when the open interval reaches the tail.
    int64_t piece = assignState.rowScratch.s1GNormalBlockCost;
    if (assignState.curS2Idx + 1U == assignState.rowScratch.s2End) {
        piece = assignState.rowScratch.s1GLastBlockCost;
    }

    for (;;) {
        if (!IsWithinTolerance(assignState.coreCache.costCap, piece / FA_TOLERANCE_RATIO,
                               assignState.coreCache.cost + piece)) {
            break;
        }
        assignState.coreCache.cost += piece;
        assignState.coreCache.block += 1U;
        assignState.curS2Idx += 1U;
        assignState.bN2Cost -= piece;
        assignState.rowScratch.s1GCost -= piece;
        assignState.bN2Block -= 1U;
        assignState.rowScratch.s1GBlock -= 1U;
    }
}

void ForceAssign(const SplitContext &planCtx, AssignContext &assignState)
{
    if (assignState.isFinished) {
        return;
    }

    int64_t curCost = assignState.rowScratch.s1GNormalBlockCost;
    if (assignState.curS2Idx == (assignState.rowScratch.s2End - 1U)) {
        curCost = assignState.rowScratch.s1GLastBlockCost;
    }

    assignState.coreCache.cost += curCost;
    assignState.coreCache.block++;
    assignState.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignState.bN2Cost = assignState.bN2Cost - curCost;
    assignState.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignState.rowScratch.s1GCost = assignState.rowScratch.s1GCost - curCost;
    assignState.rowScratch.s1GBlock--;
    UpdateCursor(planCtx, assignState);
}

bool IsNeedRecordFDInfo(const AssignContext &assignState, const SplitResult &outPlan)
{
    // FD bookkeeping is deferred until the next cut; skip core0 / single-part / unfinished rows.
    const bool hasPriorCore = (assignState.curCoreIdx > 0U);
    const bool crossedKvParts = (assignState.curKvSplitPart > 1U);
    if (!hasPriorCore || !crossedKvParts) {
        return false;
    }
    const uint32_t prevCore = assignState.curCoreIdx - 1U;
    const bool stillOnPrevCutRow =
        (assignState.curBN2Idx == outPlan.bN2End[prevCore]) &&
        (assignState.curS1GIdx == outPlan.gS1End[prevCore]);
    return !stillOnPrevCutRow;
}

void RecordFDInfo(const SplitContext &planCtx, const AssignContext &assignState, SplitResult &planOut)
{
    const BaseInfo &baseMeta = planCtx.baseMeta;
    const SplitParam &tileSpec = planCtx.tileSpec;
    const SplitInfo &splitInfo = planCtx.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = planOut.bN2End[assignState.curCoreIdx - 1U] / baseMeta.n2Size;
    uint32_t splitS1GIdx = planOut.gS1End[assignState.curCoreIdx - 1U];
    uint32_t qTokLen = GetS1SeqSize(splitBIdx, baseMeta);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (qTokLen * baseMeta.gSize - splitS1GIdx * tileSpec.mBaseSize) : tileSpec.mBaseSize;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + tileSpec.gS1BaseSizeOfFd - 1U) / tileSpec.gS1BaseSizeOfFd;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (tileSpec.gS1BaseSizeOfFd * (curFdS1gSplitPart - 1U));
    // 记录
    planOut.maxS2SplitNum = std::max(planOut.maxS2SplitNum, assignState.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    planOut.fdRes.bN2IdxOfFdHead[planOut.numOfFdHead] = planOut.bN2End[assignState.curCoreIdx - 1U];
    planOut.fdRes.gS1IdxOfFdHead[planOut.numOfFdHead] = planOut.gS1End[assignState.curCoreIdx - 1U];
    planOut.fdRes.s2SplitNumOfFdHead[planOut.numOfFdHead] = assignState.curKvSplitPart;
    planOut.fdRes.gS1SplitNumOfFdHead[planOut.numOfFdHead] = curFdS1gSplitPart;
    planOut.fdRes.gS1LastPartSizeOfFdHead[planOut.numOfFdHead] = curFdS1gLastPartSize;
    planOut.numOfFdHead++;
}

void CalcSplitPlan(uint32_t aicBudget, int64_t costCap, const SplitContext &planCtx, SplitResult &planOut)
{
    const CostInfo &costBag = planCtx.costBag;
    if (aicBudget == 0U) {
        return;
    }

    planOut.maxCost = 0;
    planOut.usedCoreNum = 0U;

    AssignContext cursor {};
    cursor.curBIdx = 0U;
    cursor.curS1GIdx = 0U;
    cursor.unassignedCost = costBag.totalCost;
    cursor.bN2Cost = costBag.bN2CostOfEachBatch[0];
    cursor.bN2Block = costBag.bN2BlockOfEachBatch[0];
    CalcBatchCache(0U, planCtx, cursor.batchScratch);
    CalcS1GCache(0U, planCtx, cursor.batchScratch, cursor.rowScratch);
    cursor.curS2Idx = cursor.rowScratch.s2Start;

    using PackStep = void (*)(const SplitContext &, AssignContext &);
    static PackStep kPackPipeline[] = {AssignByBatch, AssignByRow, AssignByBlock};

    for (uint32_t core = 0U; core < aicBudget; ++core) {
        if (planOut.maxCost > costCap) {
            return;
        }
        if (cursor.isFinished || cursor.unassignedCost <= 0) {
            break;
        }

        cursor.curCoreIdx = core;
        planOut.fdRes.s2SplitStartIdxOfCore[core] = cursor.curKvSplitPart - 1U;
        cursor.coreCache = {};
        cursor.coreCache.costCap = cursor.unassignedCost / static_cast<int64_t>(aicBudget - core);

        for (PackStep step : kPackPipeline) {
            step(planCtx, cursor);
        }
        if (cursor.coreCache.block == 0U) {
            ForceAssign(planCtx, cursor);
        }

        planOut.bN2End[core] = cursor.curBN2Idx;
        planOut.gS1End[core] = cursor.curS1GIdx;
        planOut.s2End[core] = cursor.curS2Idx;
        planOut.maxCost = std::max(planOut.maxCost, cursor.coreCache.cost);
        cursor.unassignedCost -= cursor.coreCache.cost;

        if (IsNeedRecordFDInfo(cursor, planOut)) {
            RecordFDInfo(planCtx, cursor, planOut);
            cursor.curKvSplitPart = 1U;
        }

        const bool midRow = (cursor.curS2Idx > cursor.rowScratch.s2Start) &&
                            (cursor.curS2Idx <= cursor.rowScratch.s2End);
        if (midRow) {
            cursor.curKvSplitPart += 1U;
        }
    }

    planOut.usedCoreNum = cursor.curCoreIdx + 1U;
}

void SplitFD(SplitResult &planOut)
{
    uint32_t fdTokenLoad = 0U;
    uint32_t fdHeadPieces = 0U;
    for (uint32_t head = 0U; head < planOut.numOfFdHead; ++head) {
        fdTokenLoad += planOut.fdRes.s2SplitNumOfFdHead[head] * planOut.fdRes.gS1SplitNumOfFdHead[head];
        fdHeadPieces += planOut.fdRes.gS1SplitNumOfFdHead[head];
    }

    const uint32_t vecCap = std::min(fdHeadPieces, planOut.usedCoreNum * planOut.vecCubeRatio);
    double thr = (vecCap == 0U) ? 0.0 : static_cast<double>(fdTokenLoad) / static_cast<double>(vecCap);
    int64_t packed = 0;
    uint32_t vecIdx = 0U;
    uint32_t lastHead = 0U;
    uint32_t lastPiece = 0U;

    for (uint32_t head = 0U; head < planOut.numOfFdHead; ++head) {
        const uint32_t kvParts = planOut.fdRes.s2SplitNumOfFdHead[head];
        const uint32_t pieceCount = planOut.fdRes.gS1SplitNumOfFdHead[head];
        for (uint32_t piece = 0U; piece < pieceCount; ++piece) {
            const double room = thr - static_cast<double>(packed);
            if (static_cast<double>(kvParts) > room * FD_TOLERANCE_RATIO) {
                planOut.fdRes.gS1IdxEndOfFdHead[vecIdx] = lastHead;
                planOut.fdRes.gS1IdxEndOfFdHeadSplit[vecIdx] = lastPiece;
                vecIdx += 1U;
                fdTokenLoad -= static_cast<uint32_t>(packed);
                const uint32_t remainVec = vecCap - vecIdx;
                thr = (remainVec == 0U) ? 0.0 :
                      static_cast<double>(fdTokenLoad) / static_cast<double>(remainVec);
                packed = 0;
            }
            packed += static_cast<int64_t>(kvParts);
            lastHead = head;
            lastPiece = piece;
        }
    }

    planOut.fdRes.gS1IdxEndOfFdHead[vecIdx] = lastHead;
    planOut.fdRes.gS1IdxEndOfFdHeadSplit[vecIdx] = lastPiece;
    planOut.usedVecNumOfFd = vecIdx + 1U;
}

void SplitCore(uint32_t aicBudget, const BaseInfo &baseMeta, const SplitParam &tileSpec, SplitResult &planOut)
{
    SplitContext ctx(baseMeta, tileSpec);
    CalcSplitInfo(ctx);

    if (ctx.splitInfo.isKvSeqAllZero) {
        planOut.usedCoreNum = 1U;
        planOut.bN2End[0] = baseMeta.bSize * baseMeta.n2Size;
        planOut.gS1End[0] = 0U;
        planOut.s2End[0] = 0U;
        return;
    }

    CalcCostInfo(ctx);

    const uint32_t totalBlocks = ctx.costBag.totalBlockNum;
    uint32_t coreHi = std::min(aicBudget, totalBlocks);
    uint32_t coreLo = static_cast<uint32_t>(std::sqrt(static_cast<float>(totalBlocks) + 0.25f) + 0.5f);
    if (coreLo > coreHi) {
        coreLo = coreHi;
    }

    planOut.maxCost = INT64_MAX;
    planOut.usedCoreNum = 1U;

    SplitResult scratch {aicBudget, planOut.vecCubeRatio};
    uint32_t trialCores = coreLo;
    while (trialCores <= coreHi) {
        CalcSplitPlan(trialCores, planOut.maxCost, ctx, scratch);
        if (scratch.maxCost < planOut.maxCost) {
            CopyTmpResult(scratch, planOut);
        }
        ClearTmpResult(scratch);
        ++trialCores;
    }

    if (planOut.numOfFdHead > 0U) {
        SplitFD(planOut);
    }
    planOut.usedCoreNum = std::max(planOut.usedCoreNum, 1U);
}
} // namespace optiling
