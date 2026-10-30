/*!
 * \file test_split_core.cpp
 * \brief Host-only regression for rewritten optiling::SplitCore helpers.
 *
 * Covers every free function in common_fia_split/split_core.cpp that was
 * structurally rewritten for SCA. Build via tests/ut/test_split_core_host.py.
 */
#include "split_core.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace optiling;

namespace {

int gFails = 0;

#define EXPECT_TRUE(cond)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                   \
            ++gFails;                                                                              \
        }                                                                                          \
    } while (0)

#define EXPECT_EQ(a, b)                                                                            \
    do {                                                                                           \
        const auto _va = (a);                                                                      \
        const auto _vb = (b);                                                                      \
        if (!(_va == _vb)) {                                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s (== %lld) vs %s (== %lld)\n", __FILE__, __LINE__,  \
                         #a, static_cast<long long>(_va), #b, static_cast<long long>(_vb));         \
            ++gFails;                                                                              \
        }                                                                                          \
    } while (0)

BaseInfo MakeBase(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t s2)
{
    BaseInfo info {};
    info.bSize = b;
    info.n2Size = n2;
    info.gSize = g;
    info.qTokLen = s1;
    info.kvTokLen = s2;
    info.isS1G = true;
    info.isAccumSeqS1 = false;
    info.isAccumSeqS2 = false;
    info.actualLenQDims = b;
    info.actualLenKvDims = b;
    info.attenMaskFlag = false;
    info.sparseMode = static_cast<int32_t>(SparseMode::DEFAULT_MASK);
    info.preToken = 65536;
    info.nextToken = 65536;
    return info;
}

SplitParam MakeParam(uint32_t mBase = 128, uint32_t s2Base = 512)
{
    SplitParam p {};
    p.mBaseSize = mBase;
    p.s2BaseSize = s2Base;
    p.gS1BaseSizeOfFd = 8U;
    return p;
}

void TestGetS1S2SeqSize()
{
    // empty actual lens → fixed sizes
    BaseInfo plain = MakeBase(2, 1, 1, 16, 64);
    EXPECT_EQ(GetS1SeqSize(0, plain), 16U);
    EXPECT_EQ(GetS2SeqSize(1, plain), 64U);

    // dims==1 broadcasts first entry
    BaseInfo oneDim = MakeBase(3, 1, 1, 0, 0);
    oneDim.actualLenQDims = 1U;
    oneDim.actualLenKvDims = 1U;
    oneDim.actualSeqS1Size = {11};
    oneDim.actualSeqS2Size = {22};
    EXPECT_EQ(GetS1SeqSize(2, oneDim), 11U);
    EXPECT_EQ(GetS2SeqSize(2, oneDim), 22U);

    // non-accum per-batch
    BaseInfo perBatch = MakeBase(2, 1, 1, 0, 0);
    perBatch.actualSeqS1Size = {7, 9};
    perBatch.actualSeqS2Size = {13, 17};
    EXPECT_EQ(GetS1SeqSize(0, perBatch), 7U);
    EXPECT_EQ(GetS1SeqSize(1, perBatch), 9U);
    EXPECT_EQ(GetS2SeqSize(1, perBatch), 17U);

    // accum Q/KV
    BaseInfo accum = MakeBase(3, 1, 1, 0, 0);
    accum.isAccumSeqS1 = true;
    accum.isAccumSeqS2 = true;
    accum.actualSeqS1Size = {4, 10, 18};
    accum.actualSeqS2Size = {5, 15, 30};
    EXPECT_EQ(GetS1SeqSize(0, accum), 4U);
    EXPECT_EQ(GetS1SeqSize(1, accum), 6U);
    EXPECT_EQ(GetS1SeqSize(2, accum), 8U);
    EXPECT_EQ(GetS2SeqSize(0, accum), 5U);
    EXPECT_EQ(GetS2SeqSize(2, accum), 15U);

    // prefix on KV
    BaseInfo pref = MakeBase(1, 1, 1, 8, 0);
    pref.actualSeqPrefixSize = 3;
    pref.actualSeqS2Size = {10};
    pref.actualLenKvDims = 1U;
    EXPECT_EQ(GetS2SeqSize(0, pref), 13U);
}

void TestTokenLeftUpAllSparseModes()
{
    BaseInfo info = MakeBase(1, 1, 1, 32, 64);
    info.preToken = 5;
    info.nextToken = 7;

    info.sparseMode = static_cast<int32_t>(SparseMode::BAND);
    EXPECT_EQ(CalcPreTokenLeftUp(32, 64, info), static_cast<int64_t>(32) - 64 + 5);
    EXPECT_EQ(CalcNextTokenLeftUp(32, 64, info), static_cast<int64_t>(64) - 32 + 7);

    info.sparseMode = static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    EXPECT_EQ(CalcNextTokenLeftUp(32, 64, info), static_cast<int64_t>(64) - 32);

    for (auto mode : {SparseMode::DEFAULT_MASK, SparseMode::ALL_MASK, SparseMode::LEFT_UP_CAUSAL}) {
        info.sparseMode = static_cast<int32_t>(mode);
        EXPECT_EQ(CalcPreTokenLeftUp(32, 64, info), 5);
        EXPECT_EQ(CalcNextTokenLeftUp(32, 64, info), 7);
    }
}

void TestCalcCostAndTable()
{
    const int64_t c = CalcCost(16, 64);
    EXPECT_TRUE(c > 0);
    auto table = CalcCostTable(16, 64, 8, 32);
    EXPECT_EQ(table[NORMAL_BLOCK][NORMAL_BLOCK], CalcCost(16, 64));
    EXPECT_EQ(table[TAIL_BLOCK][NORMAL_BLOCK], CalcCost(8, 64));
    EXPECT_EQ(table[NORMAL_BLOCK][TAIL_BLOCK], CalcCost(16, 32));
    EXPECT_EQ(table[TAIL_BLOCK][TAIL_BLOCK], CalcCost(8, 32));

    auto zeros = CalcCostTable(16, 64, 0, 0);
    EXPECT_EQ(zeros[TAIL_BLOCK][NORMAL_BLOCK], 0);
    EXPECT_EQ(zeros[NORMAL_BLOCK][TAIL_BLOCK], 0);
    EXPECT_EQ(zeros[TAIL_BLOCK][TAIL_BLOCK], 0);
}

void TestCalcS2RangeBranches()
{
    BaseInfo info = MakeBase(1, 1, 4, 16, 128);
    SplitParam param = MakeParam(64, 32);
    BatchCache cache {};
    cache.batchSlot = 0;
    cache.qTokLen = 16;
    cache.kvTokLen = 128;
    cache.preTokenLeftUp = 65536;
    cache.nextTokenLeftUp = 65536;

    // empty seq
    cache.qTokLen = 0;
    auto empty = CalcS2Range(0, info, param, cache);
    EXPECT_EQ(empty.first, 0U);
    EXPECT_EQ(empty.second, 0U);
    cache.qTokLen = 16;

    // no mask → full S2 tiles
    info.attenMaskFlag = false;
    auto full = CalcS2Range(0, info, param, cache);
    EXPECT_EQ(full.first, 0U);
    EXPECT_EQ(full.second, (128U + 32U - 1U) / 32U);

    // causal-ish mask with isS1G
    info.attenMaskFlag = true;
    info.sparseMode = static_cast<int32_t>(SparseMode::LEFT_UP_CAUSAL);
    cache.preTokenLeftUp = 0;
    cache.nextTokenLeftUp = 0;
    auto masked = CalcS2Range(0, info, param, cache);
    EXPECT_TRUE(masked.second >= masked.first);

    // isS1G=false path (G-major) spanning one G
    info.isS1G = false;
    cache.qTokLen = 8;
    auto gMajor = CalcS2Range(0, info, param, cache);
    EXPECT_TRUE(gMajor.second >= gMajor.first);
}

void TestCacheCostAndCopyClear()
{
    BaseInfo info = MakeBase(2, 2, 2, 32, 256);
    info.actualSeqS1Size = {32, 32};
    info.actualSeqS2Size = {256, 128};
    SplitParam param = MakeParam(64, 64);
    SplitContext ctx(info, param);

    CalcSplitInfo(ctx);
    EXPECT_TRUE(!ctx.splitInfo.isKvSeqAllZero);
    EXPECT_TRUE(ctx.splitInfo.s1GBaseNum[0] > 0U);
    EXPECT_TRUE(ctx.splitInfo.s2BaseNum[0] > 0U);

    BatchCache bCache {};
    CalcBatchCache(0, ctx, bCache);
    EXPECT_EQ(bCache.batchSlot, 0U);
    EXPECT_EQ(bCache.qTokLen, 32U);
    EXPECT_EQ(bCache.kvTokLen, 256U);

    S1GCache row {};
    CalcS1GCache(0, ctx, bCache, row);
    EXPECT_EQ(row.batchSlot, 0U);
    EXPECT_TRUE(row.s1GBlock > 0U || (row.s2Start >= row.s2End));

    // empty-span S1GCache early return
    BatchCache emptyBatch = bCache;
    emptyBatch.qTokLen = 0;
    emptyBatch.kvTokLen = 0;
    S1GCache emptyRow {};
    CalcS1GCache(0, ctx, emptyBatch, emptyRow);
    EXPECT_EQ(emptyRow.s1GBlock, 0U);
    EXPECT_EQ(emptyRow.s1GCost, 0);

    CostInfo cost(info.bSize);
    CalcBatchCost(0, ctx, cost);
    EXPECT_TRUE(cost.bN2CostOfEachBatch[0] >= 0);
    CalcCostInfo(ctx);
    EXPECT_TRUE(ctx.costBag.totalBlockNum > 0U);
    EXPECT_TRUE(ctx.costBag.totalCost > 0);

    // all-zero KV short-circuit
    BaseInfo zeroInfo = MakeBase(1, 1, 1, 0, 0);
    zeroInfo.actualSeqS1Size = {0};
    zeroInfo.actualSeqS2Size = {0};
    SplitContext zeroCtx(zeroInfo, param);
    CalcSplitInfo(zeroCtx);
    EXPECT_TRUE(zeroCtx.splitInfo.isKvSeqAllZero);
    CalcCostInfo(zeroCtx);
    EXPECT_EQ(zeroCtx.costBag.totalCost, 0);
    EXPECT_EQ(zeroCtx.costBag.totalBlockNum, 0U);

    // CopyTmpResult / ClearTmpResult
    SplitResult src {4, 2};
    src.usedCoreNum = 2;
    src.maxCost = 99;
    src.numOfFdHead = 1;
    src.maxS2SplitNum = 3;
    src.bN2End[0] = 1;
    src.gS1End[0] = 2;
    src.s2End[0] = 3;
    src.fdRes.bN2IdxOfFdHead[0] = 4;
    src.fdRes.gS1IdxOfFdHead[0] = 5;
    src.fdRes.s2SplitNumOfFdHead[0] = 6;
    src.fdRes.s2SplitStartIdxOfCore[0] = 7;
    src.fdRes.gS1SplitNumOfFdHead[0] = 8;
    src.fdRes.gS1LastPartSizeOfFdHead[0] = 9;

    SplitResult dst {4, 2};
    CopyTmpResult(src, dst);
    EXPECT_EQ(dst.usedCoreNum, 2U);
    EXPECT_EQ(dst.maxCost, 99);
    EXPECT_EQ(dst.bN2End[0], 1U);
    EXPECT_EQ(dst.s2End[0], 3U);
    EXPECT_EQ(dst.fdRes.gS1LastPartSizeOfFdHead[0], 9U);

    ClearTmpResult(dst);
    EXPECT_EQ(dst.usedCoreNum, 0U);
    EXPECT_EQ(dst.maxCost, 0);
    EXPECT_EQ(dst.numOfFdHead, 0U);
    EXPECT_EQ(dst.bN2End[0], 0U);
    EXPECT_EQ(dst.s2End[0], 0U);
}

void TestFdHelpersAndAssignPath()
{
    // IsNeedRecordFDInfo branches
    SplitResult res {4, 2};
    res.bN2End[0] = 1;
    res.gS1End[0] = 2;
    AssignContext ac {};
    ac.curCoreIdx = 0;
    ac.curKvSplitPart = 3;
    EXPECT_TRUE(!IsNeedRecordFDInfo(ac, res));

    ac.curCoreIdx = 1;
    ac.curKvSplitPart = 1;
    EXPECT_TRUE(!IsNeedRecordFDInfo(ac, res));

    ac.curKvSplitPart = 3;
    ac.curBN2Idx = 1;
    ac.curS1GIdx = 2;
    EXPECT_TRUE(!IsNeedRecordFDInfo(ac, res)); // still on prev cut row

    ac.curS1GIdx = 3;
    EXPECT_TRUE(IsNeedRecordFDInfo(ac, res));

    // RecordFDInfo + SplitFD via a plan that creates FD heads
    BaseInfo info = MakeBase(1, 1, 8, 64, 2048);
    info.actualSeqS1Size = {64};
    info.actualSeqS2Size = {2048};
    SplitParam param = MakeParam(64, 128);
    SplitResult out {16, 2};
    SplitCore(16, info, param, out);
    EXPECT_TRUE(out.usedCoreNum >= 1U);
    EXPECT_TRUE(out.usedCoreNum <= 16U);
    // FD may or may not trigger depending on balance; still call SplitFD safely when heads exist
    if (out.numOfFdHead > 0U) {
        SplitFD(out);
        EXPECT_TRUE(out.usedVecNumOfFd >= 1U);
    }
}

void TestSplitCoreMain()
{
    SplitParam param = MakeParam();

    // empty KV → usedCoreNum=1 sentinel
    BaseInfo empty = MakeBase(1, 2, 1, 0, 0);
    empty.actualSeqS1Size = {0};
    empty.actualSeqS2Size = {0};
    SplitResult emptyRes {8, 2};
    SplitCore(8, empty, param, emptyRes);
    EXPECT_EQ(emptyRes.usedCoreNum, 1U);
    EXPECT_EQ(emptyRes.bN2End[0], 1U * 2U);

    // small decode-like case
    BaseInfo small = MakeBase(1, 2, 4, 1, 128);
    small.actualSeqS1Size = {1};
    small.actualSeqS2Size = {128};
    SplitResult smallRes {8, 2};
    SplitCore(8, small, param, smallRes);
    EXPECT_TRUE(smallRes.usedCoreNum >= 1U);
    EXPECT_TRUE(smallRes.maxCost >= 0);

    // multi-batch with uneven lengths (AssignByBatch/Row/Block/ForceAssign/UpdateCursor)
    BaseInfo multi = MakeBase(3, 2, 2, 16, 512);
    multi.actualSeqS1Size = {8, 16, 4};
    multi.actualSeqS2Size = {256, 512, 64};
    SplitResult multiRes {24, 2};
    SplitCore(24, multi, param, multiRes);
    EXPECT_TRUE(multiRes.usedCoreNum >= 1U);
    EXPECT_TRUE(multiRes.usedCoreNum <= 24U);

    // causal mask path through CalcS2Range inside CalcS1GCache
    BaseInfo causal = MakeBase(1, 1, 4, 32, 256);
    causal.actualSeqS1Size = {32};
    causal.actualSeqS2Size = {256};
    causal.attenMaskFlag = true;
    causal.sparseMode = static_cast<int32_t>(SparseMode::LEFT_UP_CAUSAL);
    causal.preToken = 0;
    causal.nextToken = 0;
    SplitResult causalRes {12, 2};
    SplitCore(12, causal, param, causalRes);
    EXPECT_TRUE(causalRes.usedCoreNum >= 1U);

    // BAND sparse mode token helpers + split
    BaseInfo band = causal;
    band.sparseMode = static_cast<int32_t>(SparseMode::BAND);
    band.preToken = 16;
    band.nextToken = 16;
    SplitResult bandRes {12, 2};
    SplitCore(12, band, param, bandRes);
    EXPECT_TRUE(bandRes.usedCoreNum >= 1U);
}

} // namespace

int main()
{
    TestGetS1S2SeqSize();
    TestTokenLeftUpAllSparseModes();
    TestCalcCostAndTable();
    TestCalcS2RangeBranches();
    TestCacheCostAndCopyClear();
    TestFdHelpersAndAssignPath();
    TestSplitCoreMain();

    if (gFails != 0) {
        std::fprintf(stderr, "test_split_core: %d failure(s)\n", gFails);
        return 1;
    }
    std::printf("test_split_core: all checks passed\n");
    return 0;
}
