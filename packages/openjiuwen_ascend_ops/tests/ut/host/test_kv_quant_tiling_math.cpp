/*!
 * \file test_kv_quant_tiling_math.cpp
 * \brief Host-only regression for rewritten kv_quant MLA tiling helpers.
 *
 * Mirrors pure math / gate logic from
 * rq_uniform_kv_quant_sparse_flash_attention_tiling.cpp so PlanState rewrite
 * and CalcBalanceFDParamNums cannot drift without a failing UT.
 * Build via tests/ut/test_split_core_host.py.
 */
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

// ---- Constants locked to production tiling.cpp / tiling.h ----
constexpr uint32_t kByteBlock = 32;
constexpr uint32_t kMaxBlockSize = 1024;
constexpr uint32_t kBlockTableElemBytes = 4; // sizeof(int32_t) for block table elems
constexpr int64_t kSfaSlotBytes = 388;
constexpr int64_t kQHeadDimFixed = 576;
constexpr int64_t kRopeHeadDimFixed = 64;
constexpr int64_t kTileSizeFixed = 128;
constexpr int32_t kAttentionModeMla = 2;
constexpr int32_t kKeyQuantSchemaA = 4;
constexpr int32_t kQuantScaleCombine = 1;
constexpr uint32_t kDefaultMBase = 128;
constexpr uint32_t kDefaultS2Inner = 512;
constexpr uint32_t kFdSplitThreshold = 256;

template <typename T>
T RoundUpTo(T num, T rnd)
{
    if (rnd == 0) {
        return num;
    }
    return ((num + rnd - 1) / rnd) * rnd;
}

bool IsPowerOfTwoPositive(int64_t v)
{
    if (v <= 0) {
        return false;
    }
    const uint64_t u = static_cast<uint64_t>(v);
    return (u & (u - 1ULL)) == 0ULL;
}

// Mirror: QSFAMlaTiling::CalcBalanceFDParamNums
uint32_t CalcBalanceFDParamNums(uint32_t liveCoreCnt, uint32_t n2Size, uint32_t mBaseSize)
{
    return liveCoreCnt * 2U * n2Size * mBaseSize;
}

struct PlanState {
    uint32_t sInnerSize = 0;
    uint32_t sInnerLoopTimes = 0;
    uint32_t sInnerSizeTail = 0;
    uint32_t sInnerSizeAlign = 0;
    uint32_t usedCoreNum = 0;
    uint32_t kvSplitPart = 1;
    size_t mmResUbSize = 0;
    size_t bmm2ResUbSize = 0;
    size_t qPreSizeMla = 0;
    uint32_t headDimAlign = 0;
    bool splitKVFlag = false;
    uint32_t mBaseSize = kDefaultMBase;
    uint32_t aicNum = 0;
    uint32_t vectorizeFlag = 0;
};

enum class Layout : uint32_t { BSND = 0, TND = 1, PA_BSND = 2 };

struct TilingInfoLite {
    int64_t gSize = 1;
    int64_t s1Size = 1;
    uint32_t s2Size = 0;
    int64_t qHeadDim = kQHeadDimFixed;
    Layout qLayout = Layout::TND;
    Layout kvLayout = Layout::PA_BSND;
    int32_t blockSize = 128;
    int64_t maxBlockNumPerBatch = 1;
    int64_t sparseBlockCount = 128;
    uint32_t sparseMode = 3;
    bool isA5 = false;
    uint32_t n2Size = 1;
};

void PlanCalcInnerSize(PlanState &plan, const TilingInfoLite &info, uint32_t kvSpan)
{
    plan.sInnerSize = kDefaultS2Inner;
    const bool mayShrinkForFd = plan.splitKVFlag && (info.qLayout != Layout::TND);
    if (mayShrinkForFd) {
        if (kvSpan == kFdSplitThreshold) {
            plan.sInnerSize = kFdSplitThreshold / 2U;
        } else if (kvSpan > kFdSplitThreshold && kvSpan <= plan.sInnerSize) {
            plan.sInnerSize = (plan.sInnerSize + 1U) / 2U;
        }
    }
    plan.sInnerLoopTimes = (kvSpan + plan.sInnerSize - 1U) / plan.sInnerSize;
    plan.sInnerSizeTail = kvSpan - (plan.sInnerLoopTimes - 1U) * plan.sInnerSize;
    if (plan.sInnerSize > kvSpan) {
        plan.sInnerSize = kvSpan;
    }
    plan.sInnerSizeAlign = RoundUpTo(plan.sInnerSize, kByteBlock);

    uint32_t cubeM = static_cast<uint32_t>(info.gSize * info.s1Size);
    if (cubeM > plan.mBaseSize) {
        cubeM = plan.mBaseSize;
    }
    plan.mmResUbSize = plan.sInnerSizeAlign * RoundUpTo(cubeM, 16U);
    plan.bmm2ResUbSize = plan.headDimAlign * RoundUpTo(cubeM, 16U);
    plan.qPreSizeMla = static_cast<size_t>(info.gSize) * (plan.headDimAlign + 64U) * static_cast<size_t>(info.s1Size);
}

void PlanSplitBalanced(PlanState &plan, const TilingInfoLite &info)
{
    PlanCalcInnerSize(plan, info, info.s2Size);
    plan.usedCoreNum = plan.aicNum;
}

// Mirror NormalCalcFDWorkSpace delta (bytes added when splitKVFlag)
size_t FdWorkspaceDelta(uint32_t liveCoreCnt, uint32_t n2, uint32_t mBase, uint32_t headDimAlign,
                        uint32_t blockTypeSize, bool isDav2002)
{
    const uint32_t fdNums = CalcBalanceFDParamNums(liveCoreCnt, n2, mBase);
    const uint32_t accumOutSize = fdNums * headDimAlign;
    const uint32_t logSumExpSize = 2 * fdNums * (kByteBlock / blockTypeSize);
    size_t bytes = static_cast<size_t>(accumOutSize + logSumExpSize) * blockTypeSize;
    if (isDav2002) {
        bytes += static_cast<size_t>(liveCoreCnt) * 32U;
    }
    return bytes;
}

uint32_t SplitKvAccumOutSize(uint32_t aicNum, uint32_t n2, uint32_t mBase, uint32_t headDimAlign)
{
    return aicNum * 2U * n2 * mBase * headDimAlign;
}

uint32_t SplitKvLogSumExpSize(uint32_t aicNum, uint32_t n2, uint32_t mBase)
{
    return 2U * aicNum * 2U * n2 * mBase * (kByteBlock / kBlockTableElemBytes);
}

struct AttrI32Eq {
    const char *name;
    int32_t seen;
    int32_t want;
};

struct AttrI64Eq {
    const char *name;
    int64_t seen;
    int64_t want;
};

bool RunAttrI32EqGates(const AttrI32Eq *gates, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (gates[i].seen != gates[i].want) {
            return false;
        }
    }
    return true;
}

bool RunAttrI64EqGates(const AttrI64Eq *gates, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (gates[i].seen != gates[i].want) {
            return false;
        }
    }
    return true;
}

bool MembershipI64(int64_t seen, const uint32_t *allowed, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        if (static_cast<int64_t>(allowed[i]) == seen) {
            return true;
        }
    }
    return false;
}

// Mirror CheckFeatureMlaAntiquantPa (PAGE_ATTENTION path)
bool PaBlockSizeOk(int32_t blockSize, int64_t sparseBlockSize)
{
    if ((blockSize <= 0) || (blockSize > static_cast<int32_t>(kMaxBlockSize))) {
        return false;
    }
    if (blockSize % 16 > 0) {
        return false;
    }
    if (sparseBlockSize <= 0 || (blockSize % sparseBlockSize) != 0) {
        return false;
    }
    return true;
}

// Mirror CheckFeatureMlaAntiquantShapeSizes essentials
bool ShapeSizesOk(int64_t b, int64_t qT, bool isTnd, int64_t n1, int64_t n2, int64_t gSize, bool isA5)
{
    if (b <= 0) {
        return false;
    }
    if (qT <= 0 && isTnd) {
        return false;
    }
    if (n1 <= 0) {
        return false;
    }
    if (n2 != 1) {
        return false;
    }
    if (n1 % n2 != 0) {
        return false;
    }
    static const uint32_t kGSizeA5[] = {128, 48, 64, 32, 16, 8, 4, 2, 1};
    static const uint32_t kGSizeA2[] = {128, 64, 32, 16, 8, 4, 2, 1};
    const uint32_t *gList = isA5 ? kGSizeA5 : kGSizeA2;
    const size_t gLen = isA5 ? sizeof(kGSizeA5) / sizeof(kGSizeA5[0]) : sizeof(kGSizeA2) / sizeof(kGSizeA2[0]);
    return MembershipI64(gSize, gList, gLen);
}

bool HeadDimsOk(int64_t qHead, int64_t kHead, int64_t vHead)
{
    if (qHead != kQHeadDimFixed) {
        return false;
    }
    if (kHead == 386 || vHead == 386) {
        return false;
    }
    return kHead == kSfaSlotBytes && vHead == kSfaSlotBytes;
}

uint32_t PlanCalcVectorize(const TilingInfoLite &info)
{
    if (!info.isA5 || info.kvLayout != Layout::PA_BSND) {
        return 0U;
    }
    constexpr uint32_t S2_NUM_PER_LOOP = 128U;
    const uint64_t blockSize = static_cast<uint64_t>(info.blockSize);
    const uint32_t blocksizeFlag = static_cast<uint32_t>((blockSize & (blockSize - 1)) == 0);
    const uint32_t sparseBlockCountFlag =
        static_cast<uint32_t>((static_cast<uint32_t>(info.sparseBlockCount) & (S2_NUM_PER_LOOP - 1U)) == 0U);
    const uint32_t sparseModeFlag = static_cast<uint32_t>(info.sparseMode == 3U);
    constexpr uint32_t UB_SIZE = 248 * 1024;
    const uint32_t alignedSparseBlockCount =
        (static_cast<uint32_t>(info.sparseBlockCount) + S2_NUM_PER_LOOP - 1U) / S2_NUM_PER_LOOP * S2_NUM_PER_LOOP;
    const uint32_t vectorizeUbSize =
        RoundUpTo<uint32_t>(static_cast<uint32_t>(info.maxBlockNumPerBatch * sizeof(int32_t)), 512U) +
        RoundUpTo<uint32_t>(static_cast<uint32_t>(info.sparseBlockCount) * sizeof(int32_t), kByteBlock) +
        alignedSparseBlockCount * sizeof(int64_t);
    return static_cast<uint32_t>(vectorizeUbSize <= UB_SIZE && blocksizeFlag != 0 && sparseBlockCountFlag != 0 &&
                                 sparseModeFlag != 0);
}

void TestCalcBalanceFDParamNums()
{
    EXPECT_EQ(CalcBalanceFDParamNums(1, 1, 128), 256U);
    EXPECT_EQ(CalcBalanceFDParamNums(8, 1, 128), 2048U);
    EXPECT_EQ(CalcBalanceFDParamNums(4, 2, 64), 1024U);
    EXPECT_EQ(CalcBalanceFDParamNums(0, 1, 128), 0U);
}

void TestPlanInnerSizeDefaultNoFd()
{
    PlanState plan;
    plan.mBaseSize = 128;
    plan.aicNum = 24;
    plan.headDimAlign = RoundUpTo(static_cast<uint32_t>(kQHeadDimFixed), kByteBlock);
    plan.splitKVFlag = false;

    TilingInfoLite info;
    info.gSize = 8;
    info.s1Size = 1;
    info.s2Size = 128; // < default inner → clamp sInner to kvSpan
    info.qLayout = Layout::TND;

    PlanSplitBalanced(plan, info);
    EXPECT_EQ(plan.sInnerSize, 128U);
    EXPECT_EQ(plan.sInnerLoopTimes, 1U);
    EXPECT_EQ(plan.sInnerSizeTail, 128U);
    EXPECT_EQ(plan.sInnerSizeAlign, 128U);
    EXPECT_EQ(plan.usedCoreNum, 24U);
    // cubeM = min(8*1, 128) = 8 → RoundUpTo(8,16)=16
    EXPECT_EQ(plan.mmResUbSize, 128U * 16U);
    EXPECT_EQ(plan.bmm2ResUbSize, plan.headDimAlign * 16U);
    EXPECT_EQ(plan.qPreSizeMla, 8U * (plan.headDimAlign + 64U) * 1U);
}

void TestPlanInnerSizeFdShrinkBsnd()
{
    PlanState plan;
    plan.mBaseSize = 128;
    plan.aicNum = 8;
    plan.headDimAlign = 576; // already aligned
    plan.splitKVFlag = true;

    TilingInfoLite info;
    info.gSize = 128;
    info.s1Size = 2; // cubeM capped to mBase
    info.s2Size = 256;
    info.qLayout = Layout::BSND; // FD shrink allowed

    PlanCalcInnerSize(plan, info, info.s2Size);
    EXPECT_EQ(plan.sInnerSize, 128U); // 256 == threshold → half
    EXPECT_EQ(plan.sInnerLoopTimes, 2U);
    EXPECT_EQ(plan.sInnerSizeTail, 128U);
    EXPECT_EQ(plan.sInnerSizeAlign, 128U);
    // cubeM = min(256, 128) = 128 → RoundUpTo 128
    EXPECT_EQ(plan.mmResUbSize, 128U * 128U);
    EXPECT_EQ(plan.bmm2ResUbSize, 576U * 128U);
}

void TestPlanInnerSizeTndIgnoresFdShrink()
{
    PlanState plan;
    plan.mBaseSize = 128;
    plan.headDimAlign = 576;
    plan.splitKVFlag = true;

    TilingInfoLite info;
    info.gSize = 8;
    info.s1Size = 1;
    info.s2Size = 256;
    info.qLayout = Layout::TND;

    PlanCalcInnerSize(plan, info, info.s2Size);
    // TND → no FD shrink; sInner stays 512 then clamps to 256
    EXPECT_EQ(plan.sInnerSize, 256U);
    EXPECT_EQ(plan.sInnerLoopTimes, 1U);
}

void TestFdWorkspaceAndSplitKvParams()
{
    const uint32_t live = 8;
    const uint32_t n2 = 1;
    const uint32_t mBase = 128;
    const uint32_t headAlign = 576;
    const uint32_t bts = 2; // fp16

    EXPECT_EQ(CalcBalanceFDParamNums(live, n2, mBase), 2048U);
    // accum=2048*576, lse=2*2048*(32/2)=65536; bytes=(accum+lse)*2
    const size_t expected =
        (static_cast<size_t>(2048U * 576U) + static_cast<size_t>(2U * 2048U * (32U / 2U))) * 2U;
    EXPECT_EQ(FdWorkspaceDelta(live, n2, mBase, headAlign, bts, false), expected);
    EXPECT_EQ(FdWorkspaceDelta(live, n2, mBase, headAlign, bts, true), expected + 8U * 32U);

    EXPECT_EQ(SplitKvAccumOutSize(24, 1, 128, 576), 24U * 2U * 1U * 128U * 576U);
    EXPECT_EQ(SplitKvLogSumExpSize(24, 1, 128), 2U * 24U * 2U * 1U * 128U * (32U / 4U));
}

void TestAttrGatesAndPaChecks()
{
    const AttrI32Eq okI32[] = {
        {"attention_mode", 2, kAttentionModeMla},
        {"key_quant_mode", 4, kKeyQuantSchemaA},
        {"value_quant_mode", 4, kKeyQuantSchemaA},
        {"quant_scale_repo_mode", 1, kQuantScaleCombine},
    };
    EXPECT_TRUE(RunAttrI32EqGates(okI32, 4));
    const AttrI32Eq badI32[] = {{"attention_mode", 1, kAttentionModeMla}};
    EXPECT_TRUE(!RunAttrI32EqGates(badI32, 1));

    const AttrI64Eq okI64[] = {
        {"tile_size", 128, kTileSizeFixed},
        {"rope_head_dim", 64, kRopeHeadDimFixed},
    };
    EXPECT_TRUE(RunAttrI64EqGates(okI64, 2));

    static const uint32_t kSparseBlk[] = {16, 8, 4, 2, 1};
    EXPECT_TRUE(MembershipI64(8, kSparseBlk, 5));
    EXPECT_TRUE(!MembershipI64(3, kSparseBlk, 5));

    EXPECT_TRUE(PaBlockSizeOk(128, 16));
    EXPECT_TRUE(!PaBlockSizeOk(0, 16));
    EXPECT_TRUE(!PaBlockSizeOk(2048, 16));
    EXPECT_TRUE(!PaBlockSizeOk(120, 16)); // not 16-aligned
    EXPECT_TRUE(!PaBlockSizeOk(128, 24)); // not divisible
}

void TestShapeAndHeadRules()
{
    EXPECT_TRUE(ShapeSizesOk(/*b=*/1, /*qT=*/4, /*isTnd=*/true, /*n1=*/8, /*n2=*/1, /*g=*/8, /*a5=*/false));
    EXPECT_TRUE(!ShapeSizesOk(0, 4, true, 8, 1, 8, false));
    EXPECT_TRUE(!ShapeSizesOk(1, 0, true, 8, 1, 8, false));
    EXPECT_TRUE(!ShapeSizesOk(1, 4, true, 8, 2, 8, false)); // n2 must be 1
    EXPECT_TRUE(!ShapeSizesOk(1, 4, true, 8, 1, 48, false)); // 48 only on A5
    EXPECT_TRUE(ShapeSizesOk(1, 4, true, 8, 1, 48, true));

    EXPECT_TRUE(HeadDimsOk(576, 388, 388));
    EXPECT_TRUE(!HeadDimsOk(512, 388, 388));
    EXPECT_TRUE(!HeadDimsOk(576, 386, 388));
    EXPECT_TRUE(!HeadDimsOk(576, 388, 384));
}

void TestPowerOfTwoAndVectorize()
{
    EXPECT_TRUE(IsPowerOfTwoPositive(128));
    EXPECT_TRUE(IsPowerOfTwoPositive(1));
    EXPECT_TRUE(!IsPowerOfTwoPositive(0));
    EXPECT_TRUE(!IsPowerOfTwoPositive(-2));
    EXPECT_TRUE(!IsPowerOfTwoPositive(12));

    TilingInfoLite a2;
    a2.isA5 = false;
    a2.kvLayout = Layout::PA_BSND;
    a2.blockSize = 128;
    a2.sparseBlockCount = 128;
    a2.sparseMode = 3;
    EXPECT_EQ(PlanCalcVectorize(a2), 0U); // A2 → always 0

    TilingInfoLite a5 = a2;
    a5.isA5 = true;
    EXPECT_EQ(PlanCalcVectorize(a5), 1U);

    a5.sparseMode = 0;
    EXPECT_EQ(PlanCalcVectorize(a5), 0U);
    a5.sparseMode = 3;
    a5.blockSize = 96; // not power-of-two
    EXPECT_EQ(PlanCalcVectorize(a5), 0U);
}

void TestRoundUp()
{
    EXPECT_EQ(RoundUpTo(576U, 32U), 576U);
    EXPECT_EQ(RoundUpTo(577U, 32U), 608U);
    EXPECT_EQ(RoundUpTo(8U, 16U), 16U);
    EXPECT_EQ(RoundUpTo(0U, 16U), 0U);
}

} // namespace

int main()
{
    TestRoundUp();
    TestCalcBalanceFDParamNums();
    TestPlanInnerSizeDefaultNoFd();
    TestPlanInnerSizeFdShrinkBsnd();
    TestPlanInnerSizeTndIgnoresFdShrink();
    TestFdWorkspaceAndSplitKvParams();
    TestAttrGatesAndPaChecks();
    TestShapeAndHeadRules();
    TestPowerOfTwoAndVectorize();

    if (gFails != 0) {
        std::fprintf(stderr, "test_kv_quant_tiling_math: %d failure(s)\n", gFails);
        return 1;
    }
    std::printf("test_kv_quant_tiling_math: all checks passed\n");
    return 0;
}
