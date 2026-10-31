/*!
 * \file test_kv_quant_infershape.cpp
 * \brief Host-only regression for rewritten InferShape helpers in
 *        rq_uniform_kv_quant_sparse_flash_attention_infershape.cpp.
 *
 * Re-implements the pure shape logic (same control flow as production) so we can
 * cover ParseQueryLayout / CopyLeadingDims / ShrinkLastDimByRope / TND+BSND
 * InferShape branches without linking the full GE runtime.
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

// ---- Mirror of production anonymous helpers (keep in sync with infershape.cpp) ----

enum class QueryLayoutKind : uint8_t { kBsnd = 0, kTnd = 1, kUnknown = 255 };

QueryLayoutKind ParseQueryLayout(const char *layoutAttr)
{
    if (layoutAttr == nullptr) {
        return QueryLayoutKind::kUnknown;
    }
    if (std::strcmp(layoutAttr, "BSND") == 0) {
        return QueryLayoutKind::kBsnd;
    }
    if (std::strcmp(layoutAttr, "TND") == 0) {
        return QueryLayoutKind::kTnd;
    }
    return QueryLayoutKind::kUnknown;
}

struct FakeShape {
    std::vector<int64_t> dims;
    size_t GetDimNum() const { return dims.size(); }
    void SetDimNum(size_t n)
    {
        dims.resize(n, 0);
    }
    int64_t GetDim(size_t i) const { return dims[i]; }
    void SetDim(size_t i, int64_t v) { dims[i] = v; }
};

void CopyLeadingDims(FakeShape &dst, const FakeShape &src, size_t rank, size_t lastKeep)
{
    dst.SetDimNum(rank);
    for (size_t axis = 0; axis < lastKeep; ++axis) {
        dst.SetDim(axis, src.GetDim(axis));
    }
}

void ShrinkLastDimByRope(FakeShape &dst, const FakeShape &src, size_t lastAxis, int64_t ropeDim)
{
    const int64_t raw = src.GetDim(lastAxis);
    if (raw == -1) {
        return;
    }
    dst.SetDim(lastAxis, raw - ropeDim);
}

FakeShape InferAttentionOut(const FakeShape &query, const char *layout, int64_t ropeHeadDim)
{
    FakeShape out = query;
    const QueryLayoutKind kind = ParseQueryLayout(layout);
    switch (kind) {
        case QueryLayoutKind::kBsnd: {
            constexpr size_t kRank = 4;
            CopyLeadingDims(out, query, kRank, /*lastKeep=*/3);
            ShrinkLastDimByRope(out, query, /*lastAxis=*/3, ropeHeadDim);
            break;
        }
        case QueryLayoutKind::kTnd:
        default: {
            constexpr size_t kRank = 3;
            CopyLeadingDims(out, query, kRank, /*lastKeep=*/2);
            ShrinkLastDimByRope(out, query, /*lastAxis=*/2, ropeHeadDim);
            break;
        }
    }
    return out;
}

void TestParseQueryLayout()
{
    EXPECT_TRUE(ParseQueryLayout(nullptr) == QueryLayoutKind::kUnknown);
    EXPECT_TRUE(ParseQueryLayout("BSND") == QueryLayoutKind::kBsnd);
    EXPECT_TRUE(ParseQueryLayout("TND") == QueryLayoutKind::kTnd);
    EXPECT_TRUE(ParseQueryLayout("PA_BSND") == QueryLayoutKind::kUnknown);
    EXPECT_TRUE(ParseQueryLayout("") == QueryLayoutKind::kUnknown);
}

void TestTndInferShape()
{
    FakeShape q;
    q.dims = {4, 8, 576}; // T, N, D
    auto out = InferAttentionOut(q, "TND", /*rope=*/64);
    EXPECT_EQ(out.GetDimNum(), 3U);
    EXPECT_EQ(out.GetDim(0), 4);
    EXPECT_EQ(out.GetDim(1), 8);
    EXPECT_EQ(out.GetDim(2), 512);

    // dynamic last dim stays unset when src is -1
    FakeShape dyn;
    dyn.dims = {2, 8, -1};
    auto dynOut = InferAttentionOut(dyn, "TND", 64);
    EXPECT_EQ(dynOut.GetDim(2), -1); // ShrinkLastDimByRope early-return leaves copied -1
}

void TestBsndInferShape()
{
    FakeShape q;
    q.dims = {1, 2, 8, 576}; // B, S, N, D
    auto out = InferAttentionOut(q, "BSND", /*rope=*/64);
    EXPECT_EQ(out.GetDimNum(), 4U);
    EXPECT_EQ(out.GetDim(0), 1);
    EXPECT_EQ(out.GetDim(1), 2);
    EXPECT_EQ(out.GetDim(2), 8);
    EXPECT_EQ(out.GetDim(3), 512);
}

void TestUnknownFallsBackToTnd()
{
    FakeShape q;
    q.dims = {3, 4, 576};
    auto out = InferAttentionOut(q, "WEIRD", 64);
    EXPECT_EQ(out.GetDimNum(), 3U);
    EXPECT_EQ(out.GetDim(2), 512);
}

void TestCopyAndShrinkEdges()
{
    FakeShape src;
    src.dims = {9, 7, 5, 576};
    FakeShape dst;
    CopyLeadingDims(dst, src, 4, 3);
    EXPECT_EQ(dst.GetDimNum(), 4U);
    EXPECT_EQ(dst.GetDim(0), 9);
    EXPECT_EQ(dst.GetDim(1), 7);
    EXPECT_EQ(dst.GetDim(2), 5);
    // last dim not copied by lastKeep=3 — remains 0 from resize
    EXPECT_EQ(dst.GetDim(3), 0);

    ShrinkLastDimByRope(dst, src, 3, 64);
    EXPECT_EQ(dst.GetDim(3), 512);

    FakeShape dynDst = dst;
    FakeShape dynSrc;
    dynSrc.dims = {1, 1, 1, -1};
    ShrinkLastDimByRope(dynDst, dynSrc, 3, 64);
    EXPECT_EQ(dynDst.GetDim(3), 512); // unchanged when raw==-1
}

} // namespace

int main()
{
    TestParseQueryLayout();
    TestTndInferShape();
    TestBsndInferShape();
    TestUnknownFallsBackToTnd();
    TestCopyAndShrinkEdges();

    if (gFails != 0) {
        std::fprintf(stderr, "test_kv_quant_infershape: %d failure(s)\n", gFails);
        return 1;
    }
    std::printf("test_kv_quant_infershape: all checks passed\n");
    return 0;
}
