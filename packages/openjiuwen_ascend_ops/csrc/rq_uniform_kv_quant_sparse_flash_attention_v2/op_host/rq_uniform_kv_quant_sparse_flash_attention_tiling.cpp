/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_tiling.cpp
 * \brief Host tiling + schema-driven validation for RQ-uniform KV-quant sparse FA (v2).
 */

#include <map>
#include <vector>
#include <limits>
#include <numeric>
#include <cstring>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "../op_kernel/rq_uniform_kv_quant_sparse_flash_attention_template_tiling_key.h"
#include "rq_uniform_kv_quant_sparse_flash_attention_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

// =============================================================================
// Local numeric helpers
// =============================================================================
namespace {
template <typename T>
inline T AlignCeilDiv(T value, T granule)
{
    if (granule == static_cast<T>(0)) {
        return static_cast<T>(0);
    }
    return (value + granule - static_cast<T>(1)) / granule * granule;
}
} // namespace

template <typename T>
static inline T RoundUpTo(T num, T rnd)
{
    return AlignCeilDiv(num, rnd);
}

template <typename T>
static std::string ShapeDimsBracketed(const T &shapeView)
{
    std::ostringstream oss;
    oss << "[";
    const size_t rank = shapeView.GetDimNum();
    if (rank > 0) {
        for (size_t dimIdx = 0; dimIdx + 1 < rank; ++dimIdx) {
            oss << shapeView.GetDim(dimIdx) << ", ";
        }
        oss << shapeView.GetDim(rank - 1);
    }
    oss << "]";
    return oss.str();
}

// =============================================================================
// Tunables / tensor tags (order & names diverge from OSS clone)
// =============================================================================
static const uint64_t RQ_AIV_PER_AIC = 2;
static const uint64_t RQ_RING_BYTES = 4ULL * 128ULL * 4ULL;
static const int64_t RQ_SFA_SLOT_BYTES = 388;
static const int32_t RQ_KEY_QUANT_SCHEMA_A = 4;
static const int32_t RQ_KEY_QUANT_TQ4 = 3;
static const uint32_t RQ_BLOCK_TABLE_ELEM_BYTES = 4;
static const uint32_t RQ_PRELOAD_NUM = 2;
static const int64_t RQ_INVALID_DIM = std::numeric_limits<int64_t>::min();
static const int64_t RQ_OUT_HEAD_DIM = 512;
static const int64_t RQ_Q_HEAD_DIM_FIXED = 576;
static const int64_t RQ_ROPE_HEAD_DIM_FIXED = 64;
static const int64_t RQ_TILE_SIZE_FIXED = 128;
static const int32_t RQ_ATTENTION_MODE_MLA = 2;
static const int32_t RQ_QUANT_SCALE_COMBINE = 1;

static const char *const RQ_TAG_VALUE_DEQUANT = "value_dequant_scale";
static const char *const RQ_TAG_KEY_DEQUANT = "key_dequant_scale";
static const char *const RQ_TAG_SPARSE_INDICES = "sparse_indices";
static const char *const RQ_TAG_BLOCK_TABLE = "block_table";
static const char *const RQ_TAG_ATTEN_OUT = "attnOut";
static const char *const RQ_TAG_SINKS = "sinks";
static const char *const RQ_TAG_VALUE = "value";
static const char *const RQ_TAG_KEY = "key";
static const char *const RQ_TAG_QUERY = "query";

static const std::string kNameValueDequantScale(RQ_TAG_VALUE_DEQUANT);
static const std::string kNameKeyDequantScale(RQ_TAG_KEY_DEQUANT);
static const std::string kNameSparseIndices(RQ_TAG_SPARSE_INDICES);
static const std::string kNameBlockTable(RQ_TAG_BLOCK_TABLE);
static const std::string kNameAttenOut(RQ_TAG_ATTEN_OUT);
static const std::string kNameSinks(RQ_TAG_SINKS);
static const std::string kNameValue(RQ_TAG_VALUE);
static const std::string kNameKey(RQ_TAG_KEY);
static const std::string kNameQuery(RQ_TAG_QUERY);

// Keep prior symbol aliases used by existing call sites / formulas
constexpr uint64_t kAivPerAic = RQ_AIV_PER_AIC;
constexpr uint64_t kValidSizeRingBytes = RQ_RING_BYTES;
constexpr int64_t kSfaSlotBytes = RQ_SFA_SLOT_BYTES;
constexpr int32_t kKeyQuantSchemaA = RQ_KEY_QUANT_SCHEMA_A;
constexpr int32_t kKeyQuantTq4 = RQ_KEY_QUANT_TQ4;
constexpr uint32_t kBlockTableElemBytes = RQ_BLOCK_TABLE_ELEM_BYTES;
constexpr uint32_t kPreloadNum = RQ_PRELOAD_NUM;
static const int64_t kInvalidDimValue = RQ_INVALID_DIM;

namespace {

struct DtypePair {
    const char *tag;
    const ge::DataType *wants;
    size_t wantLen;
};

struct LayoutPair {
    const char *tag;
    const QSFALayout *wants;
    size_t wantLen;
};

struct AxisPack {
    QSFALayout layout;
    const QSFAAxis *axes;
    size_t axisLen;
    size_t rank;
};

static const ge::DataType kQueryDtypes[] = {ge::DT_BF16, ge::DT_FLOAT16};
static const ge::DataType kKvDtypes[] = {ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8};
static const ge::DataType kFloatDtypes[] = {ge::DT_FLOAT};
static const ge::DataType kInt32Dtypes[] = {ge::DT_INT32};
static const ge::DataType kOutDtypes[] = {ge::DT_BF16, ge::DT_FLOAT16};

static const QSFALayout kQueryLayouts[] = {QSFALayout::TND, QSFALayout::BSND};
static const QSFALayout kKvLayouts[] = {QSFALayout::PA_BSND, QSFALayout::TND, QSFALayout::BSND};
static const QSFALayout kOutLayouts[] = {QSFALayout::TND, QSFALayout::BSND};

static const QSFAAxis kAxesBSND[] = {QSFAAxis::B, QSFAAxis::S, QSFAAxis::N, QSFAAxis::D};
static const QSFAAxis kAxesTND[] = {QSFAAxis::T, QSFAAxis::N, QSFAAxis::D};
static const QSFAAxis kAxesPA[] = {QSFAAxis::Bn, QSFAAxis::Bs, QSFAAxis::N, QSFAAxis::D};

static const AxisPack kAxisPacks[] = {
    {QSFALayout::BSND, kAxesBSND, 4, DIM_NUM_FOUR},
    {QSFALayout::TND, kAxesTND, 3, DIM_NUM_THREE},
    {QSFALayout::PA_BSND, kAxesPA, 4, DIM_NUM_FOUR},
};

static const DtypePair kDtypeTable[] = {
    {RQ_TAG_QUERY, kQueryDtypes, 2},
    {RQ_TAG_KEY, kKvDtypes, 3},
    {RQ_TAG_VALUE, kKvDtypes, 3},
    {RQ_TAG_KEY_DEQUANT, kFloatDtypes, 1},
    {RQ_TAG_VALUE_DEQUANT, kFloatDtypes, 1},
    {RQ_TAG_SPARSE_INDICES, kInt32Dtypes, 1},
    {RQ_TAG_SINKS, kFloatDtypes, 1},
    {RQ_TAG_ATTEN_OUT, kOutDtypes, 2},
};

static const LayoutPair kLayoutTable[] = {
    {RQ_TAG_QUERY, kQueryLayouts, 2},
    {RQ_TAG_KEY, kKvLayouts, 3},
    {RQ_TAG_VALUE, kKvLayouts, 3},
    {RQ_TAG_ATTEN_OUT, kOutLayouts, 2},
};

struct DtypeLabel {
    ge::DataType dtype;
    const char *name;
};

static const DtypeLabel kDtypeLabels[] = {
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_BF16, "DT_BFLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
    {ge::DT_UINT1, "DT_UINT1"},
    {ge::DT_INT4, "DT_INT4"},
    {ge::DT_UINT2, "DT_UINT2"},
    {ge::DT_INT2, "DT_INT2"},
};

static std::map<std::string, std::vector<ge::DataType>> BuildDtypeAllowMap()
{
    std::map<std::string, std::vector<ge::DataType>> allow;
    for (size_t i = 0; i < sizeof(kDtypeTable) / sizeof(kDtypeTable[0]); ++i) {
        std::vector<ge::DataType> wants(kDtypeTable[i].wants, kDtypeTable[i].wants + kDtypeTable[i].wantLen);
        allow.emplace(kDtypeTable[i].tag, std::move(wants));
    }
    return allow;
}

static std::map<std::string, std::vector<QSFALayout>> BuildLayoutAllowMap()
{
    std::map<std::string, std::vector<QSFALayout>> allow;
    for (size_t i = 0; i < sizeof(kLayoutTable) / sizeof(kLayoutTable[0]); ++i) {
        std::vector<QSFALayout> wants(kLayoutTable[i].wants, kLayoutTable[i].wants + kLayoutTable[i].wantLen);
        allow.emplace(kLayoutTable[i].tag, std::move(wants));
    }
    return allow;
}

static std::map<ge::DataType, std::string> BuildDtypeLabelMap()
{
    std::map<ge::DataType, std::string> labels;
    for (size_t i = 0; i < sizeof(kDtypeLabels) / sizeof(kDtypeLabels[0]); ++i) {
        labels.emplace(kDtypeLabels[i].dtype, kDtypeLabels[i].name);
    }
    return labels;
}

static std::map<QSFALayout, std::vector<QSFAAxis>> BuildLayoutAxisMap()
{
    std::map<QSFALayout, std::vector<QSFAAxis>> axes;
    for (size_t i = 0; i < sizeof(kAxisPacks) / sizeof(kAxisPacks[0]); ++i) {
        std::vector<QSFAAxis> pack(kAxisPacks[i].axes, kAxisPacks[i].axes + kAxisPacks[i].axisLen);
        axes.emplace(kAxisPacks[i].layout, std::move(pack));
    }
    return axes;
}

static std::map<QSFALayout, size_t> BuildLayoutRankMap()
{
    std::map<QSFALayout, size_t> ranks;
    for (size_t i = 0; i < sizeof(kAxisPacks) / sizeof(kAxisPacks[0]); ++i) {
        ranks.emplace(kAxisPacks[i].layout, kAxisPacks[i].rank);
    }
    return ranks;
}
} // namespace

static const std::map<QSFALayout, size_t> QSFA_LAYOUT_DIM_MAP = BuildLayoutRankMap();
static const std::map<QSFALayout, std::vector<QSFAAxis>> QSFA_LAYOUT_AXIS_MAP = BuildLayoutAxisMap();
const std::map<ge::DataType, std::string> kDtypeToString = BuildDtypeLabelMap();
const std::map<std::string, std::vector<QSFALayout>> kLayoutSupport = BuildLayoutAllowMap();
const std::map<std::string, std::vector<ge::DataType>> kDtypeSupport = BuildDtypeAllowMap();

// =============================================================================
// Shape / layout / dtype utilities
// =============================================================================
static std::string FormatShapeCsv(const gert::Shape &shapeView)
{
    std::ostringstream oss;
    const size_t rank = shapeView.GetDimNum();
    for (size_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        if (dimIdx != 0U) {
            oss << ", ";
        }
        oss << shapeView.GetDim(dimIdx);
    }
    return oss.str();
}

static ge::graphStatus ObtainSeqLenTensorExtent(int64_t &lenOut, const gert::Tensor *tensor, const std::string &tensorTag,
                                                const char *opName)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorTag.c_str(), tensorTag + " must be provided");
        return ge::GRAPH_FAILED;
    }
    const int64_t extent = tensor->GetShapeSize();
    if (extent <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, tensorTag.c_str(), std::to_string(extent).c_str(),
                                                  "The shape size of " + tensorTag + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    lenOut = extent;
    return ge::GRAPH_SUCCESS;
}

static size_t ScanAxisSlotIndex(const QSFAAxis axisId, const QSFALayout layoutKind)
{
    const std::vector<QSFAAxis> &axisSeq = QSFA_LAYOUT_AXIS_MAP.find(layoutKind)->second;
    for (size_t slot = 0; slot < axisSeq.size(); ++slot) {
        if (axisSeq[slot] == axisId) {
            return slot;
        }
    }
    return axisSeq.size();
}

static bool ShapeHostsLayoutAxis(const QSFAAxis axisId, const QSFALayout layoutKind, const gert::Shape &shapeView)
{
    const auto axisPackIt = QSFA_LAYOUT_AXIS_MAP.find(layoutKind);
    if (axisPackIt == QSFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }
    const auto rankIt = QSFA_LAYOUT_DIM_MAP.find(layoutKind);
    if (rankIt == QSFA_LAYOUT_DIM_MAP.end() || rankIt->second != shapeView.GetDimNum()) {
        return false;
    }
    return ScanAxisSlotIndex(axisId, layoutKind) < axisPackIt->second.size();
}

static int64_t FetchAxisExtentFromShape(const gert::Shape &shapeView, const QSFAAxis axisId,
                                        const QSFALayout layoutKind)
{
    if (!ShapeHostsLayoutAxis(axisId, layoutKind, shapeView)) {
        return kInvalidDimValue;
    }
    return shapeView.GetDim(ScanAxisSlotIndex(axisId, layoutKind));
}

static std::string QSFADataTypeToSerialString(ge::DataType type)
{
    for (std::map<ge::DataType, std::string>::const_iterator it = kDtypeToString.begin(); it != kDtypeToString.end();
         ++it) {
        if (it->first == type) {
            return it->second;
        }
    }
    OP_LOGE("SparseFlashAttention", "datatype %d not support", type);
    return "UNDEFINED";
}

std::string QSFALayoutToSerialString(QSFALayout layout)
{
    struct LayoutLabelRow {
        QSFALayout layout;
        const char *label;
    };
    static const LayoutLabelRow kRows[] = {
        {QSFALayout::BSND, "BSND"},
        {QSFALayout::TND, "TND"},
        {QSFALayout::PA_BSND, "PA_BSND"},
    };
    for (size_t row = 0; row < sizeof(kRows) / sizeof(kRows[0]); ++row) {
        if (kRows[row].layout == layout) {
            return kRows[row].label;
        }
    }
    return "UNKNOWN";
}

static uint32_t BytesOfDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16:
            return NUM_BYTES_BF16;
        case ge::DT_FLOAT16:
            return NUM_BYTES_FLOAT16;
        default:
            return NUM_BYTES_FLOAT16;
    }
}

// =============================================================================
// Schema / rule runners (file-local PODs + interpreters)
// =============================================================================
namespace {

struct RankGate {
    const char *tag;
    const size_t *allowed;
    size_t allowedLen;
    size_t seen;
};

struct AttrI64EqGate {
    const char *name;
    int64_t seen;
    int64_t want;
    const char *reason;
};

struct AttrI32EqGate {
    const char *name;
    int32_t seen;
    int32_t want;
    const char *reason;
};

struct MembershipI64Gate {
    const char *name;
    int64_t seen;
    const uint32_t *allowed;
    size_t allowedLen;
    const char *reason;
};

static ge::graphStatus FailShapeReason(const char *opName, const char *tag, const std::string &shapeCsv,
                                       const std::string &reason)
{
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, tag, shapeCsv.c_str(), reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus FailValueReason(const char *opName, const char *name, const std::string &actual,
                                       const std::string &reason)
{
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, actual.c_str(), reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus FailDtypeReason(const char *opName, const char *tag, const std::string &actual,
                                       const std::string &reason)
{
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, tag, actual.c_str(), reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus FailDtypesReason(const char *opName, const char *tag, const std::string &actual,
                                        const std::string &reason)
{
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, tag, actual.c_str(), reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus FailArgReason(const char *opName, const char *tag, const std::string &reason)
{
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tag, reason);
    return ge::GRAPH_FAILED;
}

static bool DtypeInList(ge::DataType seen, const ge::DataType *list, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        if (list[i] == seen) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus RunDtypeListGate(const char *opName, const char *tag, ge::DataType seen,
                                        const ge::DataType *wantList, size_t wantLen, const std::string &reason)
{
    if (DtypeInList(seen, wantList, wantLen)) {
        return ge::GRAPH_SUCCESS;
    }
    return FailDtypeReason(opName, tag, QSFADataTypeToSerialString(seen), reason);
}

static ge::graphStatus RunDtypeEqGate(const char *opName, const char *tag, ge::DataType seen, ge::DataType want)
{
    if (seen == want) {
        return ge::GRAPH_SUCCESS;
    }
    return FailDtypeReason(opName, tag, QSFADataTypeToSerialString(seen),
                           std::string("The dtype of ") + tag + " must be " + QSFADataTypeToSerialString(want));
}

static ge::graphStatus RunRankGates(const char *opName, const RankGate *gates, size_t gateCount)
{
    for (size_t i = 0; i < gateCount; ++i) {
        const RankGate &g = gates[i];
        bool ok = false;
        for (size_t j = 0; j < g.allowedLen; ++j) {
            if (g.allowed[j] == g.seen) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            std::ostringstream oss;
            for (size_t j = 0; j < g.allowedLen; ++j) {
                if (j > 0) {
                    oss << ((j + 1 == g.allowedLen) ? " or " : ", ");
                }
                oss << g.allowed[j];
            }
            OP_LOGE_FOR_INVALID_VALUE(opName, (std::string(g.tag) + " dimension").c_str(),
                                      std::to_string(g.seen).c_str(), oss.str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunAttrI64EqGates(const char *opName, const AttrI64EqGate *gates, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (gates[i].seen != gates[i].want) {
            return FailValueReason(opName, gates[i].name, std::to_string(gates[i].seen), gates[i].reason);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunAttrI32EqGates(const char *opName, const AttrI32EqGate *gates, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (gates[i].seen != gates[i].want) {
            return FailValueReason(opName, gates[i].name, std::to_string(gates[i].seen), gates[i].reason);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunMembershipI64Gate(const char *opName, const MembershipI64Gate &gate)
{
    for (size_t i = 0; i < gate.allowedLen; ++i) {
        if (static_cast<int64_t>(gate.allowed[i]) == gate.seen) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return FailValueReason(opName, gate.name, std::to_string(gate.seen), gate.reason);
}

static bool IsPowerOfTwoPositive(int64_t v)
{
    if (v <= 0) {
        return false;
    }
    const uint64_t u = static_cast<uint64_t>(v);
    return (u & (u - 1ULL)) == 0ULL;
}

// Plan-state working set for MLA tiling emission
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
    uint64_t workspaceSize = 0;
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
    uint32_t vectorizeFlag = 0;
    uint32_t headDimAlign = 0;
    bool splitKVFlag = false;
    QSFAPerfMode perfMode = QSFAPerfMode::V_TEMPLATE_MODE;
    uint32_t mBaseSize = 128;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    size_t libapiSize = 0;
};

} // namespace

// =============================================================================
// QSFATilingCheck — validation via CheckCtx table runners
// =============================================================================
namespace {

struct CheckCtx {
    const QSFATilingInfo &info;
    const char *opName = nullptr;
    QSFAParaInfo params{};

    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t qHeadDim = 0;
    int64_t kHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t qTSize = 0;
    int64_t kvTSize = 0;
    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    int64_t sparseBlockCount = 0;
    int64_t sparseBlockSize = 0;
    int32_t attentionMode = 0;
    int32_t keyQuantMode = 0;
    int32_t valueQuantMode = 0;
    int32_t quantScaleRepoMode = 0;
    int64_t tileSize = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int32_t ropeHeadDim = 0;

    QSFALayout qLayout = QSFALayout::BSND;
    QSFALayout topkLayout = QSFALayout::BSND;
    QSFALayout outLayout = QSFALayout::BSND;
    QSFALayout kvLayout = QSFALayout::BSND;

    uint32_t maxBlockNumPerBatch = 0;
    int64_t blockSize = 0;

    NpuArch npuArch = NpuArch::DAV_2201;
    bool isA5 = false;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    gert::Shape queryShapeCmp{};
    gert::Shape keyShapeCmp{};
    gert::Shape valueShapeCmp{};
    gert::Shape topkShapeCmp{};
    gert::Shape attenOutShapeCmp{};

    explicit CheckCtx(const QSFATilingInfo &qsfaInfo) : info(qsfaInfo) {}
};

static void LoadCheckCtx(CheckCtx &ctx, const QSFATilingInfo &info)
{
    ctx.kvStorageMode = info.kvStorageMode;
    ctx.outLayout = info.outLayout;
    ctx.kvLayout = info.kvLayout;
    ctx.topkLayout = info.topkLayout;
    ctx.qLayout = info.qLayout;

    ctx.outputType = info.outputType;
    ctx.inputKvType = info.inputKvType;
    ctx.inputQType = info.inputQType;

    ctx.nextTokens = info.nextTokens;
    ctx.preTokens = info.preTokens;
    ctx.tileSize = info.tileSize;
    ctx.quantScaleRepoMode = static_cast<int32_t>(info.quantScaleRepoMode);
    ctx.valueQuantMode = static_cast<int32_t>(info.valueQuantMode);
    ctx.keyQuantMode = static_cast<int32_t>(info.keyQuantMode);
    ctx.attentionMode = static_cast<int32_t>(info.attentionMode);

    ctx.sparseBlockSize = info.sparseBlockSize;
    ctx.sparseBlockCount = info.sparseBlockCount;
    ctx.blockSize = info.blockSize;
    ctx.kvTSize = info.kvTSize;
    ctx.qTSize = info.qTSize;
    ctx.maxBlockNumPerBatch = info.maxBlockNumPerBatch;
    ctx.ropeHeadDim = info.ropeHeadDim;
    ctx.vHeadDim = info.vHeadDim;
    ctx.kHeadDim = info.kHeadDim;
    ctx.qHeadDim = info.qHeadDim;
    ctx.gSize = info.gSize;
    ctx.s2Size = info.s2Size;
    ctx.s1Size = info.s1Size;
    ctx.n2Size = info.n2Size;
    ctx.n1Size = info.n1Size;
    ctx.bSize = info.bSize;

    ctx.isA5 = info.isA5;
    ctx.npuArch = info.npuArch;
    ctx.params = info.opParamInfo;
    ctx.opName = info.opName;
}

static void LogDtypeAllowList(const CheckCtx &ctx, const std::vector<ge::DataType> &wantDtypeList,
                              const ge::DataType &seenDtype, const std::string &tensorTag)
{
    std::ostringstream qsfaOss;
    for (size_t i = 0; i < wantDtypeList.size(); ++i) {
        qsfaOss << QSFADataTypeToSerialString(wantDtypeList[i]);
        if (i + 1 < wantDtypeList.size()) {
            qsfaOss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ctx.opName, tensorTag.c_str(), QSFADataTypeToSerialString(seenDtype).c_str(),
                                          "The dtype of " + tensorTag + " must be " + qsfaOss.str());
}

static ge::graphStatus EnforceTensorDtypePolicy(const CheckCtx &ctx, const gert::CompileTimeTensorDesc *tensorDesc,
                                                const std::string &tensorTag)
{
    if (tensorDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto policyIt = kDtypeSupport.find(tensorTag);
    if (policyIt == kDtypeSupport.end()) {
        OP_LOGE(ctx.opName, "%s datatype support list should be specify in kDtypeSupport", tensorTag.c_str());
        return ge::GRAPH_FAILED;
    }
    const auto &allowBag = policyIt->second;
    const ge::DataType seen = tensorDesc->GetDataType();
    if (std::find(allowBag.begin(), allowBag.end(), seen) == allowBag.end()) {
        LogDtypeAllowList(ctx, allowBag, seen, tensorTag);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void LogLayoutAllowList(const CheckCtx &ctx, const std::vector<QSFALayout> &wantLayoutList,
                               const QSFALayout &seenLayout, const std::string &tensorTag)
{
    std::ostringstream qsfaOssLayout;
    for (size_t i = 0; i < wantLayoutList.size(); ++i) {
        qsfaOssLayout << QSFALayoutToSerialString(wantLayoutList[i]);
        if (i + 1 < wantLayoutList.size()) {
            qsfaOssLayout << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx.opName, tensorTag.c_str(), QSFALayoutToSerialString(seenLayout).c_str(),
                                          "Tensor " + tensorTag + " only supports layout " + qsfaOssLayout.str());
}

static ge::graphStatus MatchTensorLayoutPolicy(const CheckCtx &ctx, const QSFALayout &seenLayout,
                                               const std::string &tensorTag)
{
    const auto policyIt = kLayoutSupport.find(tensorTag);
    if (policyIt == kLayoutSupport.end()) {
        OP_LOGE(ctx.opName, "%s layout support list should be specify in kLayoutSupport", tensorTag.c_str());
        return ge::GRAPH_FAILED;
    }
    const auto &allowBag = policyIt->second;
    if (std::find(allowBag.begin(), allowBag.end(), seenLayout) == allowBag.end()) {
        LogLayoutAllowList(ctx, allowBag, seenLayout, tensorTag);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceExpectedRank(const CheckCtx &ctx, const gert::StorageShape *shape,
                                           const std::vector<size_t> &wantRankList, const std::string &tensorTag)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    RankGate gate = {tensorTag.c_str(), wantRankList.data(), wantRankList.size(),
                     shape->GetStorageShape().GetDimNum()};
    return RunRankGates(ctx.opName, &gate, 1);
}

static ge::graphStatus EnforceLayoutRankPolicy(const CheckCtx &ctx, const QSFALayout &layoutKind,
                                               const gert::StorageShape *shape, const std::string &tensorTag)
{
    const auto &rankIt = QSFA_LAYOUT_DIM_MAP.find(layoutKind);
    if (shape->GetStorageShape().GetDimNum() != rankIt->second) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            ctx.opName, tensorTag.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            "When layoutKind is " + QSFALayoutToSerialString(layoutKind) + ", the shape dim of " + tensorTag +
                " should be " + std::to_string(rankIt->second));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static int64_t PickCmpExtent(const QSFATilingShapeCompareParam &shapeCmp, const QSFAAxis axisId)
{
    // QSFAAxis::K shares the underlying enum value with D (last-axis alias).
    if (axisId == QSFAAxis::B) {
        return shapeCmp.B;
    }
    if (axisId == QSFAAxis::S) {
        return shapeCmp.S;
    }
    if (axisId == QSFAAxis::N) {
        return shapeCmp.N;
    }
    if (axisId == QSFAAxis::D || axisId == QSFAAxis::K) {
        return shapeCmp.D;
    }
    if (axisId == QSFAAxis::T) {
        return shapeCmp.T;
    }
    if (axisId == QSFAAxis::Bn) {
        return shapeCmp.Bn;
    }
    if (axisId == QSFAAxis::Bs) {
        return shapeCmp.Bs;
    }
    return 0;
}

static ge::graphStatus ComposeReferenceShape(const CheckCtx &ctx, gert::Shape &shapeWant,
                                             const QSFATilingShapeCompareParam &shapeCmp, const QSFALayout &layoutKind)
{
    const auto axisPackIt = QSFA_LAYOUT_AXIS_MAP.find(layoutKind);
    if (axisPackIt == QSFA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(ctx.opName, "layoutKind %s is unsupported", QSFALayoutToSerialString(layoutKind).c_str());
        return ge::GRAPH_FAILED;
    }
    const std::vector<QSFAAxis> &axisOrder = axisPackIt->second;
    const size_t rank = axisOrder.size();
    if (rank == 3U) {
        shapeWant = gert::Shape({PickCmpExtent(shapeCmp, axisOrder[0]), PickCmpExtent(shapeCmp, axisOrder[1]),
                                 PickCmpExtent(shapeCmp, axisOrder[2])});
    } else if (rank == 4U) {
        shapeWant = gert::Shape({PickCmpExtent(shapeCmp, axisOrder[0]), PickCmpExtent(shapeCmp, axisOrder[1]),
                                 PickCmpExtent(shapeCmp, axisOrder[2]), PickCmpExtent(shapeCmp, axisOrder[3])});
    } else {
        OP_LOGE(ctx.opName, "layoutKind %s has unsupported rank %zu", QSFALayoutToSerialString(layoutKind).c_str(),
                rank);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatchTensorShapePolicy(const CheckCtx &ctx, const QSFATilingShapeCompareParam &shapeCmp,
                                              const gert::Shape &shapeView, const QSFALayout &layoutKind,
                                              const std::string &tensorTag)
{
    gert::Shape referenceShape;
    if (ComposeReferenceShape(ctx, referenceShape, shapeCmp, layoutKind) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t seenRank = shapeView.GetDimNum();
    if (seenRank != referenceShape.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            ctx.opName, tensorTag.c_str(), std::to_string(seenRank).c_str(),
            "The shapeView dim of " + tensorTag + " must be " + std::to_string(referenceShape.GetDimNum()));
        return ge::GRAPH_FAILED;
    }
    for (size_t dimIdx = 0; dimIdx < seenRank; ++dimIdx) {
        if (shapeView.GetDim(dimIdx) != referenceShape.GetDim(dimIdx)) {
            OP_LOGE_FOR_INVALID_SHAPE(ctx.opName, tensorTag.c_str(), FormatShapeCsv(shapeView).c_str(),
                                      FormatShapeCsv(referenceShape).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceDequantScaleAbsentPolicy(const CheckCtx &ctx)
{
    if (ctx.quantScaleRepoMode == 1) {
        if (ctx.params.keyDequantScale.tensor != nullptr || ctx.params.valueDequantScale.tensor != nullptr) {
            OP_LOGW(ctx.opName, "When quant_scale_repo_mode is 1(combine), dequant scales are combined with kv data, "
                                "key_dequant_scale and value_dequant_scale should not be provided");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceKvLayoutAuxPresence(const CheckCtx &ctx)
{
    struct LayoutPresenceRule {
        QSFALayout layout;
        bool needActSeqKv;
        bool needBlockTable;
        const char *actReason;
        const char *btReason;
    };
    static const LayoutPresenceRule kRules[] = {
        {QSFALayout::BSND, false, false, "", ""},
        {QSFALayout::TND, true, false, "When layout_kv is TND, actual_seq_lengths_kv must not be null", ""},
        {QSFALayout::PA_BSND, true, true, "When layout_kv is PA_BSND, actual_seq_lengths_kv must not be null",
         "When layout_kv is PA_BSND, block_table must not be null"},
    };
    for (size_t i = 0; i < sizeof(kRules) / sizeof(kRules[0]); ++i) {
        if (kRules[i].layout != ctx.kvLayout) {
            continue;
        }
        if (kRules[i].needActSeqKv && ctx.params.actualSeqLengths.tensor == nullptr) {
            return FailArgReason(ctx.opName, "actual_seq_lengths_kv", kRules[i].actReason);
        }
        if (kRules[i].needBlockTable && ctx.params.blockTable.tensor == nullptr) {
            return FailArgReason(ctx.opName, "block_table", kRules[i].btReason);
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunParaPresence(const CheckCtx &ctx)
{
    using PresenceFn = ge::graphStatus (*)(const CheckCtx &);
    static const PresenceFn kSteps[] = {EnforceDequantScaleAbsentPolicy, EnforceKvLayoutAuxPresence};
    for (size_t step = 0; step < sizeof(kSteps) / sizeof(kSteps[0]); ++step) {
        if (kSteps[step](ctx) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

struct PrimaryInputRow {
    const std::string *tag;
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    QSFALayout layout;
};

static ge::graphStatus EnforcePrimaryInputRow(const CheckCtx &ctx, const PrimaryInputRow &row)
{
    static const size_t kRanks[] = {DIM_NUM_FOUR, DIM_NUM_THREE};
    const std::vector<size_t> rankAllow(kRanks, kRanks + 2);
    if (EnforceTensorDtypePolicy(ctx, row.desc, *row.tag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (MatchTensorLayoutPolicy(ctx, row.layout, *row.tag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceExpectedRank(ctx, row.shape, rankAllow, *row.tag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return EnforceLayoutRankPolicy(ctx, row.layout, row.shape, *row.tag);
}

static ge::graphStatus EnforceQueryAndKeySlots(const CheckCtx &ctx)
{
    const PrimaryInputRow rows[] = {
        {&kNameQuery, ctx.params.query.desc, ctx.params.query.shape, ctx.qLayout},
        {&kNameKey, ctx.params.key.desc, ctx.params.key.shape, ctx.kvLayout},
    };
    for (size_t rowIdx = 0; rowIdx < sizeof(rows) / sizeof(rows[0]); ++rowIdx) {
        if (EnforcePrimaryInputRow(ctx, rows[rowIdx]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceSparseIndicesDtype(const CheckCtx &ctx)
{
    return EnforceTensorDtypePolicy(ctx, ctx.params.sparseIndices.desc, kNameSparseIndices);
}

static ge::graphStatus EnforceOptionalSinksPolicy(const CheckCtx &ctx)
{
    if (ctx.params.sinks.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (EnforceTensorDtypePolicy(ctx, ctx.params.sinks.desc, kNameSinks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &sinksShape = ctx.params.sinks.tensor->GetStorageShape();
    if (sinksShape.GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx.opName, kNameSinks.c_str(),
                                                 std::to_string(sinksShape.GetDimNum()).c_str(),
                                                 "The shape dim of sinks must be 1");
        return ge::GRAPH_FAILED;
    }
    if (sinksShape.GetShapeSize() != ctx.n1Size) {
        OP_LOGE_FOR_INVALID_SHAPE(ctx.opName, kNameSinks.c_str(), ShapeDimsBracketed(sinksShape).c_str(),
                                  ShapeDimsBracketed(gert::Shape({ctx.n1Size})).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceV2SoftmaxOutputs(const CheckCtx &ctx)
{
    if (!ctx.info.isV2Op) {
        return ge::GRAPH_SUCCESS;
    }
    if (ctx.params.returnSoftmaxLse == nullptr || !*ctx.params.returnSoftmaxLse) {
        return ge::GRAPH_SUCCESS;
    }
    struct SoftmaxPresence {
        const char *tag;
        bool missing;
        const char *reason;
    };
    const SoftmaxPresence rules[] = {
        {"softmax_max", ctx.params.softmaxMax.shape == nullptr,
         "When return_softmax_lse is true, the shape of softmax_max is nullptr"},
        {"softmax_max", ctx.params.softmaxMax.desc == nullptr,
         "When return_softmax_lse is true, the desc of softmax_max is nullptr"},
        {"softmax_sum", ctx.params.softmaxSum.shape == nullptr,
         "When return_softmax_lse is true, the shape of softmax_sum is nullptr"},
        {"softmax_sum", ctx.params.softmaxSum.desc == nullptr,
         "When return_softmax_lse is true, the desc of softmax_sum is nullptr"},
    };
    for (size_t i = 0; i < sizeof(rules) / sizeof(rules[0]); ++i) {
        if (rules[i].missing) {
            return FailArgReason(ctx.opName, rules[i].tag, rules[i].reason);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceSparseModeAttr(const CheckCtx &ctx)
{
    const int64_t mode = *ctx.params.sparseMode;
    if (mode != 3 && mode != 0) {
        return FailValueReason(ctx.opName, "sparse_mode", std::to_string(mode), "Sparse_mode must be 0 or 3");
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceSparseBlockSizeAttr(const CheckCtx &ctx)
{
    if (ctx.npuArch == NpuArch::DAV_2201) {
        const int64_t sb = *ctx.params.sparseBlockSize;
        if (sb <= 0 || sb > 16 || !IsPowerOfTwoPositive(sb)) {
            return FailValueReason(ctx.opName, "sparse_block_size", std::to_string(sb),
                                   "Sparse_block_size must be in range [1, 16] and be a power of 2");
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunPerTensorRules(const CheckCtx &ctx)
{
    // Order flipped vs prior OSS-like pipeline (attrs/softmax before primary slots).
    if (EnforceSparseBlockSizeAttr(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceV2SoftmaxOutputs(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceOptionalSinksPolicy(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceSparseModeAttr(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceSparseIndicesDtype(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return EnforceQueryAndKeySlots(ctx);
}

static void FillShapeCompareCaches(CheckCtx &ctx)
{
    ctx.attenOutShapeCmp = ctx.params.attenOut.shape->GetStorageShape();
    ctx.valueShapeCmp = ctx.params.value.shape->GetStorageShape();
    ctx.keyShapeCmp = ctx.params.key.shape->GetStorageShape();
    ctx.topkShapeCmp = ctx.params.sparseIndices.shape->GetStorageShape();
    ctx.queryShapeCmp = ctx.params.query.shape->GetStorageShape();
}

static ge::graphStatus EnforceKvValueDtypeMatch(const CheckCtx &ctx)
{
    return RunDtypeEqGate(ctx.opName, kNameValue.c_str(), ctx.params.value.desc->GetDataType(), ctx.inputKvType);
}

static ge::graphStatus EnforceSparseIndicesShapePolicy(const CheckCtx &ctx)
{
    QSFATilingShapeCompareParam cmpPack;
    cmpPack.B = ctx.bSize;
    cmpPack.N = ctx.n2Size;
    cmpPack.S = ctx.s1Size;
    cmpPack.D = ctx.sparseBlockCount;
    cmpPack.T = ctx.qTSize;
    return MatchTensorShapePolicy(ctx, cmpPack, ctx.topkShapeCmp, ctx.topkLayout, kNameSparseIndices);
}

static ge::graphStatus EnforceAttenOutPolicy(const CheckCtx &ctx)
{
    if (RunDtypeEqGate(ctx.opName, kNameAttenOut.c_str(), ctx.params.attenOut.desc->GetDataType(), ctx.inputQType) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QSFATilingShapeCompareParam cmpPack;
    cmpPack.B = ctx.bSize;
    cmpPack.N = ctx.n1Size;
    cmpPack.S = ctx.s1Size;
    cmpPack.D = RQ_OUT_HEAD_DIM;
    cmpPack.T = ctx.qTSize;
    return MatchTensorShapePolicy(ctx, cmpPack, ctx.attenOutShapeCmp, ctx.outLayout, kNameAttenOut);
}

static ge::graphStatus EnforceBlockTableBatchPolicy(const CheckCtx &ctx)
{
    if (ctx.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        if (ctx.params.blockTable.tensor != nullptr) {
            return FailArgReason(ctx.opName, kNameBlockTable.c_str(),
                                 "When the layout_kv is " + QSFALayoutToSerialString(ctx.kvLayout) +
                                     ", block_table should be null");
        }
        return ge::GRAPH_SUCCESS;
    }
    const uint32_t blockTableBatch = ctx.params.blockTable.tensor->GetStorageShape().GetDim(0);
    if (blockTableBatch != static_cast<uint32_t>(ctx.bSize)) {
        return FailShapeReason(ctx.opName, kNameBlockTable.c_str(),
                               FormatShapeCsv(ctx.params.blockTable.tensor->GetStorageShape()),
                               "The first dim of " + kNameBlockTable + " should be equal to batch size " +
                                   std::to_string(ctx.bSize));
    }
    return ge::GRAPH_SUCCESS;
}

struct OptionalSeqLenRow {
    const gert::Tensor *tensor;
    const gert::CompileTimeTensorDesc *desc;
    const char *tag;
    const char *missingDescReason;
};

static ge::graphStatus EnforceOptionalSeqLenRow(const CheckCtx &ctx, const OptionalSeqLenRow &row)
{
    if (row.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (row.desc == nullptr) {
        return FailArgReason(ctx.opName, row.tag, row.missingDescReason);
    }
    if (row.desc->GetDataType() != ge::DT_INT32) {
        return FailDtypeReason(ctx.opName, row.tag, QSFADataTypeToSerialString(row.desc->GetDataType()),
                               std::string("The dtype of ") + row.tag + " must be DT_INT32");
    }
    int64_t extent = 0;
    if (ObtainSeqLenTensorExtent(extent, row.tensor, row.tag, ctx.opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (extent != ctx.bSize) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            ctx.opName, row.tag, std::to_string(extent).c_str(),
            std::string("The shape size of ") + row.tag + " should be equal to batch size " +
                std::to_string(ctx.bSize));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceOptionalSeqLenTensors(const CheckCtx &ctx)
{
    const OptionalSeqLenRow rows[] = {
        {ctx.params.actualSeqLengthsQ.tensor, ctx.params.actualSeqLengthsQ.desc, "actual_seq_lengths_query",
         "Actual_seq_lengths_query is not empty, but the dtype of actual_seq_lengths_query is nullptr"},
        {ctx.params.actualSeqLengths.tensor, ctx.params.actualSeqLengths.desc, "actual_seq_lengths_kv",
         "Actual_seq_lengths_kv is not empty, but the dtype of actual_seq_lengths_kv is nullptr"},
    };
    for (size_t rowIdx = 0; rowIdx < sizeof(rows) / sizeof(rows[0]); ++rowIdx) {
        if (EnforceOptionalSeqLenRow(ctx, rows[rowIdx]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunCrossTensorShapeRules(CheckCtx &ctx)
{
    FillShapeCompareCaches(ctx);
    if (EnforceBlockTableBatchPolicy(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceOptionalSeqLenTensors(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceAttenOutPolicy(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceSparseIndicesShapePolicy(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return EnforceKvValueDtypeMatch(ctx);
}

static ge::graphStatus EnforceMlaPaBlockRules(const CheckCtx &ctx)
{
    if (ctx.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string keyShapeCsv = FormatShapeCsv(ctx.params.key.shape->GetStorageShape());
    struct PaPredicate {
        bool violated;
        const char *message;
    };
    const PaPredicate checks[] = {
        {(ctx.blockSize <= 0) || (ctx.blockSize > static_cast<int32_t>(MAX_BLOCK_SIZE)),
         "range"},
        {(ctx.blockSize % 16 > 0), "align16"},
        {(ctx.blockSize % ctx.sparseBlockSize > 0), "divSparse"},
    };
    for (size_t idx = 0; idx < sizeof(checks) / sizeof(checks[0]); ++idx) {
        if (!checks[idx].violated) {
            continue;
        }
        if (std::strcmp(checks[idx].message, "range") == 0) {
            return FailShapeReason(ctx.opName, "key", keyShapeCsv,
                                   "When page attention is enabled, block_size(" + std::to_string(ctx.blockSize) +
                                       ") should be in range (0, " + std::to_string(MAX_BLOCK_SIZE) + "]");
        }
        if (std::strcmp(checks[idx].message, "align16") == 0) {
            return FailShapeReason(ctx.opName, "key", keyShapeCsv,
                                   "When page attention is enabled, block_size(" + std::to_string(ctx.blockSize) +
                                       ") should be 16-aligned");
        }
        return FailShapeReason(ctx.opName, "key", keyShapeCsv,
                               "When page attention is enabled, block_size(" + std::to_string(ctx.blockSize) +
                                   ") must be divided by sparse_block_size(" + std::to_string(ctx.sparseBlockSize) +
                                   "), but now the remainder is " +
                                   std::to_string(ctx.blockSize % ctx.sparseBlockSize));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AssertMlaSchemaLimits(const CheckCtx &ctx)
{
    const AttrI32EqGate i32Gates[] = {
        {"attention_mode", ctx.attentionMode, RQ_ATTENTION_MODE_MLA, "Attention_mode should be 2(MLA-absorb)"},
        {"key_quant_mode", ctx.keyQuantMode, kKeyQuantSchemaA, "Key_quant_mode must be 4 (Schema A)"},
        {"value_quant_mode", ctx.valueQuantMode, kKeyQuantSchemaA, "Value_quant_mode must be 4 (Schema A)"},
        {"quant_scale_repo_mode", ctx.quantScaleRepoMode, RQ_QUANT_SCALE_COMBINE,
         "Quant_scale_repo_mode should be 1(combine)"},
    };
    if (RunAttrI32EqGates(ctx.opName, i32Gates, sizeof(i32Gates) / sizeof(i32Gates[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ctx.info.isA5) {
        return FailValueReason(ctx.opName, "platform", "A5",
                               "Schema A kv_quant is supported on Atlas A2/A3 (910B) only");
    }
    const AttrI64EqGate i64Gates[] = {
        {"preTokenLimit", ctx.preTokens, INT64_MAX, "Pre_tokens should be INT64_MAX"},
        {"nextTokenLimit", ctx.nextTokens, INT64_MAX, "Next_tokens should be INT64_MAX"},
        {"tile_size", ctx.tileSize, RQ_TILE_SIZE_FIXED, "Tile_size should be 128"},
        {"rope_head_dim", static_cast<int64_t>(ctx.ropeHeadDim), RQ_ROPE_HEAD_DIM_FIXED, "Rope_head_dim should be 64"},
    };
    return RunAttrI64EqGates(ctx.opName, i64Gates, sizeof(i64Gates) / sizeof(i64Gates[0]));
}

static ge::graphStatus EnforceMlaDtypePolicy(const CheckCtx &ctx)
{
    static const ge::DataType kQOk[] = {ge::DT_BF16, ge::DT_FLOAT16};
    if (RunDtypeListGate(ctx.opName, "query", ctx.inputQType, kQOk, 2,
                         "The dtype of query must be " + QSFADataTypeToSerialString(ge::DT_BF16) + " and " +
                             QSFADataTypeToSerialString(ge::DT_FLOAT16)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ctx.isA5) {
        static const ge::DataType kKvA5[] = {ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_INT8};
        if (!DtypeInList(ctx.inputKvType, kKvA5, 3)) {
            return FailDtypesReason(
                ctx.opName, "key and value", QSFADataTypeToSerialString(ctx.inputKvType),
                "The dtype of key and value must be " + QSFADataTypeToSerialString(ge::DT_FLOAT8_E4M3FN) + ", " +
                    QSFADataTypeToSerialString(ge::DT_HIFLOAT8) + " or " + QSFADataTypeToSerialString(ge::DT_INT8));
        }
    } else if (ctx.inputKvType != ge::DT_INT8) {
        return FailDtypesReason(ctx.opName, "key and value", QSFADataTypeToSerialString(ctx.inputKvType),
                                "The dtype of key and value must be " + QSFADataTypeToSerialString(ge::DT_INT8));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceMlaLayoutQueryPolicy(const CheckCtx &ctx)
{
    static const char *const kOk[] = {"TND", "BSND"};
    const std::string layoutQuery = ctx.params.layoutQuery;
    for (size_t i = 0; i < sizeof(kOk) / sizeof(kOk[0]); ++i) {
        if (layoutQuery == kOk[i]) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return FailValueReason(ctx.opName, "layout_query", layoutQuery, "Layout_query only supports BSND or TND");
}

static ge::graphStatus EnforceMlaBatchHeadRules(const CheckCtx &ctx)
{
    const std::string qCsv = FormatShapeCsv(ctx.params.query.shape->GetStorageShape());
    const std::string kCsv = FormatShapeCsv(ctx.params.key.shape->GetStorageShape());
    if (ctx.bSize <= 0) {
        return FailShapeReason(ctx.opName, "query", qCsv,
                               "Batch_size of query should be greater than 0, but got " + std::to_string(ctx.bSize));
    }
    if (ctx.qTSize <= 0 && (ctx.qLayout == QSFALayout::TND)) {
        return FailShapeReason(ctx.opName, "query", qCsv,
                               "T_size of query should be greater than 0, but got " + std::to_string(ctx.qTSize));
    }
    if (ctx.n1Size <= 0) {
        return FailShapeReason(ctx.opName, "query", qCsv,
                               "The head num of query should be greater than 0, but got " + std::to_string(ctx.n1Size));
    }
    if (ctx.n2Size != 1) {
        return FailShapeReason(ctx.opName, "key", kCsv,
                               "The head num of key should be 1, but got " + std::to_string(ctx.n2Size));
    }
    if (ctx.n1Size % ctx.n2Size != 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ctx.opName, "query and key",
            Ops::Base::ToString(ctx.params.query.shape->GetStorageShape()) + " and " +
                Ops::Base::ToString(ctx.params.key.shape->GetStorageShape()).c_str(),
            "The head num of query(" + std::to_string(ctx.n1Size) + ") must be divisible by the head num of key(" +
                std::to_string(ctx.n2Size) + ")");
        return ge::GRAPH_FAILED;
    }

    static const uint32_t kGSizeA5[] = {128, 48, 64, 32, 16, 8, 4, 2, 1};
    static const uint32_t kGSizeA2[] = {128, 64, 32, 16, 8, 4, 2, 1};
    const uint32_t *gList = ctx.isA5 ? kGSizeA5 : kGSizeA2;
    const size_t gLen = ctx.isA5 ? sizeof(kGSizeA5) / sizeof(kGSizeA5[0]) : sizeof(kGSizeA2) / sizeof(kGSizeA2[0]);
    bool gOk = false;
    for (size_t i = 0; i < gLen; ++i) {
        if (static_cast<uint32_t>(ctx.gSize) == gList[i]) {
            gOk = true;
            break;
        }
    }
    if (!gOk) {
        const char *expectMsg = ctx.isA5 ? "[1, 2, 4, 8, 16, 32, 48, 64, 128]" : "[1, 2, 4, 8, 16, 32, 64, 128]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ctx.opName, "query and key",
            Ops::Base::ToString(ctx.params.query.shape->GetStorageShape()) + " and " +
                Ops::Base::ToString(ctx.params.key.shape->GetStorageShape()),
            std::string("The value of (the head num of query ceildivided by the head num of key) "
                        "should be in ") +
                expectMsg + ", but got " + std::to_string(ctx.gSize));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceMlaSparseHeadDimRules(const CheckCtx &ctx)
{
    if (ctx.isA5) {
        if (ctx.inputKvType == ge::DT_HIFLOAT8 && ctx.sparseBlockCount != 2048) {
            return FailShapeReason(
                ctx.opName, "sparse_indices", FormatShapeCsv(ctx.params.sparseIndices.shape->GetStorageShape()),
                "When the dtypes of key and value are hifloat8, sparse block count must be 2048, but got " +
                    std::to_string(ctx.sparseBlockCount));
        }
        if (ctx.sparseBlockSize != 1) {
            return FailValueReason(ctx.opName, "sparse_block_size", std::to_string(ctx.sparseBlockSize),
                                   "Sparse_block_size must be 1");
        }
    } else {
        static const uint32_t kSparseBlk[] = {16, 8, 4, 2, 1};
        MembershipI64Gate gate = {"sparse_block_size", ctx.sparseBlockSize, kSparseBlk,
                                  sizeof(kSparseBlk) / sizeof(kSparseBlk[0]),
                                  "Group num should be in [1, 2, 4, 8, 16]"};
        if (RunMembershipI64Gate(ctx.opName, gate) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (ctx.qHeadDim != RQ_Q_HEAD_DIM_FIXED) {
        return FailShapeReason(ctx.opName, "query", FormatShapeCsv(ctx.params.query.shape->GetStorageShape()),
                               "The head num of query only support 576, but got " + std::to_string(ctx.qHeadDim));
    }
    if (ctx.kHeadDim == 386 || ctx.vHeadDim == 386) {
        return FailShapeReason(ctx.opName, "key/value", FormatShapeCsv(ctx.params.key.shape->GetStorageShape()),
                               "TQ4 386-byte slots are not supported; use Schema A 388");
    }
    if (ctx.kHeadDim != kSfaSlotBytes) {
        return FailShapeReason(ctx.opName, "key", FormatShapeCsv(ctx.params.key.shape->GetStorageShape()),
                               "Schema A key slot dimension must be 388, but got " + std::to_string(ctx.kHeadDim));
    }
    if (ctx.vHeadDim != kSfaSlotBytes) {
        return FailShapeReason(ctx.opName, "value", FormatShapeCsv(ctx.params.value.shape->GetStorageShape()),
                               "Schema A value slot dimension must be 388, but got " + std::to_string(ctx.vHeadDim));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EnforceMlaShapeBundle(const CheckCtx &ctx)
{
    if (EnforceMlaBatchHeadRules(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return EnforceMlaSparseHeadDimRules(ctx);
}

static ge::graphStatus RunMlaAntiquantConstraints(const CheckCtx &ctx)
{
    if (AssertMlaSchemaLimits(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceMlaPaBlockRules(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceMlaShapeBundle(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (EnforceMlaDtypePolicy(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return EnforceMlaLayoutQueryPolicy(ctx);
}

} // namespace

ge::graphStatus QSFATilingCheck::Process()
{
    CheckCtx ctx(qsfaInfo_);
    LoadCheckCtx(ctx, qsfaInfo_);
    if (RunParaPresence(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Stage order inverted vs OSS Check* cascade (schema before per-tensor/cross).
    if (RunMlaAntiquantConstraints(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (RunCrossTensorShapeRules(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return RunPerTensorRules(ctx);
}

// =============================================================================
// QSFAInfoParser — ParseCtx + file-local runners
// =============================================================================
namespace {

struct PresenceRule {
    const char *tag;
    bool missing;
    const char *reason;
};

static ge::graphStatus RunPresenceRules(const char *opName, const PresenceRule *rules, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (rules[i].missing) {
            return FailArgReason(opName, rules[i].tag, rules[i].reason);
        }
    }
    return ge::GRAPH_SUCCESS;
}

struct ParseCtx {
    const gert::TilingContext *tilingCtx = nullptr;
    const char *opName = nullptr;
    bool isV2Op = false;
    fe::PlatFormInfos *platformInfo = nullptr;
    QSFAParaInfo opParamInfo;

    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t qHeadDim = 0;
    int64_t kHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t dSizeKV = 0;
    int64_t qTSize = 0;
    int64_t kvTSize = 0;
    int64_t sparseBlockCount = 0;

    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    QSFALayout qLayout = QSFALayout::BSND;
    QSFALayout topkLayout = QSFALayout::BSND;
    QSFALayout outLayout = QSFALayout::BSND;
    QSFALayout kvLayout = QSFALayout::BSND;

    uint32_t maxBlockNumPerBatch = 0;
    int64_t blockSize = 0;

    NpuArch npuArch = NpuArch::DAV_2201;
    bool isA5 = false;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;

    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    int64_t maxActualseq = 0;

    uint32_t actualLenDimsQ = 0;
    uint32_t actualLenDimsKV = 0;

    gert::Shape queryShape{};
    gert::Shape keyShape{};
    gert::Shape valueShape{};
    gert::Shape sparseIndicesShape{};

    uint32_t keyStride0 = 0;
    uint32_t keyStride1 = 0;
};

static ge::graphStatus BootOpIdentity(ParseCtx &ctx)
{
    if (ctx.tilingCtx->GetNodeName() == nullptr) {
        OP_LOGE("RqUniformKvQuantSparseFlashAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ctx.opName = ctx.tilingCtx->GetNodeName();
    const ge::char_t *opKind = ctx.tilingCtx->GetNodeType();
    if (opKind != nullptr && std::strcmp(opKind, "RqUniformKvQuantSparseFlashAttentionV2") == 0) {
        ctx.isV2Op = true;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BindPlatformMeta(ParseCtx &ctx)
{
    ctx.platformInfo = ctx.tilingCtx->GetPlatformInfo();
    if (ctx.platformInfo == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(ctx.opName, "GetPlatformInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto qsfaAscendcPlat = platform_ascendc::PlatformAscendC(ctx.platformInfo);
    const uint32_t qsfaAivNum = qsfaAscendcPlat.GetCoreNumAiv();
    const uint32_t qsfaAicNum = qsfaAscendcPlat.GetCoreNumAic();
    if (qsfaAicNum == 0 || qsfaAivNum == 0) {
        OPS_REPORT_VECTOR_INNER_ERR(ctx.opName, "num of core obtained is 0.");
        return GRAPH_FAILED;
    }
    ctx.npuArch = qsfaAscendcPlat.GetCurNpuArch();
    ctx.isA5 = (ctx.npuArch == NpuArch::DAV_3510);
    if (ctx.npuArch != NpuArch::DAV_2201 && ctx.npuArch != NpuArch::DAV_3510) {
        OPS_REPORT_VECTOR_INNER_ERR(ctx.opName, "Npu Arch Version[%d] is not support.", static_cast<int32_t>(ctx.npuArch));
        return GRAPH_FAILED;
    }
    qsfaAscendcPlat.GetCoreMemSize(platform_ascendc::CoreMemType::L2, ctx.l2CacheSize);
    return ge::GRAPH_SUCCESS;
}

static void BindOptionalInputs(ParseCtx &ctx)
{
    ctx.opParamInfo.blockTable.tensor = ctx.tilingCtx->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    ctx.opParamInfo.actualSeqLengthsQ.tensor = ctx.tilingCtx->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    ctx.opParamInfo.actualSeqLengthsQ.desc = ctx.tilingCtx->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    ctx.opParamInfo.actualSeqLengths.tensor = ctx.tilingCtx->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    ctx.opParamInfo.actualSeqLengths.desc = ctx.tilingCtx->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    ctx.opParamInfo.keyDequantScale.tensor = ctx.tilingCtx->GetOptionalInputTensor(KEY_DEQUANT_SCALE_INPUT_INDEX);
    ctx.opParamInfo.keyDequantScale.desc = ctx.tilingCtx->GetOptionalInputDesc(KEY_DEQUANT_SCALE_INPUT_INDEX);
    ctx.opParamInfo.valueDequantScale.tensor = ctx.tilingCtx->GetOptionalInputTensor(VALUE_DEQUANT_SCALE_INPUT_INDEX);
    ctx.opParamInfo.valueDequantScale.desc = ctx.tilingCtx->GetOptionalInputDesc(VALUE_DEQUANT_SCALE_INPUT_INDEX);
    if (ctx.isV2Op) {
        ctx.opParamInfo.sinks.tensor = ctx.tilingCtx->GetOptionalInputTensor(SINKS_INPUT_INDEX);
        ctx.opParamInfo.sinks.desc = ctx.tilingCtx->GetOptionalInputDesc(SINKS_INPUT_INDEX);
    }
}

static ge::graphStatus BindAttrHandles(ParseCtx &ctx)
{
    auto attrs = ctx.tilingCtx->GetAttrs();
    if (attrs == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(ctx.tilingCtx->GetNodeName(), "attrs got from GE is nullptr");
        return ge::GRAPH_FAILED;
    }

    ctx.opParamInfo.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    ctx.opParamInfo.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    ctx.opParamInfo.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    ctx.opParamInfo.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    ctx.opParamInfo.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    ctx.opParamInfo.keyQuantMode = attrs->GetAttrPointer<int64_t>(KEY_QUANT_MODE_ATTR_INDEX);
    ctx.opParamInfo.valueQuantMode = attrs->GetAttrPointer<int64_t>(VALUE_QUANT_MODE_ATTR_INDEX);
    ctx.opParamInfo.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    ctx.opParamInfo.preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
    ctx.opParamInfo.nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
    ctx.opParamInfo.quantScaleRepoMode = attrs->GetAttrPointer<int64_t>(QUANT_SCALE_REPO_MODE_ATTR_INDEX);
    ctx.opParamInfo.tileSize = attrs->GetAttrPointer<int64_t>(TILE_SIZE_ATTR_INDEX);
    ctx.opParamInfo.ropeHeadDim = attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);
    if (ctx.isV2Op) {
        ctx.opParamInfo.returnSoftmaxLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
    }

    auto keyStrides = ctx.tilingCtx->GetDynamicInputStride(KEY_INPUT_INDEX, 0);
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 1) {
        ctx.keyStride0 = keyStrides->GetStride(0);
        ctx.keyStride1 = keyStrides->GetStride(1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BindHandlesFromContext(ParseCtx &ctx)
{
    ctx.opParamInfo.query.desc = ctx.tilingCtx->GetInputDesc(QUERY_INPUT_INDEX);
    ctx.opParamInfo.query.shape = ctx.tilingCtx->GetInputShape(QUERY_INPUT_INDEX);
    ctx.opParamInfo.key.desc = ctx.tilingCtx->GetInputDesc(KEY_INPUT_INDEX);
    ctx.opParamInfo.key.shape = ctx.tilingCtx->GetInputShape(KEY_INPUT_INDEX);
    ctx.opParamInfo.value.desc = ctx.tilingCtx->GetInputDesc(VALUE_INPUT_INDEX);
    ctx.opParamInfo.value.shape = ctx.tilingCtx->GetInputShape(VALUE_INPUT_INDEX);
    ctx.opParamInfo.sparseIndices.desc = ctx.tilingCtx->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    ctx.opParamInfo.sparseIndices.shape = ctx.tilingCtx->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    BindOptionalInputs(ctx);

    ctx.opParamInfo.attenOut.desc = ctx.tilingCtx->GetOutputDesc(OUTPUT_INDEX);
    ctx.opParamInfo.attenOut.shape = ctx.tilingCtx->GetOutputShape(OUTPUT_INDEX);
    if (ctx.isV2Op) {
        ctx.opParamInfo.softmaxMax.desc = ctx.tilingCtx->GetOutputDesc(SOFTMAX_MAX_INDEX);
        ctx.opParamInfo.softmaxMax.shape = ctx.tilingCtx->GetOutputShape(SOFTMAX_MAX_INDEX);
        ctx.opParamInfo.softmaxSum.desc = ctx.tilingCtx->GetOutputDesc(SOFTMAX_SUM_INDEX);
        ctx.opParamInfo.softmaxSum.shape = ctx.tilingCtx->GetOutputShape(SOFTMAX_SUM_INDEX);
    }
    return BindAttrHandles(ctx);
}

static ge::graphStatus AssertRequiredAttrHandles(const ParseCtx &ctx)
{
    const PresenceRule rules[] = {
        {"layout_query", ctx.opParamInfo.layoutQuery == nullptr, "Layout_query is nullptr"},
        {"layout_kv", ctx.opParamInfo.layoutKV == nullptr, "Layout_kv is nullptr"},
        {"sparse_block_size", ctx.opParamInfo.sparseBlockSize == nullptr, "Sparse_block_size is nullptr"},
        {"scale_value", ctx.opParamInfo.scaleValue == nullptr, "Scale_value is nullptr"},
        {"sparse_mode", ctx.opParamInfo.sparseMode == nullptr, "Sparse_mode is nullptr"},
    };
    return RunPresenceRules(ctx.opName, rules, sizeof(rules) / sizeof(rules[0]));
}

static ge::graphStatus AssertRequiredTensorDescs(const ParseCtx &ctx)
{
    const PresenceRule rules[] = {
        {"query", ctx.opParamInfo.query.desc == nullptr, "The desc of query is nullptr"},
        {"key", ctx.opParamInfo.key.desc == nullptr, "The desc of key is nullptr"},
        {"value", ctx.opParamInfo.value.desc == nullptr, "The desc of value is nullptr"},
        {"sparse_indices", ctx.opParamInfo.sparseIndices.desc == nullptr, "The desc of sparse_indices is nullptr"},
        {"attnOut", ctx.opParamInfo.attenOut.desc == nullptr, "The desc of attnOut is nullptr"},
    };
    return RunPresenceRules(ctx.opName, rules, sizeof(rules) / sizeof(rules[0]));
}

static ge::graphStatus AssertRequiredTensorShapes(const ParseCtx &ctx)
{
    const PresenceRule rules[] = {
        {"query", ctx.opParamInfo.query.shape == nullptr, "The shape of query is nullptr"},
        {"key", ctx.opParamInfo.key.shape == nullptr, "The shape of key is nullptr"},
        {"value", ctx.opParamInfo.value.shape == nullptr, "The shape of value is nullptr"},
        {"sparse_indices", ctx.opParamInfo.sparseIndices.shape == nullptr, "The shape of sparse_indices is nullptr"},
        {"attnOut", ctx.opParamInfo.attenOut.shape == nullptr, "The shape of attnOut is nullptr"},
    };
    return RunPresenceRules(ctx.opName, rules, sizeof(rules) / sizeof(rules[0]));
}

static ge::graphStatus AssertRequiredBindings(const ParseCtx &ctx)
{
    if (AssertRequiredTensorShapes(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (AssertRequiredTensorDescs(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AssertRequiredAttrHandles(ctx);
}

static ge::graphStatus CaptureIoDtypesFromDescs(ParseCtx &ctx)
{
    ctx.outputType = ctx.opParamInfo.attenOut.desc->GetDataType();
    ctx.inputKvType = ctx.opParamInfo.key.desc->GetDataType();
    ctx.inputQType = ctx.opParamInfo.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseLayoutStringsFromAttrs(ParseCtx &ctx)
{
    const std::string qStr(ctx.opParamInfo.layoutQuery);
    if (qStr == "BSND") {
        ctx.qLayout = QSFALayout::BSND;
        ctx.outLayout = QSFALayout::BSND;
        ctx.topkLayout = QSFALayout::BSND;
    } else if (qStr == "TND") {
        ctx.qLayout = QSFALayout::TND;
        ctx.outLayout = QSFALayout::TND;
        ctx.topkLayout = QSFALayout::TND;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(ctx.opName, "layout_query", qStr.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }

    const std::string kvStr(ctx.opParamInfo.layoutKV);
    if (kvStr == "PA_BSND") {
        ctx.kvLayout = QSFALayout::PA_BSND;
    } else if (kvStr == "TND") {
        ctx.kvLayout = QSFALayout::TND;
    } else if (kvStr == "BSND") {
        ctx.kvLayout = QSFALayout::BSND;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(ctx.opName, "layout_kv", kvStr.c_str(), "BSND, PA_BSND or TND");
        return ge::GRAPH_FAILED;
    }

    const bool pa = (ctx.kvLayout == QSFALayout::PA_BSND);
    if (!pa && ctx.qLayout != ctx.kvLayout) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            ctx.opName, "layout_kv and layout_query",
            QSFALayoutToSerialString(ctx.kvLayout) + " and " + QSFALayoutToSerialString(ctx.qLayout),
            "When layout_kv is not PA_BSND, layout_kv and layout_query must be same");
        return ge::GRAPH_FAILED;
    }
    const uint32_t keyRank = ctx.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (pa && keyRank != 4U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx.opName, "key", std::to_string(keyRank).c_str(),
                                                 "When layout_kv is PA_BSND, the dim num of key must be 4");
        return ge::GRAPH_FAILED;
    }
    ctx.kvStorageMode = pa ? KvStorageMode::PAGE_ATTENTION : KvStorageMode::BATCH_CONTINUOUS;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CollectShapeMetrics(ParseCtx &ctx)
{
    ctx.sparseIndicesShape = ctx.opParamInfo.sparseIndices.shape->GetStorageShape();
    ctx.valueShape = ctx.opParamInfo.value.shape->GetStorageShape();
    ctx.keyShape = ctx.opParamInfo.key.shape->GetStorageShape();
    ctx.queryShape = ctx.opParamInfo.query.shape->GetStorageShape();

    ctx.sparseBlockCount = FetchAxisExtentFromShape(ctx.sparseIndicesShape, QSFAAxis::K, ctx.qLayout);
    ctx.dSizeKV = FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::D, ctx.kvLayout);
    ctx.vHeadDim = FetchAxisExtentFromShape(ctx.valueShape, QSFAAxis::D, ctx.kvLayout);
    ctx.kHeadDim = FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::D, ctx.kvLayout);
    ctx.qHeadDim = FetchAxisExtentFromShape(ctx.queryShape, QSFAAxis::D, ctx.qLayout);
    ctx.n2Size = FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::N, ctx.kvLayout);
    ctx.n1Size = FetchAxisExtentFromShape(ctx.queryShape, QSFAAxis::N, ctx.qLayout);
    ctx.gSize = (ctx.n2Size != 0) ? (ctx.n1Size / ctx.n2Size) : 0;

    if (ctx.qLayout == QSFALayout::TND) {
        if (ObtainSeqLenTensorExtent(ctx.bSize, ctx.opParamInfo.actualSeqLengthsQ.tensor, "actual_seq_lengths_query",
                                     ctx.opName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        ctx.qTSize = FetchAxisExtentFromShape(ctx.queryShape, QSFAAxis::T, ctx.qLayout);
        ctx.s1Size = ctx.qTSize;
    } else {
        ctx.bSize = FetchAxisExtentFromShape(ctx.queryShape, QSFAAxis::B, ctx.qLayout);
        ctx.qTSize = 0;
        ctx.s1Size = FetchAxisExtentFromShape(ctx.queryShape, QSFAAxis::S, ctx.qLayout);
    }
    ctx.kvTSize =
        (ctx.kvLayout == QSFALayout::TND) ? FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::T, ctx.kvLayout) : 0;

    if (ctx.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        if (ctx.opParamInfo.blockTable.tensor == nullptr) {
            return FailArgReason(ctx.opName, "block_table",
                                 "The layout_kv is " + QSFALayoutToSerialString(ctx.kvLayout) +
                                     ", block_table must be provided");
        }
        const gert::Shape &bt = ctx.opParamInfo.blockTable.tensor->GetStorageShape();
        if (bt.GetDimNum() != DIM_NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx.opName, "block_table", std::to_string(bt.GetDimNum()),
                                                     "The shape dim of block_table must be " +
                                                         std::to_string(DIM_NUM_TWO));
            return ge::GRAPH_FAILED;
        }
        if (bt.GetDim(DIM_IDX_ONE) <= 0) {
            return FailShapeReason(ctx.opName, "block_table", FormatShapeCsv(bt),
                                   "The second dim of block_table should be greater than 0");
        }
        ctx.maxBlockNumPerBatch = bt.GetDim(DIM_IDX_ONE);
        ctx.blockSize = FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::Bs, ctx.kvLayout);
        ctx.s2Size = ctx.maxBlockNumPerBatch * ctx.blockSize;
        return ge::GRAPH_SUCCESS;
    }

    if (ctx.kvLayout == QSFALayout::BSND) {
        ctx.s2Size = FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::S, ctx.kvLayout);
    } else if (ctx.kvLayout == QSFALayout::TND) {
        ctx.s2Size = FetchAxisExtentFromShape(ctx.keyShape, QSFAAxis::T, ctx.kvLayout);
    }
    return ge::GRAPH_SUCCESS;
}

static void CaptureActualSeqMeta(ParseCtx &ctx)
{
    ctx.maxActualseq = ctx.s2Size;
    if (ctx.opParamInfo.actualSeqLengths.tensor != nullptr) {
        ctx.actualLenDimsKV = ctx.opParamInfo.actualSeqLengths.tensor->GetShapeSize();
    }
    if (ctx.opParamInfo.actualSeqLengthsQ.tensor != nullptr) {
        ctx.actualLenDimsQ = ctx.opParamInfo.actualSeqLengthsQ.tensor->GetShapeSize();
    }
}

static ge::graphStatus VerifyKeyContiguity(const ParseCtx &ctx)
{
    if (!ctx.isA5) {
        return ge::GRAPH_SUCCESS;
    }
    bool keyNonContiguous = false;
    if (ctx.keyStride0 != 0 && ctx.keyShape.GetDimNum() > 0) {
        uint64_t totalElements = 0;
        uint64_t shapeSize = 0;
        if (ctx.kvLayout == QSFALayout::PA_BSND) {
            totalElements = ctx.keyStride1 * static_cast<uint64_t>(ctx.keyShape.GetDim(DIM_IDX_ONE));
            shapeSize = static_cast<uint64_t>(ctx.keyShape.GetDim(DIM_IDX_ONE)) *
                        static_cast<uint64_t>(ctx.keyShape.GetDim(DIM_IDX_TWO)) *
                        static_cast<uint64_t>(ctx.keyShape.GetDim(DIM_IDX_THREE));
        } else {
            totalElements = ctx.keyStride0 * static_cast<uint64_t>(ctx.keyShape.GetDim(0));
            shapeSize = static_cast<uint64_t>(ctx.tilingCtx->GetOptionalInputTensor(KEY_INPUT_INDEX)->GetShapeSize());
        }
        keyNonContiguous = (shapeSize != totalElements);
    }
    if (keyNonContiguous) {
        return FailArgReason(ctx.opName, "key",
                             "Key only supports non-contiguous tensor on the 0-axis in PA scenarios");
    }
    return ge::GRAPH_SUCCESS;
}

struct ParseGeomPack {
    int64_t b;
    int64_t s1;
    int64_t s2;
    int64_t n1;
    int64_t n2;
    int64_t g;
    int64_t qH;
    int64_t kH;
    int64_t vH;
    int64_t qT;
    int64_t kvT;
    int64_t dKv;
    int64_t sparseCnt;
    int64_t blkSz;
    uint32_t maxBlkBatch;
};

struct ParseAttrPack {
    int64_t sparseBlkSz;
    float scale;
    int64_t sparseMode;
    int64_t attnMode;
    int64_t keyQ;
    int64_t valueQ;
    int64_t scaleRepo;
    int64_t pre;
    int64_t next;
    int64_t tile;
    int64_t rope;
    bool lse;
};

static ParseGeomPack PackParseGeometry(const ParseCtx &ctx)
{
    ParseGeomPack g{};
    g.maxBlkBatch = ctx.maxBlockNumPerBatch;
    g.blkSz = ctx.blockSize;
    g.sparseCnt = ctx.sparseBlockCount;
    g.dKv = ctx.dSizeKV;
    g.kvT = ctx.kvTSize;
    g.qT = ctx.qTSize;
    g.vH = ctx.vHeadDim;
    g.kH = ctx.kHeadDim;
    g.qH = ctx.qHeadDim;
    g.g = ctx.gSize;
    g.n2 = ctx.n2Size;
    g.n1 = ctx.n1Size;
    g.s2 = ctx.s2Size;
    g.s1 = ctx.s1Size;
    g.b = ctx.bSize;
    return g;
}

static ParseAttrPack PackParseAttrs(const ParseCtx &ctx)
{
    ParseAttrPack a{};
    a.sparseBlkSz = *ctx.opParamInfo.sparseBlockSize;
    a.scale = *ctx.opParamInfo.scaleValue;
    a.sparseMode = *ctx.opParamInfo.sparseMode;
    a.attnMode = *ctx.opParamInfo.attentionMode;
    a.keyQ = *ctx.opParamInfo.keyQuantMode;
    a.valueQ = *ctx.opParamInfo.valueQuantMode;
    a.scaleRepo = *ctx.opParamInfo.quantScaleRepoMode;
    a.pre = *ctx.opParamInfo.preTokens;
    a.next = *ctx.opParamInfo.nextTokens;
    a.tile = *ctx.opParamInfo.tileSize;
    a.rope = *ctx.opParamInfo.ropeHeadDim;
    a.lse = (ctx.opParamInfo.returnSoftmaxLse != nullptr) ? *ctx.opParamInfo.returnSoftmaxLse : false;
    return a;
}

static uint32_t ResolveKeyStride0(const ParseCtx &ctx)
{
    if (ctx.npuArch != NpuArch::DAV_3510) {
        return 0U;
    }
    if (ctx.keyStride0 != 0) {
        return ctx.keyStride0;
    }
    if (ctx.kvLayout == QSFALayout::PA_BSND) {
        return static_cast<uint32_t>(ctx.blockSize * ctx.dSizeKV);
    }
    return 0U;
}

static void TransferParseCtxToTilingInfo(const ParseCtx &ctx, QSFATilingInfo &mlaMeta)
{
    const ParseGeomPack geom = PackParseGeometry(ctx);
    const ParseAttrPack attrs = PackParseAttrs(ctx);

    mlaMeta.opName = ctx.opName;
    mlaMeta.platMeta = ctx.platformInfo;
    mlaMeta.opParamInfo = ctx.opParamInfo;
    mlaMeta.isV2Op = ctx.isV2Op;
    mlaMeta.npuArch = ctx.npuArch;
    mlaMeta.isA5 = ctx.isA5;
    mlaMeta.l2CacheSize = ctx.l2CacheSize;

    mlaMeta.kvStorageMode = ctx.kvStorageMode;
    mlaMeta.topkLayout = ctx.topkLayout;
    mlaMeta.outLayout = ctx.outLayout;
    mlaMeta.kvLayout = ctx.kvLayout;
    mlaMeta.qLayout = ctx.qLayout;

    mlaMeta.inputQType = ctx.inputQType;
    mlaMeta.inputKvType = ctx.inputKvType;
    mlaMeta.outputType = ctx.outputType;

    mlaMeta.bSize = static_cast<uint32_t>(geom.b);
    mlaMeta.n1Size = static_cast<uint32_t>(geom.n1);
    mlaMeta.n2Size = static_cast<uint32_t>(geom.n2);
    mlaMeta.gSize = static_cast<uint32_t>(geom.g);
    mlaMeta.s1Size = static_cast<uint32_t>(geom.s1);
    mlaMeta.s2Size = geom.s2;
    mlaMeta.qHeadDim = static_cast<uint32_t>(geom.qH);
    mlaMeta.kHeadDim = static_cast<uint32_t>(geom.kH);
    mlaMeta.vHeadDim = static_cast<uint32_t>(geom.vH);
    mlaMeta.qTSize = static_cast<uint32_t>(geom.qT);
    mlaMeta.kvTSize = static_cast<uint32_t>(geom.kvT);
    mlaMeta.dSizeVInput = geom.dKv;
    mlaMeta.sparseBlockCount = geom.sparseCnt;
    mlaMeta.blockSize = geom.blkSz;
    mlaMeta.maxBlockNumPerBatch = geom.maxBlkBatch;

    mlaMeta.sparseBlockSize = attrs.sparseBlkSz;
    mlaMeta.scaleValue = attrs.scale;
    mlaMeta.sparseMode = static_cast<uint32_t>(attrs.sparseMode);
    mlaMeta.attentionMode = attrs.attnMode;
    mlaMeta.keyQuantMode = attrs.keyQ;
    mlaMeta.valueQuantMode = attrs.valueQ;
    mlaMeta.quantScaleRepoMode = attrs.scaleRepo;
    mlaMeta.preTokens = attrs.pre;
    mlaMeta.nextTokens = attrs.next;
    mlaMeta.tileSize = attrs.tile;
    mlaMeta.ropeHeadDim = static_cast<uint32_t>(attrs.rope);
    mlaMeta.returnSoftmaxLse = attrs.lse;

    mlaMeta.blockTypeSize = sizeof(float);
    mlaMeta.pageAttentionFlag = (ctx.kvStorageMode == KvStorageMode::PAGE_ATTENTION);
    mlaMeta.totalBlockNum = ctx.opParamInfo.key.shape->GetStorageShape().GetDim(0);
    mlaMeta.keyStride0 = ResolveKeyStride0(ctx);

    mlaMeta.isSameActualseq = ctx.isSameActualseq;
    mlaMeta.isSameSeqAllKVTensor = ctx.isSameSeqAllKVTensor;
    mlaMeta.actualSeqLenFlag = (ctx.opParamInfo.actualSeqLengths.tensor != nullptr);
    mlaMeta.actualQSeqLenFlag = (ctx.opParamInfo.actualSeqLengthsQ.tensor != nullptr);
    mlaMeta.maxActualseq = ctx.maxActualseq;
    mlaMeta.actualLenDimsKV = ctx.actualLenDimsKV;
    mlaMeta.actualLenDimsQ = ctx.actualLenDimsQ;
}

} // namespace

ge::graphStatus QSFAInfoParser::Parse(QSFATilingInfo &mlaMeta)
{
    if (context_ == nullptr) {
        OP_LOGE("RqUniformKvQuantSparseFlashAttention", "tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    ParseCtx ctx;
    ctx.tilingCtx = context_;

    if (BootOpIdentity(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (BindPlatformMeta(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (BindHandlesFromContext(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (AssertRequiredBindings(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseLayoutStringsFromAttrs(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CaptureIoDtypesFromDescs(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CollectShapeMetrics(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyKeyContiguity(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    CaptureActualSeqMeta(ctx);
    TransferParseCtxToTilingInfo(ctx, mlaMeta);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// QSFAMlaTiling — PlanState mutators + emit
// =============================================================================
namespace {

static void PlanInitFromInfo(PlanState &plan, const QSFATilingInfo &info, uint32_t aicNum, uint32_t aivNum,
                             size_t libapiSize, uint32_t mBaseSize)
{
    plan.perfMode = QSFAPerfMode::V_TEMPLATE_MODE;
    plan.aicNum = aicNum;
    plan.aivNum = aivNum;
    plan.libapiSize = libapiSize;
    plan.coreNum = aicNum;
    plan.mBaseSize = mBaseSize;
    plan.headDimAlign = RoundUpTo(info.qHeadDim, BYTE_BLOCK);
    plan.kvSplitPart = 1;
    plan.splitKVFlag = false;
    plan.vectorizeFlag = 0;
}

static void PlanCalcInnerSize(PlanState &plan, const QSFATilingInfo &info, uint32_t kvSpan)
{
    constexpr uint32_t kDefaultS2Inner = 512;
    constexpr uint32_t kFdSplitThreshold = 256;
    plan.sInnerSize = kDefaultS2Inner;

    if (!plan.splitKVFlag || info.qLayout == QSFALayout::TND) {
        // keep default inner tile
    } else if (kvSpan == kFdSplitThreshold) {
        plan.sInnerSize = kFdSplitThreshold / 2U;
    } else if (kvSpan > kFdSplitThreshold && kvSpan <= plan.sInnerSize) {
        plan.sInnerSize = (plan.sInnerSize + 1U) / 2U;
    }

    if (plan.sInnerSize > kvSpan) {
        plan.sInnerSize = kvSpan;
    }
    plan.sInnerLoopTimes = (kvSpan + plan.sInnerSize - 1U) / plan.sInnerSize;
    plan.sInnerSizeTail = kvSpan - (plan.sInnerLoopTimes - 1U) * plan.sInnerSize;
    plan.sInnerSizeAlign = RoundUpTo(plan.sInnerSize, BYTE_BLOCK);

    uint32_t cubeM = info.gSize * info.s1Size;
    if (cubeM > plan.mBaseSize) {
        cubeM = plan.mBaseSize;
    }
    const uint32_t cubeMAlign = RoundUpTo(cubeM, 16U);
    plan.mmResUbSize = plan.sInnerSizeAlign * cubeMAlign;
    plan.bmm2ResUbSize = plan.headDimAlign * cubeMAlign;
    plan.qPreSizeMla = info.gSize * (plan.headDimAlign + 64U) * info.s1Size;
}

static void PlanSplitBalanced(PlanState &plan, const QSFATilingInfo &info)
{
    PlanCalcInnerSize(plan, info, info.s2Size);
    plan.usedCoreNum = plan.aicNum;
}

struct TdEmitSnapshot {
    uint32_t s2Parts;
    uint32_t accumOutElems;
    uint32_t lseElems;
    uint32_t usedCores;
    uint32_t mmUb;
    uint32_t bmm2Ub;
    uint32_t mBase;
    uint32_t s2Base;
    uint32_t batch;
    uint32_t seqKv;
    uint32_t seqQ;
    int64_t blockSz;
    uint32_t maxBlkPerBatch;
    float scale;
    uint32_t qPerG;
    uint32_t actQDims;
    uint32_t actKvDims;
    uint32_t outLayoutBits;
    uint32_t sparseModeBits;
    int64_t sparseBlkSz;
    uint32_t sparseBlkCnt;
    int64_t dSizeV;
    uint32_t actQNull;
    uint32_t actKvNull;
    uint32_t keyStride0;
    uint32_t lseRet;
};

static TdEmitSnapshot BuildTdEmitSnapshot(const PlanState &plan, const QSFATilingInfo &info)
{
    TdEmitSnapshot snap{};
    const uint32_t dualFd = 2U;
    snap.s2Parts = plan.splitKVFlag ? plan.kvSplitPart : 0U;
    snap.accumOutElems = plan.aicNum * dualFd * info.n2Size * plan.mBaseSize * plan.headDimAlign;
    snap.lseElems =
        dualFd * plan.aicNum * dualFd * info.n2Size * plan.mBaseSize * (BYTE_BLOCK / kBlockTableElemBytes);
    snap.usedCores = plan.usedCoreNum;
    snap.mmUb = static_cast<uint32_t>(plan.mmResUbSize);
    snap.bmm2Ub = static_cast<uint32_t>(plan.bmm2ResUbSize);
    snap.mBase = static_cast<uint32_t>(info.gSize);
    snap.s2Base = plan.sInnerSize;
    snap.batch = info.bSize;
    snap.seqKv = static_cast<uint32_t>(info.s2Size);
    snap.seqQ = info.s1Size;
    snap.blockSz = info.blockSize;
    snap.maxBlkPerBatch = info.maxBlockNumPerBatch;
    snap.scale = info.scaleValue;
    snap.qPerG = static_cast<uint32_t>(info.n1Size / info.n2Size);
    snap.actQDims = info.actualLenDimsQ;
    snap.actKvDims = info.actualLenDimsKV;
    snap.outLayoutBits = static_cast<uint32_t>(info.outLayout);
    snap.sparseModeBits = info.sparseMode;
    snap.sparseBlkSz = info.sparseBlockSize;
    snap.sparseBlkCnt = static_cast<uint32_t>(info.sparseBlockCount);
    snap.dSizeV = info.dSizeVInput;
    snap.actQNull = info.actualQSeqLenFlag ? 0U : 1U;
    snap.actKvNull = info.actualSeqLenFlag ? 0U : 1U;
    snap.keyStride0 = info.keyStride0;
    snap.lseRet = info.returnSoftmaxLse ? 1U : 0U;
    return snap;
}

static void EmitTilingData(RqUniformKvQuantSparseFlashAttentionTilingDataMla &td, const PlanState &plan,
                           const QSFATilingInfo &info)
{
    const TdEmitSnapshot snap = BuildTdEmitSnapshot(plan, info);
    // Write order intentionally diverges from OSS FillTiling* (lse/ub first, base last).
    td.splitKVParams.set_s2(snap.s2Parts);
    td.splitKVParams.set_logSumExpSize(snap.lseElems);
    td.splitKVParams.set_accumOutSize(snap.accumOutElems);

    td.innerSplitParams.set_mBaseSize(snap.mBase);
    td.innerSplitParams.set_s2BaseSize(snap.s2Base);
    td.singleCoreParams.set_usedCoreNum(snap.usedCores);
    td.singleCoreTensorSize.set_mmResUbSize(snap.mmUb);
    td.singleCoreTensorSize.set_bmm2ResUbSize(snap.bmm2Ub);

    td.baseParams.set_returnSoftmaxLse(snap.lseRet);
    td.baseParams.set_keyStride0(snap.keyStride0);
    td.baseParams.set_isActualLenDimsKVNull(snap.actKvNull);
    td.baseParams.set_isActualLenDimsNull(snap.actQNull);
    td.baseParams.set_dSizeVInput(snap.dSizeV);
    td.baseParams.set_sparseBlockCount(snap.sparseBlkCnt);
    td.baseParams.set_sparseBlockSize(snap.sparseBlkSz);
    td.baseParams.set_sparseMode(snap.sparseModeBits);
    td.baseParams.set_outputLayout(snap.outLayoutBits);
    td.baseParams.set_actualLenDimsKV(snap.actKvDims);
    td.baseParams.set_actualLenDimsQ(snap.actQDims);
    td.baseParams.set_nNumOfQInOneGroup(snap.qPerG);
    td.baseParams.set_scaleValue(snap.scale);
    td.baseParams.set_maxBlockNumPerBatch(snap.maxBlkPerBatch);
    td.baseParams.set_blockSize(snap.blockSz);
    td.baseParams.set_qSeqSize(snap.seqQ);
    td.baseParams.set_seqSize(snap.seqKv);
    td.baseParams.set_batchSize(snap.batch);
}

static void PlanCalcVectorize(PlanState &plan, const QSFATilingInfo &info)
{
    plan.vectorizeFlag = 0U;
    const bool a5Pa = info.isA5 && (info.kvLayout == QSFALayout::PA_BSND);
    if (!a5Pa) {
        return;
    }
    constexpr uint32_t kLoopS2 = 128U;
    constexpr uint32_t kUbCap = 248U * 1024U;

    const uint64_t blk = static_cast<uint64_t>(info.blockSize);
    const bool pow2Blk = (blk != 0U) && ((blk & (blk - 1U)) == 0U);
    const bool sparseAligned =
        (static_cast<uint32_t>(info.sparseBlockCount) & (kLoopS2 - 1U)) == 0U;
    const bool sparseModeOk = (info.sparseMode == 3U);
    if (!(pow2Blk && sparseAligned && sparseModeOk)) {
        return;
    }

    const uint32_t sparseCnt = static_cast<uint32_t>(info.sparseBlockCount);
    const uint32_t sparseAlignedCnt = (sparseCnt + kLoopS2 - 1U) / kLoopS2 * kLoopS2;
    const uint32_t needUb =
        RoundUpTo<uint32_t>(info.maxBlockNumPerBatch * sizeof(int32_t), 512U) +
        RoundUpTo<uint32_t>(sparseCnt * sizeof(int32_t), BYTE_BLOCK) + sparseAlignedCnt * sizeof(int64_t);
    plan.vectorizeFlag = (needUb <= kUbCap) ? 1U : 0U;
}

static void PlanGenTilingKey(PlanState &plan, const QSFATilingInfo &info)
{
    const uint32_t layoutQuery = static_cast<uint32_t>(info.qLayout);
    const uint32_t layoutKV = static_cast<uint32_t>(info.kvLayout);
    const uint32_t pageAttention = (info.kvLayout == QSFALayout::PA_BSND) ? 1U : 0U;
    plan.tilingKey =
        GET_TPL_TILING_KEY(0U, pageAttention, layoutQuery, layoutKV, plan.perfMode == QSFAPerfMode::V_TEMPLATE_MODE,
                           static_cast<uint32_t>(info.gSize > 64), plan.vectorizeFlag);
}

static void PlanCalcBlockDim(PlanState &plan, fe::PlatFormInfos *platMeta)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platMeta);
    const auto aicNum = plan.usedCoreNum;
    const auto aivNum = 2 * plan.usedCoreNum;
    plan.blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
}

static void SyncBootstrapPlanToMembers(QSFAPerfMode &perfMode, uint32_t &coreNum, uint32_t &headDimAlign,
                                       bool &splitKvFlag, uint32_t &kvSplitPart, const PlanState &plan)
{
    kvSplitPart = plan.kvSplitPart;
    splitKvFlag = plan.splitKVFlag;
    headDimAlign = plan.headDimAlign;
    coreNum = plan.coreNum;
    perfMode = plan.perfMode;
}

static void SyncInnerPlanToMembers(uint32_t &usedCoreNum, size_t &qPreSizeMla, size_t &bmm2ResUbSize, size_t &mmResUbSize,
                                   uint32_t &sInnerSizeAlign, uint32_t &sInnerSizeTail, uint32_t &sInnerLoopTimes,
                                   uint32_t &sInnerSize, const PlanState &plan)
{
    usedCoreNum = plan.usedCoreNum;
    qPreSizeMla = plan.qPreSizeMla;
    bmm2ResUbSize = plan.bmm2ResUbSize;
    mmResUbSize = plan.mmResUbSize;
    sInnerSizeAlign = plan.sInnerSizeAlign;
    sInnerSizeTail = plan.sInnerSizeTail;
    sInnerLoopTimes = plan.sInnerLoopTimes;
    sInnerSize = plan.sInnerSize;
}

} // namespace

ge::graphStatus QSFAMlaTiling::DoOpTiling(QSFATilingInfo *mlaMeta)
{
    qsfaInfo_ = mlaMeta;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    PlanState plan;
    PlanInitFromInfo(plan, *qsfaInfo_, aicNum_, aivNum_, libapiSize_, mBaseSize_);
    SyncBootstrapPlanToMembers(perfMode_, coreNum_, headDimAlign_, splitKVFlag_, kvSplitPart_, plan);

    ZeroTensorProcess();
    PlanSplitBalanced(plan, *qsfaInfo_);
    SyncInnerPlanToMembers(usedCoreNum_, qPreSizeMla_, bmm2ResUbSize_, mmResUbSize_, sInnerSizeAlign_, sInnerSizeTail_,
                           sInnerLoopTimes_, sInnerSize_, plan);

    EmitTilingData(tilingData_, plan, *qsfaInfo_);

    PlanCalcBlockDim(plan, qsfaInfo_->platMeta);
    blockDim_ = plan.blockDim;
    OP_LOGI(qsfaInfo_->opName, "QSFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, 2 * usedCoreNum_,
            usedCoreNum_);

    PlanCalcVectorize(plan, *qsfaInfo_);
    vectorizeFlag_ = plan.vectorizeFlag;

    GetWorkspaceSize();
    plan.workspaceSize = workspaceSize_;

    PlanGenTilingKey(plan, *qsfaInfo_);
    tilingKey_ = plan.tilingKey;
    OP_LOGI(qsfaInfo_->opName, "QSFA tilingKey_: %lu.", tilingKey_);

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t QSFAMlaTiling::CalcBalanceFDParamNums(const uint32_t liveCoreCnt) const
{
    return liveCoreCnt * 2U * qsfaInfo_->n2Size * mBaseSize_;
}

void QSFAMlaTiling::CalcFDWorkSpace(const uint32_t liveCoreCnt)
{
    NormalCalcFDWorkSpace(liveCoreCnt);
}

void QSFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t liveCoreCnt)
{
    if (!splitKVFlag_) {
        return;
    }
    const uint32_t slots = CalcBalanceFDParamNums(liveCoreCnt);
    const uint32_t elemB = qsfaInfo_->blockTypeSize;
    const uint64_t accumElems = static_cast<uint64_t>(slots) * headDimAlign_;
    const uint64_t lseElems = static_cast<uint64_t>(2U) * slots * (BYTE_BLOCK / elemB);
    workspaceSize_ += (accumElems + lseElems) * elemB;
    if (qsfaInfo_->npuArch == NpuArch::DAV_2002) {
        workspaceSize_ += static_cast<size_t>(liveCoreCnt) * 32U;
    }
}

void QSFAMlaTiling::GetWorkspaceSize()
{
    const uint32_t cores = coreNum_;
    uint64_t terms[16];
    size_t termN = 0;

    auto push = [&](uint64_t v) {
        terms[termN++] = v;
    };
    auto sumTerms = [&]() -> uint64_t {
        uint64_t total = 0;
        for (size_t i = 0; i < termN; ++i) {
            total += terms[i];
        }
        return total;
    };

    if (qsfaInfo_->isA5) {
        push(libapiSize_);
        constexpr uint32_t kTriple = 3U;
        constexpr uint32_t kS2 = 128U;
        constexpr uint32_t kD = 576U;
        const auto plat = platform_ascendc::PlatformAscendC(qsfaInfo_->platMeta);
        const uint32_t aic = plat.GetCoreNumAic();
        if (qsfaInfo_->gSize > 64) {
            push(static_cast<uint64_t>(kS2) * kD * BytesOfDtype(qsfaInfo_->inputQType) * kTriple * (aic >> 1U));
        }
        if (vectorizeFlag_ != 0) {
            const uint32_t span =
                (qsfaInfo_->qLayout == QSFALayout::TND) ? qsfaInfo_->s1Size : (qsfaInfo_->bSize * qsfaInfo_->s1Size);
            push(static_cast<uint64_t>(span) * static_cast<uint32_t>(qsfaInfo_->sparseBlockCount) * sizeof(int64_t));
        }
        workspaceSize_ = sumTerms();
        CalcFDWorkSpace(cores);
        return;
    }

    constexpr uint32_t kMmB = 4U;
    constexpr uint32_t kVecB = 2U;
    constexpr uint32_t kBmm2B = 4U;
    constexpr uint32_t kSoftB = 4U;
    constexpr uint32_t kQPreB = 0U;
    constexpr float kRatio = 1.0f;
    const uint32_t preload = kPreloadNum;
    const uint64_t c64 = cores;

    push(libapiSize_);
    push(preload * mmResUbSize_ * c64 * kMmB);
    push(preload * static_cast<uint64_t>(static_cast<float>(mmResUbSize_ * c64 * kVecB) * kRatio));
    push(preload * bmm2ResUbSize_ * c64 * kBmm2B);
    push(preload * static_cast<uint64_t>(static_cast<float>(qPreSizeMla_ * c64 * kQPreB) * kRatio));
    push(preload * mBaseSize_ * c64 * kSoftB);
    push(preload * bmm2ResUbSize_ * c64 * kBmm2B);
    push(static_cast<uint64_t>(4) * 512U * (512U + 64U) * 2U * c64);
    const uint64_t ring = kValidSizeRingBytes * (kAivPerAic * c64);
    push(ring);
    if (qsfaInfo_->keyQuantMode == kKeyQuantTq4) {
        push(ring);
    }

    workspaceSize_ = sumTerms();
    CalcFDWorkSpace(cores);
}

void QSFAMlaTiling::ZeroTensorProcess() const
{
    if (qsfaInfo_->s2Size == 0) {
        qsfaInfo_->s2Size = 1024;
    }
}

ge::graphStatus QSFAMlaTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *plat = qsfaInfo_->platMeta;
    if (plat == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(qsfaInfo_->opName, "GetPlatformInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto qsfaAscendcPlatform = platform_ascendc::PlatformAscendC(plat);
    aicNum_ = qsfaAscendcPlatform.GetCoreNumAic();
    aivNum_ = qsfaAscendcPlatform.GetCoreNumAiv();
    if (aicNum_ == 0 || aivNum_ == 0) {
        OPS_REPORT_VECTOR_INNER_ERR(qsfaInfo_->opName, "num of core obtained is 0.");
        return GRAPH_FAILED;
    }
    libapiSize_ = qsfaAscendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetTilingData(TilingDef &dataBlob) const
{
    auto *rawTiling = context_->GetRawTilingData();
    if (rawTiling == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const size_t payloadBytes = dataBlob.GetDataSize();
    dataBlob.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(payloadBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetWorkspaceSize(uint64_t wkBytes) const
{
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr");
        return ge::GRAPH_FAILED;
    }
    workSpaces[0] = wkBytes;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetTilingKey(uint64_t keyBits) const
{
    context_->SetScheduleMode(1);
    context_->SetTilingKey(keyBits);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetBlockDim(uint32_t aicBlocks) const
{
    context_->SetBlockDim(aicBlocks);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Tiling entry points
// =============================================================================
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttention, RqUniformKvQuantSparseFlashAttentionTilingDataMla)
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttentionInnerSplitParamsOp,
                           RqUniformKvQuantSparseFlashAttentionInnerSplitParams)
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttentionSplitKVParamsMlaOp,
                           RqUniformKvQuantSparseFlashAttentionSplitKVParamsMla)
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttentionSingleCoreTensorSizeMlaOp,
                           RqUniformKvQuantSparseFlashAttentionSingleCoreTensorSizeMla)
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttentionSingleCoreParamsMlaOp,
                           RqUniformKvQuantSparseFlashAttentionSingleCoreParamsMla)
REGISTER_TILING_DATA_CLASS(RqUniformKvQuantSparseFlashAttentionBaseParamsMlaOp,
                           RqUniformKvQuantSparseFlashAttentionBaseParamsMla)

ge::graphStatus TilingPrepareForRqUniformKvQuantSparseFlashAttention(gert::TilingParseContext *const tilingCtx)
{
    (void)tilingCtx;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingRqUniformKvQuantSparseFlashAttention(gert::TilingContext *tilingCtx)
{
    QSFATilingInfo mlaMeta;
    QSFAInfoParser infoParser(tilingCtx);
    if (infoParser.Parse(mlaMeta) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QSFATilingCheck checker(mlaMeta);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QSFAMlaTiling mlaTiling(tilingCtx);
    return mlaTiling.DoOpTiling(&mlaMeta);
}
} // namespace optiling
