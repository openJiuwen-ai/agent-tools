/*!
 * \file rq_uniform_kv_quant_sparse_flash_attention_tiling.h
 * \brief
 */
#ifndef RQ_UNIFORM_KV_QUANT_SFA_TILING_HOST_H
#define RQ_UNIFORM_KV_QUANT_SFA_TILING_HOST_H

#include <sstream>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/soc_spec.h"
namespace optiling {
// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INPUT_INDEX = 3;
constexpr uint32_t KEY_DEQUANT_SCALE_INPUT_INDEX = 4;
constexpr uint32_t VALUE_DEQUANT_SCALE_INPUT_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 6;
constexpr uint32_t ACT_SEQ_LEN_Q_INPUT_INDEX = 7;
constexpr uint32_t ACT_SEQ_LEN_KV_INPUT_INDEX = 8;
constexpr uint32_t SINKS_INPUT_INDEX = 9;
// Outputs Index
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t SOFTMAX_MAX_INDEX = 1;
constexpr uint32_t SOFTMAX_SUM_INDEX = 2;
// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t KEY_QUANT_MODE_ATTR_INDEX = 1;
constexpr uint32_t VALUE_QUANT_MODE_ATTR_INDEX = 2;
constexpr uint32_t SPARSE_BLOCK_SIZE_ATTR_INDEX = 3;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 5;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 6;
constexpr uint32_t PRE_TOKENS_ATTR_INDEX = 7;
constexpr uint32_t NEXT_TOKENS_ATTR_INDEX = 8;
constexpr uint32_t ATTENTION_MODE_ATTR_INDEX = 9;
constexpr uint32_t QUANT_SCALE_REPO_MODE_ATTR_INDEX = 10;
constexpr uint32_t TILE_SIZE_ATTR_INDEX = 11;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 12;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 13;
// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;

// Dim Num
constexpr size_t DIM_NUM_ONE = 1;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t QSFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

// ------------------公共定义--------------------------
enum class QSFALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

struct QSFATilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class QSFAPerfMode : uint32_t {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE
};

enum class QSFAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3, // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7, // block size
};

struct QSFARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QSFAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// -----------算子Tiling入参结构体定义---------------
struct QSFAParaInfo {
    QSFARequiredParaInfo query = {nullptr, nullptr};
    QSFARequiredParaInfo key = {nullptr, nullptr};
    QSFARequiredParaInfo value = {nullptr, nullptr};
    QSFARequiredParaInfo sparseIndices = {nullptr, nullptr};
    QSFAOptionalParaInfo blockTable = {nullptr, nullptr};
    QSFAOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    QSFAOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    QSFAOptionalParaInfo queryRope = {nullptr, nullptr};
    QSFAOptionalParaInfo keyRope = {nullptr, nullptr};
    QSFAOptionalParaInfo keyDequantScale = {nullptr, nullptr};
    QSFAOptionalParaInfo valueDequantScale = {nullptr, nullptr};
    QSFAOptionalParaInfo sinks = {nullptr, nullptr};
    QSFARequiredParaInfo attenOut = {nullptr, nullptr};
    QSFARequiredParaInfo softmaxMax = {nullptr, nullptr};
    QSFARequiredParaInfo softmaxSum = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const uint32_t *sparseBlockCount = nullptr;
    const uint32_t *blockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
    const int64_t *attentionMode = nullptr;
    const int64_t *keyQuantMode = nullptr;
    const int64_t *valueQuantMode = nullptr;
    const int64_t *quantScaleRepoMode = nullptr;
    const int64_t *tileSize = nullptr;
    const int64_t *ropeHeadDim = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(RqUniformKvQuantSparseFlashAttentionBaseParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(int64_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(int64_t, dSizeVInput)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsNull)
TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsKVNull)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0) // PA mode non-contiguous stride
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse)
END_TILING_DATA_DEF

BEGIN_TILING_DATA_DEF(RqUniformKvQuantSparseFlashAttentionSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF

BEGIN_TILING_DATA_DEF(RqUniformKvQuantSparseFlashAttentionSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF

BEGIN_TILING_DATA_DEF(RqUniformKvQuantSparseFlashAttentionSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)            // S2切分份数
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)  // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize) // FD workspace
END_TILING_DATA_DEF

// 内切基本块参数
BEGIN_TILING_DATA_DEF(RqUniformKvQuantSparseFlashAttentionInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF

BEGIN_TILING_DATA_DEF(RqUniformKvQuantSparseFlashAttentionTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(RqUniformKvQuantSparseFlashAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(RqUniformKvQuantSparseFlashAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(RqUniformKvQuantSparseFlashAttentionSingleCoreParamsMla, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(RqUniformKvQuantSparseFlashAttentionSingleCoreTensorSizeMla, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(RqUniformKvQuantSparseFlashAttentionInnerSplitParams, innerSplitParams);
END_TILING_DATA_DEF

template <typename T>
inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

std::string QSFALayoutToSerialString(QSFALayout layout);

// -----------算子Tiling入参信息类---------------
struct QSFATilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platMeta = nullptr;
    QSFAParaInfo opParamInfo;
    bool isV2Op = false;

    // Base Param
    NpuArch npuArch = NpuArch::DAV_2201;
    bool isA5 = false;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qHeadDim = 0;
    uint32_t kHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0;  // 仅TND时生效
    uint32_t kvTSize = 0; // 仅TND时生效
    float scaleValue = 0;
    uint32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;

    bool pageAttentionFlag = false;
    int64_t blockSize = 0;
    uint32_t blockTypeSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;

    bool actualQSeqLenFlag = false;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;
    std::vector<int64_t> kvListSeqLens{};

    uint32_t sparseMode = 0;

    int64_t attentionMode = 0;
    int64_t keyQuantMode = 0;
    int64_t valueQuantMode = 0;
    int64_t quantScaleRepoMode = 0;
    int64_t tileSize = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;

    QSFALayout qLayout = QSFALayout::BSND;
    QSFALayout topkLayout = QSFALayout::BSND;
    QSFALayout outLayout = QSFALayout::BSND;
    QSFALayout kvLayout = QSFALayout::BSND;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;
    int64_t dSizeVInput = 0;

    bool returnSoftmaxLse = false;

    uint32_t keyStride0 = 0; // PA mode non-contiguous stride on 0-axis
};

// ---------------算子Tiling类---------------
class QSFAMlaTiling {
public:
    explicit QSFAMlaTiling(gert::TilingContext *tilingCtx)
        : context_(tilingCtx)
    {}
    ge::graphStatus DoOpTiling(QSFATilingInfo *mlaMeta);

private:
    ge::graphStatus SetBlockDim(uint32_t aicBlocks) const;
    ge::graphStatus SetTilingKey(uint64_t keyBits) const;
    ge::graphStatus SetWorkspaceSize(uint64_t wkBytes) const;
    ge::graphStatus SetTilingData(TilingDef &dataBlob) const;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus GetPlatformInfo();
    void ZeroTensorProcess() const;
    void NormalCalcFDWorkSpace(const uint32_t liveCoreCnt);
    void CalcFDWorkSpace(const uint32_t liveCoreCnt);
    void GetWorkspaceSize();
    uint32_t CalcBalanceFDParamNums(const uint32_t liveCoreCnt) const;

    bool balanceModeFlag_ = false;
    bool splitKVFlag_ = false;
    uint32_t vectorizeFlag_ = 0;

    uint32_t coreNum_ = 0;
    QSFAPerfMode perfMode_ = QSFAPerfMode::V_TEMPLATE_MODE;
    uint32_t kvSplitPart_ = 1;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSizeMla_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    RqUniformKvQuantSparseFlashAttentionTilingDataMla tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;

    QSFATilingInfo *qsfaInfo_ = nullptr;
};

// -----------算子Tiling入参信息解析及Check类---------------
class QSFATilingCheck {
public:
    explicit QSFATilingCheck(const QSFATilingInfo &mlaMeta)
        : qsfaInfo_(mlaMeta) {};
    ~QSFATilingCheck() = default;
    ge::graphStatus Process();

private:
    const QSFATilingInfo &qsfaInfo_;
};

class QSFAInfoParser {
public:
    explicit QSFAInfoParser(const gert::TilingContext *tilingCtx)
        : context_(tilingCtx)
    {}
    ~QSFAInfoParser() = default;

    ge::graphStatus Parse(QSFATilingInfo &mlaMeta);

private:
    const gert::TilingContext *context_ = nullptr;
};

struct RqUniformKvQuantSparseFlashAttentionCompileInfo {
    int64_t aicCount;
};

ge::graphStatus TilingPrepareForRqUniformKvQuantSparseFlashAttention(gert::TilingParseContext *const tilingCtx);
ge::graphStatus TilingRqUniformKvQuantSparseFlashAttention(gert::TilingContext *tilingCtx);
} // namespace optiling
#endif // RQ_UNIFORM_KV_QUANT_SFA_TILING_HOST_H
