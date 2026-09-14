/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"

namespace bit_residual {

using namespace AscendC;

// NOTE: TQ_UB_ALIGN used by TqAlignUp32; define it first.
static constexpr uint32_t TQ_UB_ALIGN = 32;

static constexpr uint32_t TqAlignUp32(uint32_t x) {
    return (x + TQ_UB_ALIGN - 1) / TQ_UB_ALIGN * TQ_UB_ALIGN;
}

#define UB_VARIBALE_AND_OFF(ub_name, ub_size, last_var_name) \
    static constexpr uint32_t ub_name##_OFFSET = TqAlignUp32(last_var_name##_OFFSET + last_var_name##_SIZE); \
    static constexpr uint32_t ub_name##_SIZE = TqAlignUp32(ub_size);


// -----------global--------------------------------------------------------
static constexpr int ASCEND_BLOCK_BYTES = 32;
static constexpr int TQ_PACK_D = 128;
// Must match br_dequant BR_KEY_UNIFORM_SCHEME:
//   1=A asymmetric uniform, 2=B symmetric uniform, 3=C legacy LSB-sign+q7.
// Default A; tests / serving use A. Set to 3 only for legacy A/B perf compare.
#ifndef BR_KEY_UNIFORM_SCHEME
#define BR_KEY_UNIFORM_SCHEME 1
#endif
static constexpr uint32_t TQ_KEY_UNIFORM_SCHEME = BR_KEY_UNIFORM_SCHEME;
static constexpr uint32_t TQ_BLOCK_ROWS = 16;
static constexpr uint32_t TQ_CUBE_M_ALIGN = 16;
static constexpr uint32_t TQ_VECTOR_BATCH = TQ_BLOCK_ROWS;
static constexpr uint32_t TQ_CUBE_BATCH_ELEMS = TQ_BLOCK_ROWS * TQ_PACK_D;
static constexpr uint32_t TQ_ROT_K = TQ_PACK_D;
static constexpr uint32_t TQ_ROT_N = TQ_PACK_D;
static constexpr uint32_t TQ_DTYPE_BYTES = sizeof(uint16_t);

// -----------cube--------------------------------------------------------
static constexpr uint32_t TQ_MANUAL_WORKSPACE_BYTE_OFFSET = 512 * 1024;
static constexpr uint32_t TQ_MANUAL_GROUP_ROWS = TQ_BLOCK_ROWS;
static constexpr uint32_t TQ_MANUAL_STREAM_KIND_COUNT = 2;  // key stream + value stream per tile
static constexpr uint32_t TQ_MANUAL_ROT_TILE_M = 32;
static constexpr uint32_t TQ_MANUAL_ROT_TILE_ELEMS = TQ_MANUAL_ROT_TILE_M * TQ_ROT_K;
static constexpr uint32_t TQ_MANUAL_HEADS_PER_TILE =
    TQ_MANUAL_ROT_TILE_M / TQ_MANUAL_GROUP_ROWS;
static constexpr uint32_t TQ_MANUAL_AIV_SLICE_M = TQ_MANUAL_ROT_TILE_M / 2;
static constexpr uint32_t TQ_MANUAL_HEADS_PER_AIV =
    TQ_MANUAL_AIV_SLICE_M / TQ_MANUAL_GROUP_ROWS;
static constexpr uint32_t TQ_MANUAL_AIV_SLICE_ELEMS =
    TQ_MANUAL_AIV_SLICE_M * TQ_ROT_K;
static constexpr uint32_t TQ_MANUAL_ROT_A_L1_OFFSET = 0;
static constexpr uint32_t TQ_MANUAL_ROT_B_L1_OFFSET =
    TQ_MANUAL_ROT_A_L1_OFFSET + TQ_MANUAL_ROT_TILE_ELEMS * TQ_DTYPE_BYTES;
static constexpr uint32_t TQ_MANUAL_WORKSPACE_BUFFER_COUNT = 2;
static constexpr uint32_t TQ_MANUAL_WORKSPACE_STRIDE_ELEMS =
    TQ_MANUAL_ROT_TILE_M * TQ_ROT_N;
// cWorkGm carries the AIC Fixpipe F322F16 output (half), so the per-core
// payload is measured in half elements, not float.
static constexpr uint32_t TQ_MANUAL_C_WORKSPACE_HALVES_PER_CORE =
    TQ_MANUAL_WORKSPACE_BUFFER_COUNT * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS;
// manualWorkspace_ is __gm__ T* (T = half/bf16, both 2 B).  Index stride must
// cover the same bytes as the per-core half payload.
static constexpr uint32_t TQ_MANUAL_WORKSPACE_ELEMS_PER_CORE =
    TQ_MANUAL_C_WORKSPACE_HALVES_PER_CORE * sizeof(half) / sizeof(uint16_t);
static constexpr uint16_t TQ_MANUAL_SYNC_A_FREE = 0;
static constexpr uint16_t TQ_MANUAL_SYNC_A_READY = 1;
static constexpr uint16_t TQ_MANUAL_SYNC_C_FREE = 2;
static constexpr uint16_t TQ_MANUAL_SYNC_C_READY = 3;
static constexpr uint16_t TQ_MANUAL_SYNC_PP_STRIDE = 4;

// -----------vector--------------------------------------------------------
static constexpr uint32_t TQ_AIV_SUB_BLOCKS = 2;
static constexpr uint32_t TQ_KEY_ROW_CODE_BYTES = TQ_PACK_D;
static constexpr uint32_t TQ_VAL_ROW_CODE_BYTES = TQ_PACK_D / 2;
static constexpr uint32_t TQ_ROW_META_BYTES = sizeof(uint16_t);
static constexpr uint32_t TQ_META_TILE_BYTES = TQ_BLOCK_ROWS * TQ_ROW_META_BYTES;
static constexpr uint32_t TQ_PACKED_TILE_SCRATCH_BYTES =
    2 * TQ_META_TILE_BYTES + TQ_BLOCK_ROWS * TQ_KEY_ROW_CODE_BYTES;
static constexpr uint32_t TQ_KEY_BYTES_PER_ROW = TQ_KEY_ROW_CODE_BYTES + 2 * TQ_ROW_META_BYTES;
static constexpr uint32_t TQ_VAL_BYTES_PER_ROW = TQ_VAL_ROW_CODE_BYTES + 2 * TQ_ROW_META_BYTES;
// ── Encoded tile layout (intermediate format in UB before merge to cache) ──
// Key: 16 contiguous 64-word encoded rows, then 16 bases, then 16 steps.
static constexpr uint32_t TQ_KEY_ENCODED_ROW_BYTES = TQ_PACK_D * sizeof(uint8_t);
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_BYTES = TQ_KEY_ENCODED_ROW_BYTES;
static constexpr uint32_t TQ_KEY_ENCODED_ROW_STRIDE_WORDS =
    TQ_KEY_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_ENCODED_BASE_BATCH_BYTE_OFFSET =
    TQ_VECTOR_BATCH * TQ_KEY_ENCODED_ROW_BYTES;
static constexpr uint32_t TQ_KEY_ENCODED_STEP_BATCH_BYTE_OFFSET =
    TQ_KEY_ENCODED_BASE_BATCH_BYTE_OFFSET + TQ_VECTOR_BATCH * sizeof(uint16_t);
static constexpr uint32_t TQ_KEY_ENCODED_BATCH_BYTES =
    TQ_KEY_ENCODED_STEP_BATCH_BYTE_OFFSET + TQ_VECTOR_BATCH * sizeof(uint16_t);
// Value: 16 contiguous 128-word code rows, then 16 vmins, then 16 vsteps.
static constexpr uint32_t TQ_VAL_ENCODED_ROW_BYTES =
    TQ_PACK_D * sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_BYTES =
    TQ_VAL_ENCODED_ROW_BYTES;
static constexpr uint32_t TQ_VAL_ENCODED_ROW_STRIDE_WORDS =
    TQ_VAL_ENCODED_ROW_STRIDE_BYTES / sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_VMIN_BATCH_BYTE_OFFSET =
    TQ_VECTOR_BATCH * TQ_VAL_ENCODED_ROW_BYTES;
static constexpr uint32_t TQ_VAL_ENCODED_VSTEP_BATCH_BYTE_OFFSET =
    TQ_VAL_ENCODED_VMIN_BATCH_BYTE_OFFSET +
    TQ_VECTOR_BATCH * sizeof(uint16_t);
static constexpr uint32_t TQ_VAL_ENCODED_BATCH_BYTES =
    TQ_VAL_ENCODED_VSTEP_BATCH_BYTE_OFFSET +
    TQ_VECTOR_BATCH * sizeof(uint16_t);

struct ManualKey1StreamDesc {
    uint32_t tokenStart;
    uint32_t slotStart;
    uint32_t blockIdx;
    uint32_t valueGroupInBlock;
    uint32_t headTileStart;
    uint32_t startGroupRow;
    uint32_t validRows;
    uint32_t streamOrdinal;
    bool preserveValue;
    bool isValue;
};

// ── LocalTensor static buffer utilities (following turboquant_pack_kv_for_cache4bit) ──────

template <typename T>
__aicore__ inline AscendC::LocalTensor<T> TqMakeLocalTensor(
    TPosition pos,
    uint32_t address,
    uint32_t elems) {
    AscendC::TBuffAddr tensorAddr {};
    tensorAddr.dataLen = elems * sizeof(T);
    tensorAddr.bufferAddr = address;
    tensorAddr.bufferHandle = nullptr;
    tensorAddr.logicPos = static_cast<uint8_t>(pos);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    tensorAddr.absAddr = GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(pos)) + address;
#endif
    AscendC::LocalTensor<T> tensor;
    tensor.SetAddr(tensorAddr);
    return tensor;
}

template <TPosition Pos>
struct TqStaticLocalBuffer {
    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> GetBufferByByte(
        uint32_t address,
        uint32_t bytes) const {
        return TqMakeLocalTensor<T>(Pos, address, bytes / sizeof(T));
    }
};

struct TqStaticLocalResource {
    TqStaticLocalBuffer<TPosition::VECIN> vecIn;
    TqStaticLocalBuffer<TPosition::VECOUT> vecOut;
    TqStaticLocalBuffer<TPosition::VECCALC> vecCalc;
};

#if defined(ORIG_DTYPE_KEY)
#if (ORIG_DTYPE_KEY == DT_BF16)
using TqDataT = bfloat16_t;
#else
using TqDataT = half;
#endif
#elif defined(DTYPE_KEY)
#if (DTYPE_KEY == DT_BF16)
using TqDataT = bfloat16_t;
#else
using TqDataT = half;
#endif
#else
using TqDataT = half;
#endif

__aicore__ inline uint32_t TqManualRawBlockIdx() {
    if ASCEND_IS_AIV {
        return static_cast<uint32_t>(AscendC::GetBlockIdx() /
                                     AscendC::GetSubBlockNum());
    }
    return static_cast<uint32_t>(AscendC::GetBlockIdx());
}

using TqManualMmadResource = Catlass::Arch::Resource<Catlass::Arch::AtlasA2>;


template <AscendC::HardEvent EVT>
__aicore__ inline void TqSyncFixed(uint32_t eventId = EVENT_ID0) {
    SetFlag<EVT>(eventId);
    WaitFlag<EVT>(eventId);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreSet(uint16_t flag) {
    AscendC::CrossCoreSetFlag<0x2, PIPE>(flag);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreWait(uint16_t flag) {
    AscendC::CrossCoreWaitFlag<0x2, PIPE>(flag);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreWaitForBothAiv(uint16_t flag) {
    TqCrossCoreWait<PIPE>(flag);
}

template <pipe_t PIPE>
__aicore__ inline void TqCrossCoreSetForBothAiv(uint16_t flag) {
    TqCrossCoreSet<PIPE>(flag);
}

// csrc/kernels build has no PipeSync / SetWaitFlag; use SetFlag+WaitFlag (moe_gating_top_k).
__aicore__ inline void TqSyncMte2ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void TqSyncMte2ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e);
    WaitFlag<HardEvent::MTE2_S>(e);
}

__aicore__ inline void TqSyncVToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
}

__aicore__ inline void TqSyncSToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
}

__aicore__ inline void TqSyncSToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

__aicore__ inline void TqSyncVToMte3() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void TqSyncMte3ToMte2() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

__aicore__ inline void TqSyncMte3ToV() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e);
    WaitFlag<HardEvent::MTE3_V>(e);
}

__aicore__ inline void TqSyncMte3ToS() {
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(e);
    WaitFlag<HardEvent::MTE3_S>(e);
}


// Group rows are not necessarily 32B-aligned; use DataCopyPad for GM writes.
__aicore__ inline void copy_packed_ub_to_gm(
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    AscendC::LocalTensor<uint8_t>& packedLocal,
    uint32_t nbytes) {
    TqSyncVToMte3();
    TqSyncSToMte3();
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPad(packedGm[gm_offset], packedLocal, copyParams);
    TqSyncMte3ToMte2();
    TqSyncMte3ToV();
    TqSyncMte3ToS();
}

__aicore__ inline void copy_packed_gm_to_ub(
    AscendC::LocalTensor<uint8_t>& packedLocal,
    AscendC::GlobalTensor<uint8_t>& packedGm,
    uint64_t gm_offset,
    uint32_t nbytes) {
    AscendC::DataCopyExtParams copyParams{1, nbytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(packedLocal, packedGm[gm_offset], copyParams, padParams);
    TqSyncMte2ToV();
    TqSyncMte2ToS();
}

// ── VECCALC (UB) static layout ──────────────────────────────────────────────────────
//
// Hand-built LocalTensor buffer addresses live in one logical UB address space.
// Do not reuse the same byte offsets across VECIN/VECOUT/VECCALC positions.
// The rotated half input (AIC Fixpipe F322F16 output) uses two slots so MTE2
// can prefetch stream N+1 while the vector pipeline encodes stream N.  Encode
// scratch and encoded output stay single-buffered because they are consumed in
// order by the cache RMW stage.
// ── Base variable (offset = 0, no predecessor) ────────────────────────────────

// ── A_ENCODED has a max-expression size; pre-compute before chaining ──────────
static constexpr uint32_t TQ_UB_ENCODED_BATCH_BYTES =
    (TQ_KEY_ENCODED_BATCH_BYTES > TQ_VAL_ENCODED_BATCH_BYTES)
        ? TQ_KEY_ENCODED_BATCH_BYTES
        : TQ_VAL_ENCODED_BATCH_BYTES;

static constexpr uint32_t TQ_UB_BASE_OFFSET = 0;
static constexpr uint32_t TQ_UB_BASE_SIZE = 0;

static constexpr uint32_t TQ_BATCH_ELEMS = TQ_VECTOR_BATCH * TQ_PACK_D;
static constexpr uint32_t TQ_UB_ENCODE_FP32_ROW_BYTES = TQ_PACK_D * sizeof(float);  // 512 B per fp32 row
// fat slot: one full 16×128 half tile (A / quant_i16 / final_quant / pack_u8),
// 2048 elements × 2 B = 4 KB.  Also doubles as fp32 scratch (16 floats) for the
// bf16 metadata path once the quant chain has finished.
static constexpr uint32_t TQ_UB_ENCODE_SMALL_BUF = TQ_BLOCK_ROWS * TQ_PACK_D * sizeof(half);   // 4 KB
// Scalar (reduce / Brcb) tiles live in the half domain.  A Brcb broadcast of
// one per-row scalar produces one datablock per row = VECTOR_BATCH datablocks,
// i.e. VECTOR_BATCH × (ASCEND_BLOCK_BYTES/sizeof(half)) half elements.  This is
// also the worst-case per-row reduce output width (one datablock per row), so it
// sizes every scalar sub-block slot.
static constexpr uint32_t TQ_ENCODE_SCALAR_TILE_ELEMS =
    TQ_VECTOR_BATCH * (ASCEND_BLOCK_BYTES / sizeof(half));
static constexpr uint32_t TQ_ENCODE_SCALAR_SLOTS = 4;  // max→baseBlk, min, gap→step, stepBlk
// scalar slot: TQ_ENCODE_SCALAR_SLOTS sub-blocks of TQ_ENCODE_SCALAR_TILE_ELEMS
// half each.  Peak occupancy is min + step + one transient.
static constexpr uint32_t TQ_UB_ENCODE_SCALAR_BUF = TqAlignUp32(
    TQ_ENCODE_SCALAR_SLOTS * TQ_ENCODE_SCALAR_TILE_ELEMS * sizeof(half));

// ── Chained layout via UB_VARIBALE_AND_OFF ────────────────────────────────────
// XY_BATCH/XY_BATCH1 hold the AIC Fixpipe F322F16 half output directly (no
// fp32→half Cast on the AIV), so each slot is one 16×128 half tile = 4 KB.
UB_VARIBALE_AND_OFF(TQ_UB_XY_BATCH,        TQ_BATCH_ELEMS * sizeof(half), TQ_UB_BASE)
UB_VARIBALE_AND_OFF(TQ_UB_XY_BATCH1,       TQ_BATCH_ELEMS * sizeof(half), TQ_UB_XY_BATCH)
// ENCODE_TMPB: the third 4 KB half slot freed by halving the two XY slots
// (8 KB fp32 → 4 KB half each).  Home of signVec (key) / pack_half in the
// half-input pipeline where fat is already busy as absVec/shiftedSign.
UB_VARIBALE_AND_OFF(TQ_UB_ENCODE_TMPB,     TQ_UB_ENCODE_SMALL_BUF, TQ_UB_XY_BATCH1)
UB_VARIBALE_AND_OFF(TQ_UB_ENCODE_FAT,      TQ_UB_ENCODE_SMALL_BUF, TQ_UB_ENCODE_TMPB)     // fat: absVec/quant/final/pack_u8 + bf16 meta scratch
UB_VARIBALE_AND_OFF(TQ_UB_ENCODE_SCALAR,   TQ_UB_ENCODE_SCALAR_BUF, TQ_UB_ENCODE_FAT)    // scal: max→baseBlk, min, gap→step, stepBlk
UB_VARIBALE_AND_OFF(TQ_UB_A_ENCODED_BATCH, TQ_UB_ENCODED_BATCH_BYTES, TQ_UB_ENCODE_SCALAR)
UB_VARIBALE_AND_OFF(TQ_UB_PACKED_ROW,      TQ_PACKED_TILE_SCRATCH_BYTES, TQ_UB_A_ENCODED_BATCH)
static constexpr uint32_t TQ_UB_TOTAL_BYTES =
    TqAlignUp32(TQ_UB_PACKED_ROW_OFFSET + TQ_UB_PACKED_ROW_SIZE);
static_assert(TQ_UB_TOTAL_BYTES <= TOTAL_UB_SIZE,
              "Bit-residual K8v4 static UB slices exceed UB size.");

template <typename T>
class BitResidualPackK8v4Resource {
public:
    __aicore__ inline void Init() {}

    // ── Row buffers ───────────────────────────────────────────────────────────
    //   Time-shared views: XY_BATCH and A_ENCODED_BATCH each hold multiple
    //   accessors with different element types/counts.  Use _SIZE for the full
    //   slot size; sub-view counts are derived from the raw constants.
    __aicore__ inline AscendC::LocalTensor<half> YBatchHalf(
        uint32_t bufferIndex = 0) {
        const uint32_t offset = bufferIndex == 0
            ? TQ_UB_XY_BATCH_OFFSET
            : TQ_UB_XY_BATCH1_OFFSET;
        return local_.vecCalc.GetBufferByByte<half>(
            offset,
            TQ_UB_XY_BATCH_SIZE);
    }

    // ── Encode buffers ─────────────────────────────────────────────────────
    //   fat (4 KB): one full half tile for the A → quant → final → pack_u8
    //   chain; also the fp32 scratch for the bf16 metadata path.
    //   scal (2 KB): four 256-half sub-blocks for per-row reduce scalars and
    //   Brcb broadcast tiles (max→baseBlk, min, gap→step, stepBlk).
    __aicore__ inline AscendC::LocalTensor<uint8_t> EncodeBuffer1() {
        return local_.vecCalc.GetBufferByByte<uint8_t>(
            TQ_UB_ENCODE_FAT_OFFSET, TQ_UB_ENCODE_FAT_SIZE);
    }
    __aicore__ inline AscendC::LocalTensor<uint8_t> EncodeBuffer2() {
        return local_.vecCalc.GetBufferByByte<uint8_t>(
            TQ_UB_ENCODE_SCALAR_OFFSET, TQ_UB_ENCODE_SCALAR_SIZE);
    }
    // tmpB (4 KB): third half scratch slot — home of signVec (key encoder,
    // Step 2) reused as pack_half (Step 13).  Carved from the 4 KB freed by
    // halving each XY slot from 8 KB fp32 to 4 KB half.
    __aicore__ inline AscendC::LocalTensor<uint8_t> EncodeTmpB() {
        return local_.vecCalc.GetBufferByByte<uint8_t>(
            TQ_UB_ENCODE_TMPB_OFFSET, TQ_UB_ENCODE_TMPB_SIZE);
    }

    __aicore__ inline AscendC::LocalTensor<uint16_t> KeyEncodedBatch() {
        return local_.vecCalc.GetBufferByByte<uint16_t>(
            TQ_UB_A_ENCODED_BATCH_OFFSET,
            TQ_UB_A_ENCODED_BATCH_SIZE);
    }

    // ── VECCALC (UB compute scratch) ─────────────────────────────────────────
    __aicore__ inline AscendC::LocalTensor<uint8_t> PackedRow() {
        return local_.vecCalc.GetBufferByByte<uint8_t>(
            TQ_UB_PACKED_ROW_OFFSET, TQ_UB_PACKED_ROW_SIZE);
    }

private:
    TqStaticLocalResource local_;
};


template <typename T>
class BitResidualPackK8v4Context {
public:
    __aicore__ inline explicit BitResidualPackK8v4Context(
        __gm__ uint8_t* rawWorkspace,
        uint32_t nVec,
        uint32_t vecPerCore,
        uint32_t numHeads,
        uint32_t blockSize,
        uint32_t numBlocks,
        uint32_t dataCores,
        uint32_t numReqs,
        uint32_t keyStrideToken,
        uint32_t keyStrideHead,
        uint32_t valueStrideToken,
        uint32_t valueStrideHead,
        uint64_t keyStorageOffset,
        uint64_t valueStorageOffset)
        : nVec_(nVec),
          numHeads_(numHeads == 0 ? 1 : numHeads),
          tokenCount_((nVec + (numHeads == 0 ? 1 : numHeads) - 1) / (numHeads == 0 ? 1 : numHeads)),
          blockSize_(blockSize == 0 ? 1 : blockSize),
          numBlocks_(numBlocks),
          cacheSlots_((blockSize == 0 ? 1 : blockSize) * numBlocks),
          dataCores_(dataCores == 0 ? 1 : dataCores),
          numReqs_(numReqs),
          keyStrideToken_(keyStrideToken),
          keyStrideHead_(keyStrideHead),
          valueStrideToken_(valueStrideToken),
          valueStrideHead_(valueStrideHead),
          keyStorageOffset_(keyStorageOffset),
          valueStorageOffset_(valueStorageOffset),
          keySpan_(MakeInputSpan(keyStorageOffset, keyStrideToken, keyStrideHead)),
          valueSpan_(MakeInputSpan(valueStorageOffset, valueStrideToken, valueStrideHead)),
          workspaceReady_(rawWorkspace != nullptr),
          manualWorkspace_(rawWorkspace == nullptr
                               ? nullptr
                               : reinterpret_cast<__gm__ T*>(
                                     rawWorkspace + TQ_MANUAL_WORKSPACE_BYTE_OFFSET)) {
        (void)vecPerCore;
    }

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        __gm__ T* rotation_t,
        __gm__ int32_t* slot_mapping,
        __gm__ int32_t* query_start_loc,
        __gm__ uint8_t* key_cache,
        __gm__ uint8_t* value_cache) {
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key), keySpan_);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), valueSpan_);
        rotationTGm_.SetGlobalBuffer(rotation_t, (uint64_t)TQ_PACK_D * TQ_PACK_D);
        slotMappingGm_.SetGlobalBuffer(slot_mapping, tokenCount_);
        queryStartLocGm_.SetGlobalBuffer(query_start_loc, static_cast<uint64_t>(numReqs_) + 1);
        keyCacheGm_.SetGlobalBuffer(
            key_cache,
            (uint64_t)numBlocks_ * numHeads_ *
                blockSize_ * TQ_KEY_BYTES_PER_ROW);
        valueCacheGm_.SetGlobalBuffer(
            value_cache,
            (uint64_t)numBlocks_ * numHeads_ *
                blockSize_ * TQ_VAL_BYTES_PER_ROW);

        resource_.Init();
    }

public:

    __aicore__ inline void InitManualWorkspaceTensors(
        uint32_t manualGroupId,
        AscendC::GlobalTensor<half>& cWorkGm) const {
        __gm__ T* groupWork =
            manualWorkspace_ + static_cast<uint64_t>(manualGroupId) *
                                   TQ_MANUAL_WORKSPACE_ELEMS_PER_CORE;
        cWorkGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(groupWork),
            TQ_MANUAL_C_WORKSPACE_HALVES_PER_CORE);
    }

    __aicore__ inline uint32_t ManualStreamBufferIndex(uint32_t streamOrdinal) const {
        return streamOrdinal & 1u;
    }

    __aicore__ inline uint16_t ManualFlagBase(uint32_t streamOrdinal) const {
        return static_cast<uint16_t>((streamOrdinal & 1u) * TQ_MANUAL_SYNC_PP_STRIDE);
    }

    __aicore__ inline uint32_t ManualBufferOffset(uint32_t streamOrdinal) const {
        return ManualStreamBufferIndex(streamOrdinal) * TQ_MANUAL_WORKSPACE_STRIDE_ELEMS;
    }

    __aicore__ inline uint64_t MakeInputSpan(
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead) const {
        if (tokenCount_ == 0 || numHeads_ == 0) {
            return storageOffset;
        }
        return storageOffset +
               static_cast<uint64_t>(tokenCount_ - 1) * strideToken +
               static_cast<uint64_t>(numHeads_ - 1) * strideHead +
               TQ_PACK_D;
    }

    __aicore__ inline bool ReadRequestTokenRange(
        uint32_t reqIdx,
        uint32_t& tokenStart,
        uint32_t& tokenEnd) const {
        if (reqIdx >= numReqs_) {
            return false;
        }
        const int32_t start = queryStartLocGm_.GetValue(reqIdx);
        const int32_t end = queryStartLocGm_.GetValue(reqIdx + 1);
        if (start < 0 || end < start) {
            return false;
        }
        tokenStart = static_cast<uint32_t>(start);
        tokenEnd = static_cast<uint32_t>(end);
        if (tokenStart > tokenCount_ || tokenEnd > tokenCount_) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool CanUseManualKey1() const {
        if (!workspaceReady_ || manualWorkspace_ == nullptr ||
            dataCores_ == 0 || numHeads_ < TQ_MANUAL_HEADS_PER_TILE ||
            numHeads_ % TQ_MANUAL_HEADS_PER_TILE != 0 ||
            tokenCount_ == 0 || tokenCount_ * numHeads_ != nVec_ ||
            blockSize_ % TQ_BLOCK_ROWS != 0 ||
            keyStrideHead_ != TQ_PACK_D ||
            valueStrideHead_ != TQ_PACK_D ||
            keyStrideToken_ == 0 ||
            valueStrideToken_ == 0) {
            return false;
        }

        uint32_t totalTiles = 0;
        for (uint32_t reqIdx = 0; reqIdx < numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!ReadRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                return false;
            }
            const uint32_t rowCount = seqEnd - seqStart;
            if (rowCount == 0) {
                continue;
            }
            // Gate only: first token must have a valid slot. Per-token paged
            // addressing is handled in ResolvePackChunk (do not assume
            // firstSlot+rowCount is a contiguous physical range).
            uint32_t firstSlot = 0;
            if (!ResolveTokenSlotU(seqStart, firstSlot)) {
                return false;
            }
            totalTiles += ((rowCount + TQ_MANUAL_GROUP_ROWS - 1) / TQ_MANUAL_GROUP_ROWS) *
                          (numHeads_ / TQ_MANUAL_HEADS_PER_TILE);
        }
        return totalTiles != 0;
    }


    __aicore__ inline bool ResolveTokenSlotU(
        uint32_t tokenIdx,
        uint32_t& slotU) const {
        if (tokenIdx >= tokenCount_) {
            return false;
        }
        const int32_t slot = slotMappingGm_.GetValue(tokenIdx);
        if (slot < 0 || static_cast<uint32_t>(slot) >= cacheSlots_) {
            return false;
        }
        slotU = static_cast<uint32_t>(slot);
        return true;
    }

    // One ManualKey1 stream must cover a contiguous physical-slot run that
    // stays inside a single 16-row tile group and a single PA block. Callers
    // re-resolve at each chunk so non-contiguous block_table (e.g. [6,5])
    // maps token 128 to slot 5*bs rather than firstSlot+128.
    __aicore__ inline bool ResolvePackChunk(
        uint32_t tokenStart,
        uint32_t tokensLeft,
        uint32_t& slotOut,
        uint32_t& startGroupRowOut,
        uint32_t& validRowsOut) const {
        if (tokensLeft == 0 || !ResolveTokenSlotU(tokenStart, slotOut)) {
            return false;
        }
        startGroupRowOut = slotOut % TQ_MANUAL_GROUP_ROWS;
        const uint32_t rowsInGroup = TQ_MANUAL_GROUP_ROWS - startGroupRowOut;
        const uint32_t rowsInBlock = blockSize_ - (slotOut % blockSize_);
        uint32_t n = tokensLeft;
        if (n > rowsInGroup) {
            n = rowsInGroup;
        }
        if (n > rowsInBlock) {
            n = rowsInBlock;
        }
        validRowsOut = n;
        return n != 0;
    }

public:
    BitResidualPackK8v4Resource<T> resource_;
    const uint32_t nVec_;
    const uint32_t numHeads_;
    const uint32_t tokenCount_;
    const uint32_t blockSize_;
    const uint32_t numBlocks_;
    const uint32_t cacheSlots_;
    const uint32_t dataCores_;
    const uint32_t numReqs_;
    const uint32_t keyStrideToken_;
    const uint32_t keyStrideHead_;
    const uint32_t valueStrideToken_;
    const uint32_t valueStrideHead_;
    const uint64_t keyStorageOffset_;
    const uint64_t valueStorageOffset_;
    const uint64_t keySpan_;
    const uint64_t valueSpan_;
    const bool workspaceReady_;
    __gm__ T* const manualWorkspace_;

    AscendC::GlobalTensor<T> keyGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> rotationTGm_;
    AscendC::GlobalTensor<int32_t> slotMappingGm_;
    AscendC::GlobalTensor<int32_t> queryStartLocGm_;
    AscendC::GlobalTensor<uint8_t> keyCacheGm_;
    AscendC::GlobalTensor<uint8_t> valueCacheGm_;
};

}  // namespace bit_residual
