/*!
 * \file pack_k8v4_ub_layout.h
 * \brief A5 UB layout — mode0 fat/scal/encodedBatch + Key fp8 code-only.
 *
 * Value encodedBatch offsets match mode0 TqValEncodedVminOff/VstepOff
 * (TQ_VECTOR_BATCH=16). Tile rows are fixed to 16 like mode0.
 */
#pragma once

#include <cstdint>

#if !defined(__aicore__)
#define __aicore__
#define K8A5_LOCAL_AICORE_FALLBACK 1
#endif

namespace rq_uniform {
namespace arch35 {

// Same as mode0 TQ_VECTOR_BATCH / TQ_BLOCK_ROWS.
static constexpr uint32_t K8_A5_TILE_ROWS = 16;
static constexpr uint32_t K8_A5_PP = 2;
// mode0 TQ_ENCODE_SCALAR_TILE_ELEMS
static constexpr uint32_t K8_A5_SCALAR_TILE_ELEMS = K8_A5_TILE_ROWS * 16U;
static constexpr uint32_t K8_A5_SCALAR_SLOTS = 4;
static constexpr uint32_t K8_A5_SCALAR_BUF_BYTES =
    K8_A5_SCALAR_SLOTS * K8_A5_SCALAR_TILE_ELEMS * 2U;

__aicore__ inline uint32_t K8A5Align32(uint32_t x)
{
    return (x + 31U) / 32U * 32U;
}

// mode0 TqValEncodedVminOff(packD) = TQ_VECTOR_BATCH * packD * sizeof(uint16_t)
__aicore__ inline uint32_t K8A5ValEncodedVminBytes(uint32_t packD)
{
    return K8_A5_TILE_ROWS * packD * 2U;
}

__aicore__ inline uint32_t K8A5ValEncodedVstepBytes(uint32_t packD)
{
    return K8A5ValEncodedVminBytes(packD) + K8_A5_TILE_ROWS * 2U;
}

__aicore__ inline uint32_t K8A5ValEncodedBytes(uint32_t packD)
{
    return K8A5Align32(K8A5ValEncodedVstepBytes(packD) + K8_A5_TILE_ROWS * 2U);
}

__aicore__ inline uint32_t K8A5UbBytes(uint32_t packD)
{
    const uint32_t tileRows = K8_A5_TILE_ROWS;
    const uint32_t srcK = tileRows * packD * 2U;
    const uint32_t srcV = tileRows * packD * 2U;
    const uint32_t fat = tileRows * packD * 2U;
    const uint32_t scal = K8_A5_SCALAR_BUF_BYTES;
    const uint32_t keyCode = tileRows * packD;  // fp8 code only
    const uint32_t valEnc = K8A5ValEncodedBytes(packD);
    uint32_t off = 0;
    off = K8A5Align32(off + srcK);
    off = K8A5Align32(off + srcV);
    off = K8A5Align32(off + fat);
    off = K8A5Align32(off + scal);
    off = K8A5Align32(off + keyCode);
    off = K8A5Align32(off + valEnc);
    return off;
}

struct K8A5UbOffsets {
    uint32_t offSrcK;
    uint32_t offSrcV;
    uint32_t offFat;
    uint32_t offScal;
    uint32_t offKeyCode;
    uint32_t offValEnc;
    uint32_t size;
};

__aicore__ inline K8A5UbOffsets K8A5MakeOffsets(uint32_t packD)
{
    K8A5UbOffsets o {};
    const uint32_t tileRows = K8_A5_TILE_ROWS;
    const uint32_t srcK = tileRows * packD * 2U;
    const uint32_t srcV = tileRows * packD * 2U;
    const uint32_t fat = tileRows * packD * 2U;
    const uint32_t scal = K8_A5_SCALAR_BUF_BYTES;
    const uint32_t keyCode = tileRows * packD;
    const uint32_t valEnc = K8A5ValEncodedBytes(packD);
    o.offSrcK = 0;
    o.offSrcV = K8A5Align32(o.offSrcK + srcK);
    o.offFat = K8A5Align32(o.offSrcV + srcV);
    o.offScal = K8A5Align32(o.offFat + fat);
    o.offKeyCode = K8A5Align32(o.offScal + scal);
    o.offValEnc = K8A5Align32(o.offKeyCode + keyCode);
    o.size = K8A5Align32(o.offValEnc + valEnc);
    return o;
}

}  // namespace arch35
}  // namespace rq_uniform

#if defined(K8A5_LOCAL_AICORE_FALLBACK)
#undef __aicore__
#undef K8A5_LOCAL_AICORE_FALLBACK
#endif
