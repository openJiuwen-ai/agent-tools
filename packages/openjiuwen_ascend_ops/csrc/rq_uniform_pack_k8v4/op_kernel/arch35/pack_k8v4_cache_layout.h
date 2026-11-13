/*!
 * \file pack_k8v4_cache_layout.h
 * \brief A5 mode1 (TQ_K8_FP8) GM SoA offsets — Key row = D bytes, Value U4+meta.
 */
#pragma once

#include "../rq_uniform_pack_k8v4_common.h"

namespace rq_uniform {
namespace arch35 {

__aicore__ inline uint64_t K8A5KeyHeadBase(
    uint32_t blockIdx, uint32_t headIdx, uint32_t numHeads, uint32_t blockSize, uint32_t packD)
{
    const uint64_t headStride = static_cast<uint64_t>(blockSize) * packD;
    return (static_cast<uint64_t>(blockIdx) * numHeads + headIdx) * headStride;
}

__aicore__ inline uint64_t K8A5ValHeadBase(
    uint32_t blockIdx, uint32_t headIdx, uint32_t numHeads, uint32_t blockSize, uint32_t packD)
{
    const uint64_t headStride =
        static_cast<uint64_t>(blockSize) * TqValRowBytes(packD);
    return (static_cast<uint64_t>(blockIdx) * numHeads + headIdx) * headStride;
}

__aicore__ inline uint64_t K8A5KeyCodeOff(uint64_t headBase, uint32_t pos, uint32_t packD)
{
    return headBase + static_cast<uint64_t>(pos) * packD;
}

__aicore__ inline uint64_t K8A5ValCodeOff(uint64_t headBase, uint32_t pos, uint32_t packD)
{
    return headBase + static_cast<uint64_t>(pos) * (packD / 2U);
}

__aicore__ inline uint64_t K8A5ValVminOff(
    uint64_t headBase, uint32_t blockSize, uint32_t packD, uint32_t pos)
{
    return headBase + static_cast<uint64_t>(blockSize) * (packD / 2U) +
           static_cast<uint64_t>(pos) * TQ_ROW_META_BYTES;
}

__aicore__ inline uint64_t K8A5ValVstepOff(
    uint64_t headBase, uint32_t blockSize, uint32_t packD, uint32_t pos)
{
    return headBase + static_cast<uint64_t>(blockSize) * (packD / 2U + TQ_ROW_META_BYTES) +
           static_cast<uint64_t>(pos) * TQ_ROW_META_BYTES;
}

}  // namespace arch35
}  // namespace rq_uniform
