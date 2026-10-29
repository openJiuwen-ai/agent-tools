#ifndef RQ_UNIFORM_PACK_SFA_COMMON_H
#define RQ_UNIFORM_PACK_SFA_COMMON_H

#include <cstdint>

namespace rq_uniform_sfa {

constexpr uint32_t NOPE_DIM = 512;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t CODE_BYTES = NOPE_DIM / 2;          // 256
constexpr uint32_t ROPE_BYTES = ROPE_DIM * sizeof(uint16_t);  // 128
constexpr uint32_t META_BYTES = 4;                     // scale + offset fp16
constexpr uint32_t SLOT_BYTES = CODE_BYTES + ROPE_BYTES + META_BYTES;  // 388
constexpr uint32_t UB_ROW_ALIGN = 416;  // CeilAlign(388, 32)
constexpr uint32_t TILE_ROWS = 32;      // V2 batch; 16 aligned with k4v4, 32 fills 910B UB

constexpr uint32_t CODE_OFF = 0;
constexpr uint32_t ROPE_OFF = CODE_BYTES;              // 256
constexpr uint32_t SCALE_OFF = CODE_BYTES + ROPE_BYTES;  // 384
constexpr uint32_t OFFSET_OFF = SCALE_OFF + 2;         // 386

// After MergeKv + DequantSfaRows: nope(512) + rope(64) = 576
constexpr uint32_t DEQUANT_COMBINE_DIM = NOPE_DIM + ROPE_DIM;

static_assert(SLOT_BYTES == 388, "SFA fused slot must be 388 bytes");
static_assert(SLOT_BYTES != 386, "must not reuse TQ4 386-byte protocol");

}  // namespace rq_uniform_sfa

#endif  // RQ_UNIFORM_PACK_SFA_COMMON_H
