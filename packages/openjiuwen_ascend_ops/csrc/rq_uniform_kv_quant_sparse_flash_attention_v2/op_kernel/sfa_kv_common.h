#ifndef SFA_KV_COMMON_H
#define SFA_KV_COMMON_H

#include <cstdint>

namespace sfa_kv {

constexpr uint32_t NOPE_DIM = 512;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t CODE_BYTES = NOPE_DIM / 2;  // 256
constexpr uint32_t ROPE_BYTES = ROPE_DIM * sizeof(uint16_t);  // 128
constexpr uint32_t META_BYTES = 4;
constexpr uint32_t SLOT_BYTES = CODE_BYTES + ROPE_BYTES + META_BYTES;  // 388
constexpr uint32_t UB_ROW_ALIGN = 416;
constexpr uint32_t CODE_OFF = 0;
constexpr uint32_t ROPE_OFF = CODE_BYTES;
constexpr uint32_t SCALE_OFF = CODE_BYTES + ROPE_BYTES;  // 384
constexpr uint32_t OFFSET_OFF = SCALE_OFF + 2;
constexpr uint32_t DEQUANT_COMBINE_DIM = NOPE_DIM + ROPE_DIM;  // 576

static_assert(SLOT_BYTES == 388, "SFA fused slot must be 388 bytes");

}  // namespace sfa_kv

#endif  // SFA_KV_COMMON_H
