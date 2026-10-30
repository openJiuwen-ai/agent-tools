/**
 * RqUniform K4V4 pack cache layout (Key/Value both int4 SoA).
 * Must match rq_uniform_pack_k4v4.
 */
#ifndef BR_PACK_LAYOUT_H
#define BR_PACK_LAYOUT_H

namespace tq_pack {

static constexpr uint32_t BR_HEAD_SIZE = 128U;
static constexpr uint32_t BR_BLOCK_ROWS = 16U;
static constexpr uint32_t BR_KEY_CODE_BYTES = BR_HEAD_SIZE / 2U;  // int4
static constexpr uint32_t BR_VAL_CODE_BYTES = BR_HEAD_SIZE / 2U;
static constexpr uint32_t BR_META_BYTES = 4U;  // base+step or vmin+vstep per row
static constexpr uint32_t BR_KEY_ROW_BYTES = BR_KEY_CODE_BYTES + BR_META_BYTES;
static constexpr uint32_t BR_VAL_ROW_BYTES = BR_VAL_CODE_BYTES + BR_META_BYTES;
static constexpr uint32_t BR_KEY_TILE_BYTES = BR_BLOCK_ROWS * BR_KEY_ROW_BYTES;   // 1088
static constexpr uint32_t BR_VAL_TILE_BYTES = BR_BLOCK_ROWS * BR_VAL_ROW_BYTES;   // 1088

__aicore__ inline uint32_t TqKeyHeadStride(uint32_t blockSize)
{
    return blockSize * BR_KEY_ROW_BYTES;
}

__aicore__ inline uint32_t TqValHeadStride(uint32_t blockSize)
{
    return blockSize * BR_VAL_ROW_BYTES;
}

__aicore__ inline uint64_t TqKeyHeadBase(uint32_t physBlock, uint32_t kvHead, uint32_t numKvHeads,
                                         uint32_t blockSize)
{
    return (static_cast<uint64_t>(physBlock) * numKvHeads + kvHead) *
           static_cast<uint64_t>(TqKeyHeadStride(blockSize));
}

__aicore__ inline uint64_t TqValHeadBase(uint32_t physBlock, uint32_t kvHead, uint32_t numKvHeads,
                                         uint32_t blockSize)
{
    return (static_cast<uint64_t>(physBlock) * numKvHeads + kvHead) *
           static_cast<uint64_t>(TqValHeadStride(blockSize));
}

__aicore__ inline uint64_t TqKeyCodeOffset(uint64_t headBase, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(posInBlock) * BR_KEY_CODE_BYTES;
}

__aicore__ inline uint64_t TqKeyBaseOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * BR_KEY_CODE_BYTES +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t TqKeyStepOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * (BR_KEY_CODE_BYTES + 2U) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t TqValCodeOffset(uint64_t headBase, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(posInBlock) * BR_VAL_CODE_BYTES;
}

__aicore__ inline uint64_t TqValVminOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * BR_VAL_CODE_BYTES +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t TqValVstepOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock)
{
    return headBase + static_cast<uint64_t>(blockSize) * (BR_VAL_CODE_BYTES + 2U) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

}  // namespace tq_pack

#endif
