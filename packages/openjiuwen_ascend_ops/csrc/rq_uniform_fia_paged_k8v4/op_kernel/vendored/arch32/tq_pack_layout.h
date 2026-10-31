/**
 * RqUniform K8V4 pack cache layout (must match rq_uniform_pack_k8v4/IMPL_DESIGN.md).
 *
 * D=128 and D=256 share the same SoA: [codes | meta0 | meta1] per head.
 * Cube Haar tiles stay 128×128 (L0B 32 KiB); D=256 is a 2×2 of those tiles.
 */
#ifndef BR_PACK_LAYOUT_H
#define BR_PACK_LAYOUT_H

namespace tq_pack {

static constexpr uint32_t BR_CUBE_TILE_D = 128U;
static constexpr uint32_t BR_HEAD_SIZE = 128U;  // D=128 compile-time default
static constexpr uint32_t BR_HEAD_SIZE_256 = 256U;
static constexpr uint32_t BR_BLOCK_ROWS = 16U;
static constexpr uint32_t BR_KEY_CODE_BYTES = BR_HEAD_SIZE;
static constexpr uint32_t BR_VAL_CODE_BYTES = BR_HEAD_SIZE / 2U;
static constexpr uint32_t BR_META_BYTES = 4U;  // base+step or vmin+vstep per row
static constexpr uint32_t BR_KEY_ROW_BYTES = BR_KEY_CODE_BYTES + BR_META_BYTES;
static constexpr uint32_t BR_VAL_ROW_BYTES = BR_VAL_CODE_BYTES + BR_META_BYTES;
static constexpr uint32_t BR_KEY_TILE_BYTES = BR_BLOCK_ROWS * BR_KEY_ROW_BYTES;   // 2112 (D=128)
static constexpr uint32_t BR_VAL_TILE_BYTES = BR_BLOCK_ROWS * BR_VAL_ROW_BYTES;   // 1088 (D=128)
// D=256 tiles: 16*(256+4)=4160 key, 16*(128+4)=2112 value.

__aicore__ inline uint32_t TqKeyCodeBytes(uint32_t headSize)
{
    return headSize;
}

__aicore__ inline uint32_t TqValCodeBytes(uint32_t headSize)
{
    return headSize / 2U;
}

__aicore__ inline uint32_t TqKeyRowBytes(uint32_t headSize)
{
    return headSize + BR_META_BYTES;
}

__aicore__ inline uint32_t TqValRowBytes(uint32_t headSize)
{
    return headSize / 2U + BR_META_BYTES;
}

__aicore__ inline uint32_t TqKeyTileBytesForHead(uint32_t headSize)
{
    return BR_BLOCK_ROWS * TqKeyRowBytes(headSize);
}

__aicore__ inline uint32_t TqValTileBytesForHead(uint32_t headSize)
{
    return BR_BLOCK_ROWS * TqValRowBytes(headSize);
}

__aicore__ inline uint32_t TqKeyHeadStride(uint32_t blockSize, uint32_t headSize = BR_HEAD_SIZE)
{
    return blockSize * TqKeyRowBytes(headSize);
}

__aicore__ inline uint32_t TqValHeadStride(uint32_t blockSize, uint32_t headSize = BR_HEAD_SIZE)
{
    return blockSize * TqValRowBytes(headSize);
}

__aicore__ inline uint64_t TqKeyHeadBase(uint32_t physBlock, uint32_t kvHead, uint32_t numKvHeads,
                                         uint32_t blockSize, uint32_t headSize = BR_HEAD_SIZE)
{
    return (static_cast<uint64_t>(physBlock) * numKvHeads + kvHead) *
           static_cast<uint64_t>(TqKeyHeadStride(blockSize, headSize));
}

__aicore__ inline uint64_t TqValHeadBase(uint32_t physBlock, uint32_t kvHead, uint32_t numKvHeads,
                                         uint32_t blockSize, uint32_t headSize = BR_HEAD_SIZE)
{
    return (static_cast<uint64_t>(physBlock) * numKvHeads + kvHead) *
           static_cast<uint64_t>(TqValHeadStride(blockSize, headSize));
}

__aicore__ inline uint64_t TqKeyCodeOffset(uint64_t headBase, uint32_t posInBlock,
                                           uint32_t headSize = BR_HEAD_SIZE)
{
    return headBase + static_cast<uint64_t>(posInBlock) * TqKeyCodeBytes(headSize);
}

__aicore__ inline uint64_t TqKeyBaseOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock,
                                           uint32_t headSize = BR_HEAD_SIZE)
{
    return headBase + static_cast<uint64_t>(blockSize) * TqKeyCodeBytes(headSize) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t TqKeyStepOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock,
                                           uint32_t headSize = BR_HEAD_SIZE)
{
    return headBase + static_cast<uint64_t>(blockSize) * (TqKeyCodeBytes(headSize) + 2U) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t TqValCodeOffset(uint64_t headBase, uint32_t posInBlock,
                                           uint32_t headSize = BR_HEAD_SIZE)
{
    return headBase + static_cast<uint64_t>(posInBlock) * TqValCodeBytes(headSize);
}

__aicore__ inline uint64_t TqValVminOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock,
                                           uint32_t headSize = BR_HEAD_SIZE)
{
    return headBase + static_cast<uint64_t>(blockSize) * TqValCodeBytes(headSize) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

__aicore__ inline uint64_t TqValVstepOffset(uint64_t headBase, uint32_t blockSize, uint32_t posInBlock,
                                            uint32_t headSize = BR_HEAD_SIZE)
{
    return headBase + static_cast<uint64_t>(blockSize) * (TqValCodeBytes(headSize) + 2U) +
           static_cast<uint64_t>(posInBlock) * 2U;
}

}  // namespace tq_pack

#endif
