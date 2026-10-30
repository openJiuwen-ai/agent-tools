/**
 * Schema A attention V2 (doc 11).
 *
 * P0: vector dequant of pack-V2 signed int4. No GetValue, no per-row PIPE_ALL.
 *     Cast(int4) → Adds(+8) → Brcb(scale/offset) → Mul → Add.
 *     fp16 stays in half. bf16 finishes with fp32 CAST_RINT.
 * P1: CopyOutSfaV2 chunks 8 rows, double-buffers nope/rope, and overlaps
 *     V(chunk i+1) with MTE3(chunk i). Rope is packed into a contiguous UB
 *     tile and written with srcStride 0.
 * P2: only QSFAT::dequantV2 (the V2 op) calls these. Legacy DequantSfaRows stays.
 * P3: cube MM, softmax, and the TQ4 LUT path are not modified here.
 */
#ifndef RQ_UNIFORM_KV_QUANT_SPARSE_FLASH_ATTENTION_DEQUANT_V2_H
#define RQ_UNIFORM_KV_QUANT_SPARSE_FLASH_ATTENTION_DEQUANT_V2_H

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DequantSfaRowsV2(LocalTensor<KV_T> &srcTensor,
                                                                  LocalTensor<K_ROPE_T> &dstNope,
                                                                  LocalTensor<K_ROPE_T> &dstRope, int32_t row0,
                                                                  int32_t n)
{
    constexpr uint32_t kHd = sfa_kv::NOPE_DIM;
    constexpr uint32_t kCodeBytes = sfa_kv::CODE_BYTES;
    constexpr uint32_t kRopeBytes = sfa_kv::ROPE_BYTES;
    constexpr int32_t kChunk = 8;
    constexpr uint32_t kHalfMask = 128;
    constexpr uint32_t kChunks = kHd / kHalfMask;
    constexpr uint32_t kRowBlocks = kHd / 16;
    constexpr uint32_t kCompactOff = 0;
    constexpr uint32_t kMetaOff = 2048;
    constexpr uint32_t kIdxScaleOff = 2304;
    constexpr uint32_t kIdxOffsetOff = 2336;
    constexpr uint32_t kScaleOff = 2368;
    constexpr uint32_t kOffsetOff = 2400;
    constexpr uint32_t kScaleBlkOff = 2432;
    constexpr uint32_t kOffsetBlkOff = 2688;
    constexpr uint32_t kHalfOff = 2944;
    constexpr uint32_t kFloatOff = 11136;
    static_assert(kFloatOff + kChunk * kHd * sizeof(float) <= ConstInfo::BUFFER_SIZE_BYTE_32K,
                  "V2 dequant scratch exceeds inputBuff2");

    if (unlikely(n <= 0)) {
        return;
    }

    uint32_t rowBytes =
        QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput), static_cast<uint32_t>(BYTE_BLOCK));
    LocalTensor<uint8_t> srcU8 = srcTensor.template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> scratch = inputBuff2.Get<uint8_t>();
    LocalTensor<uint8_t> compact = scratch[kCompactOff];
    LocalTensor<half> metaBlk = scratch[kMetaOff].template ReinterpretCast<half>();
    LocalTensor<int32_t> idxScale = scratch[kIdxScaleOff].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> idxOffset = scratch[kIdxOffsetOff].template ReinterpretCast<int32_t>();
    LocalTensor<half> scaleVec = scratch[kScaleOff].template ReinterpretCast<half>();
    LocalTensor<half> offsetVec = scratch[kOffsetOff].template ReinterpretCast<half>();
    LocalTensor<half> scaleBlk = scratch[kScaleBlkOff].template ReinterpretCast<half>();
    LocalTensor<half> offsetBlk = scratch[kOffsetBlkOff].template ReinterpretCast<half>();
    LocalTensor<half> halfSrc = scratch[kHalfOff].template ReinterpretCast<half>();
    LocalTensor<float> workF = scratch[kFloatOff].template ReinterpretCast<float>();

    uint32_t srcBase = static_cast<uint32_t>(row0) * rowBytes;
    DataCopyParams codeParams;
    codeParams.blockCount = static_cast<uint16_t>(n);
    codeParams.blockLen = static_cast<uint16_t>(kCodeBytes / BYTE_BLOCK);
    codeParams.srcStride = static_cast<uint16_t>((rowBytes - kCodeBytes) / BYTE_BLOCK);
    codeParams.dstStride = 0;
    DataCopy(compact, srcU8[srcBase], codeParams);

    // Each row's scale/offset sits on the 32B block at byte 384. Next row is 13 blocks away.
    DataCopyParams metaParams;
    metaParams.blockCount = static_cast<uint16_t>(n);
    metaParams.blockLen = 1;
    metaParams.srcStride = static_cast<uint16_t>(rowBytes / BYTE_BLOCK - 1);
    metaParams.dstStride = 0;
    DataCopy(metaBlk.template ReinterpretCast<uint8_t>(), srcU8[srcBase + sfa_kv::SCALE_OFF], metaParams);
    if (n < kChunk) {
        PipeBarrier<PIPE_V>();
        Duplicate(metaBlk[static_cast<uint32_t>(n) * 16], static_cast<half>(0.0f),
                  static_cast<uint32_t>(kChunk - n) * 16);
    }

    DataCopyParams ropeParams;
    ropeParams.blockCount = static_cast<uint16_t>(n);
    ropeParams.blockLen = static_cast<uint16_t>(kRopeBytes / BYTE_BLOCK);
    ropeParams.srcStride = static_cast<uint16_t>((rowBytes - kRopeBytes) / BYTE_BLOCK);
    ropeParams.dstStride = 0;
    DataCopy(dstRope.template ReinterpretCast<uint8_t>(), srcU8[srcBase + sfa_kv::ROPE_OFF], ropeParams);
    PipeBarrier<PIPE_V>();

    Duplicate(scaleVec, static_cast<half>(0.0f), 16);
    Duplicate(offsetVec, static_cast<half>(0.0f), 16);
    // Byte offsets of scale (0, 32, ...) and offset (2, 34, ...) inside the gathered blocks.
    ArithProgression<int32_t>(idxScale, 0, 32, kChunk);
    ArithProgression<int32_t>(idxOffset, 2, 32, kChunk);
    PipeBarrier<PIPE_V>();

    Gather(scaleVec, metaBlk, idxScale.template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(kChunk));
    Gather(offsetVec, metaBlk, idxOffset.template ReinterpretCast<uint32_t>(), 0, static_cast<uint32_t>(kChunk));
    PipeBarrier<PIPE_V>();

    const uint32_t nElems = static_cast<uint32_t>(n) * kHd;
    auto quantI4 = compact.template ReinterpretCast<int4b_t>();
    Cast(halfSrc, quantI4, RoundMode::CAST_NONE, nElems);
    PipeBarrier<PIPE_V>();
    Adds(halfSrc, halfSrc, static_cast<half>(8.0f), nElems);
    PipeBarrier<PIPE_V>();

    Brcb(scaleBlk, scaleVec, 1, BrcbRepeatParams(1, 8));
    Brcb(offsetBlk, offsetVec, 1, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();

    BinaryRepeatParams rp(1, 1, 0, static_cast<uint8_t>(kRowBlocks), static_cast<uint8_t>(kRowBlocks), 1);
    for (uint32_t c = 0; c < kChunks; ++c) {
        Mul<half, false>(halfSrc[c * kHalfMask], halfSrc[c * kHalfMask], scaleBlk, kHalfMask,
                         static_cast<uint8_t>(n), rp);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t c = 0; c < kChunks; ++c) {
        Add<half, false>(halfSrc[c * kHalfMask], halfSrc[c * kHalfMask], offsetBlk, kHalfMask,
                         static_cast<uint8_t>(n), rp);
    }
    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) {
        Cast(workF, halfSrc, RoundMode::CAST_NONE, nElems);
        PipeBarrier<PIPE_V>();
        Cast(dstNope, workF, RoundMode::CAST_RINT, nElems);
    } else {
        DataCopy(dstNope, halfSrc, nElems);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyOutSfaV2(LocalTensor<KV_T> &srcTensor, int32_t dealRow,
                                                              int64_t s2GmStartOffset, int64_t mte3Size,
                                                              const RunInfo &kvVecRunInfo)
{
    constexpr int32_t kChunk = 8;
    constexpr uint32_t kNope = sfa_kv::NOPE_DIM;
    constexpr uint32_t kRope = sfa_kv::ROPE_DIM;
    LocalTensor<K_ROPE_T> nopeUb = tmpBuff1.Get<K_ROPE_T>();
    LocalTensor<K_ROPE_T> ropeUb = tmpBuff2.Get<K_ROPE_T>();
    constexpr uint32_t nopeStride = static_cast<uint32_t>(kChunk) * kNope;
    constexpr uint32_t ropeStride = static_cast<uint32_t>(kChunk) * kRope;
    const uint64_t evts[2] = {SYNC_OUTPUT_BUF1_FLAG, SYNC_OUTPUT_BUF2_FLAG};

    uint32_t mergeGmStride = 512U * kvVecConstInfo.combineHeadDim;
    uint64_t gmBase = kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride +
                      static_cast<uint64_t>(s2GmStartOffset + mte3Size) * kvVecConstInfo.combineHeadDim;
    uint32_t nopeBytes = kvVecConstInfo.headDim * sizeof(K_ROPE_T);
    uint32_t ropeBytes = kvVecConstInfo.headDimRope * sizeof(K_ROPE_T);
    uint32_t nopeDstStride = (kvVecConstInfo.combineHeadDim - kvVecConstInfo.headDim) * sizeof(K_ROPE_T);
    uint32_t ropeDstStride =
        (kvVecConstInfo.combineHeadDim - kvVecConstInfo.headDimRope) * sizeof(K_ROPE_T);

    for (int32_t base = 0; base < dealRow; base += kChunk) {
        int32_t n = dealRow - base < kChunk ? dealRow - base : kChunk;
        uint32_t buf = static_cast<uint32_t>(base / kChunk) & 1U;
        WaitFlag<AscendC::HardEvent::MTE3_V>(evts[buf]);
        LocalTensor<K_ROPE_T> nopeTile = nopeUb[buf * nopeStride];
        LocalTensor<K_ROPE_T> ropeTile = ropeUb[buf * ropeStride];
        DequantSfaRowsV2(srcTensor, nopeTile, ropeTile, base, n);
        SetFlag<AscendC::HardEvent::V_MTE3>(evts[buf]);
        WaitFlag<AscendC::HardEvent::V_MTE3>(evts[buf]);

        DataCopyExtParams nopeParams;
        nopeParams.blockCount = static_cast<uint16_t>(n);
        nopeParams.blockLen = nopeBytes;
        nopeParams.srcStride = 0;
        nopeParams.dstStride = nopeDstStride;
        DataCopyPad(kvMergeGm_[gmBase + static_cast<uint64_t>(base) * kvVecConstInfo.combineHeadDim],
                    nopeTile, nopeParams);

        DataCopyExtParams ropeGmParams;
        ropeGmParams.blockCount = static_cast<uint16_t>(n);
        ropeGmParams.blockLen = ropeBytes;
        ropeGmParams.srcStride = 0;
        ropeGmParams.dstStride = ropeDstStride;
        DataCopyPad(kvMergeGm_[gmBase + static_cast<uint64_t>(base) * kvVecConstInfo.combineHeadDim +
                               kvVecConstInfo.headDim],
                    ropeTile, ropeGmParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(evts[buf]);
    }
}

#endif
