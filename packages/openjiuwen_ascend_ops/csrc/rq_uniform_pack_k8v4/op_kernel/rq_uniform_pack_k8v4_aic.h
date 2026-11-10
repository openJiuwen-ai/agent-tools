#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "catlass/arch/resource.hpp"
#include "rq_uniform_pack_k8v4_common.h"

namespace rq_uniform {

using namespace AscendC;

// ── 950 (dav_3510) LoadData port ─────────────────────────────────────────
// CANN 9.1's dav_3510 shim reinterprets v1 LoadData2DParams fields
// (repeatTimes→kStep, srcStride→mStep and srcStride), which does not
// reproduce the tile placement this manual Mmad pipeline expects. On 3510
// drive the loads through LoadData2DParamsV2, with fields following the
// 3510 matmul-stage recipes (load_to_l0a/b_load2dV2.h):
//   V2 units: mStep = 16 elements, kStep = 32B, src/dstStride = 512B.
//   Fractal addressing: src(m_i,k_j) = k_j*srcStride*512B +
//   m_i*512B, dst likewise with dstStride -- identical for transpose and
//   non-transpose (transpose permutes inside each 16x16 fractal only). So
//   the m direction is always the contiguous 512B direction and srcStride /
//   dstStride are the K-direction (column) pitches; a dense tile uses
//   srcStride = dstStride = mStep.
//   B (transposed): mStep = K fractals, kStep = N/16, src = dst = 8; L1
//     staged k-outer/m-inner (k-fractal groups of mStep m-fractals).
//   A (non-transposed): mStep = 2, kStep = K/16 = 8, src = dst = 2 (= mStep,
//     dense); L1 must be staged k-outer/m-interleaved: head h's k-fractal j
//     at 32B-block 32j + 16h (Nd2Nz in LoadRawTensorToL1, dstNzC0Stride = 32
//     blocks -- same contract as the B rotation staging). The 910b
//     m-outer L1 ([m0 4KB][m1 4KB]) is unspecifiable in V2 terms: any
//     srcStride mis-calens either m1 (srcStride=1 fingerprint) or m0.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define TQ_PACK_LOADDATA_V2 1
#else
#define TQ_PACK_LOADDATA_V2 0
#endif

template <typename T>
class RqUniformPackK8v4CubeService {
public:
    __aicore__ inline explicit RqUniformPackK8v4CubeService(
        RqUniformPackK8v4Context<T>& context)
        : context_(context) {}

    __aicore__ inline void Init() {
        eventListMte1M = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
        eventListMmte1 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
        eventListMte1Mte2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
        eventListMte2Mte1 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
    }

    __aicore__ inline void Finite() {
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_M>(eventListMte1M);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
    }

    __aicore__ inline void Process(uint32_t manualGroupId) {
        auto& op = context_;
        AscendC::GlobalTensor<half> cWorkGm;
        op.InitManualWorkspaceTensors(manualGroupId, cWorkGm);
        Init();

        TqManualMmadResource manualResource;

        // 950 entry handshake: block until the paired AIV subcores have run
        // their launch-entry C_FREE resets (and the pair's mode-2 state is
        // engaged), so the compute loop's first C_FREE wait can never race
        // ahead of them.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        TqCrossCoreWaitForBothAiv<PIPE_FIX>(TQ_MANUAL_SYNC_ENTRY_READY);
#endif
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        if (op.CubeNTiles() == 1U) {
            LoadResidentRotation(manualResource, op.rotationTGm_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        if (op.CubeNTiles() == 1U) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        }

        uint32_t manualTileOrdinal = 0;
        const uint32_t subBlocksPerBlock = op.blockSize_ / TQ_BLOCK_ROWS;
        const uint32_t headTileCount =
            (op.numHeads_ + TQ_MANUAL_HEADS_PER_TILE - 1) /
            TQ_MANUAL_HEADS_PER_TILE;
        for (uint32_t reqIdx = 0; reqIdx < op.numReqs_; ++reqIdx) {
            uint32_t seqStart = 0;
            uint32_t seqEnd = 0;
            if (!op.ReadRequestTokenRange(reqIdx, seqStart, seqEnd)) {
                return;
            }
            if (seqStart == seqEnd) {
                continue;
            }
            const uint32_t rowCount = seqEnd - seqStart;
            for (uint32_t rowOff = 0; rowOff < rowCount;) {
                const uint32_t tokenStart = seqStart + rowOff;
                uint32_t slot = 0;
                uint32_t startGroupRow = 0;
                uint32_t validRows = 0;
                if (!op.ResolvePackChunk(tokenStart, rowCount - rowOff, slot,
                                         startGroupRow, validRows)) {
                    return;
                }
                const uint32_t blockIdx = slot / op.blockSize_;
                const uint32_t blockOffset = slot - blockIdx * op.blockSize_;
                const uint32_t subBlockInBlock = blockOffset / TQ_BLOCK_ROWS;
                const uint32_t valueGroupInBlock = blockOffset / TQ_MANUAL_GROUP_ROWS;
                const bool preserveValue =
                    startGroupRow != 0 || validRows < TQ_MANUAL_GROUP_ROWS;
                for (uint32_t headTileStart = 0;
                     headTileStart < op.numHeads_;
                     headTileStart += TQ_MANUAL_HEADS_PER_TILE) {
                    const uint64_t workOrdinal =
                        (static_cast<uint64_t>(blockIdx) * subBlocksPerBlock +
                         subBlockInBlock) * headTileCount +
                        headTileStart / TQ_MANUAL_HEADS_PER_TILE;
                    if (workOrdinal % op.dataCores_ == manualGroupId) {
                        const uint32_t keyStream =
                            manualTileOrdinal * TQ_MANUAL_STREAM_KIND_COUNT;
                        ManualKey1StreamDesc keyDesc {
                            tokenStart, slot, blockIdx, valueGroupInBlock,
                            headTileStart, startGroupRow, validRows, keyStream,
                            preserveValue, false};
                        ManualKey1StreamDesc valueDesc {
                            tokenStart, slot, blockIdx, valueGroupInBlock,
                            headTileStart, startGroupRow, validRows, keyStream + 1,
                            preserveValue, true};
                        ComputeStream(manualResource, cWorkGm, keyDesc);
                        ComputeStream(manualResource, cWorkGm, valueDesc);
                        ++manualTileOrdinal;
                    }
                }
                rowOff += validRows;
            }
        }

        // ── Kernel-exit synchronization ───────────────────
        // 1. AIV_DONE handshake: the AIC retires only after both AIV
        //    subcores drained their Y-cache writes, so it can never outrun
        //    the AIVs' kernel-exit state.
        // 2. Event drains: the loop leaves one pending MTE1_MTE2 and one
        //    M_MTE1 signal that no in-loop iteration consumes - drain both
        //    before releasing the event IDs in Finite().
        //
        // Do NOT add a software Wait<FIX_M> drain here: with unitFlag=0b11
        // fixpipes the FIX_M event is not raised on 950, and the exit
        // handshake above is the synchronization that matters.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        TqCrossCoreWaitForBothAiv<PIPE_FIX>(TQ_MANUAL_SYNC_AIV_DONE);
#endif
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        Finite();
    }

private:
    __aicore__ inline void LoadResidentRotation(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& rotationTGm) {
        auto bL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_B_L1_OFFSET);
        AscendC::Nd2NzParams params;
#if TQ_PACK_LOADDATA_V2
        // K-outer staging (single ND pass, C0 stride = K rows) matching the
        // 3510 transposed-B recipe the V2 load below assumes.
        params.ndNum = 1;
        params.nValue = TQ_ROT_K;
        params.dValue = TQ_ROT_N;
        params.srcDValue = TQ_ROT_N;
        params.srcNdMatrixStride = 0;
        params.dstNzC0Stride = TQ_ROT_K;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
#else
        params.ndNum = TQ_ROT_K / TQ_CUBE_M_ALIGN;
        params.nValue = TQ_CUBE_M_ALIGN;
        params.dValue = TQ_ROT_N;
        params.srcDValue = TQ_ROT_N;
        params.srcNdMatrixStride = TQ_CUBE_M_ALIGN * TQ_ROT_N;
        params.dstNzC0Stride = TQ_CUBE_M_ALIGN;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = TQ_CUBE_M_ALIGN * TQ_ROT_N;
#endif
        AscendC::DataCopy(bL1, rotationTGm, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);

        auto rotationBL0 = resource.l0BBuf.template GetBufferByByte<T>(0);
        // Wait until M engine has finished consuming any prior L0B data
        // before MTE1 overwrites it with the resident rotation B matrix.
#if TQ_PACK_LOADDATA_V2
        // Single V2 transposed load: mStep = K/16, kStep = N/16,
        // srcStride = K/16, dstStride = N/16.
        AscendC::LoadData2DParamsV2 bLoad;
        bLoad.mStartPosition = 0;
        bLoad.kStartPosition = 0;
        bLoad.mStep = TQ_ROT_K / TQ_CUBE_M_ALIGN;
        bLoad.kStep = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.srcStride = TQ_ROT_K / TQ_CUBE_M_ALIGN;
        bLoad.dstStride = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.ifTranspose = true;
        bLoad.sid = 0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        AscendC::LoadData(rotationBL0, bL1, bLoad);
#else
        AscendC::LoadData2DParams bLoad;
        bLoad.startIndex = 0;
        bLoad.repeatTimes = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.srcStride = 1;
        bLoad.sid = 0;
        bLoad.dstGap = 0;
        bLoad.ifTranspose = true;
        bLoad.addrMode = 0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        for (uint32_t i = 0; i < TQ_ROT_K / TQ_CUBE_M_ALIGN; ++i) {
            AscendC::LoadData(
                rotationBL0[i * TQ_ROT_N * TQ_CUBE_M_ALIGN],
                bL1[i * TQ_ROT_N * TQ_CUBE_M_ALIGN], bLoad);
        }
#endif
        // MTE1 finished writing L0B.  Signal M that L0B is ready.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M);
    }

    __aicore__ inline void LoadRawTensorToL1(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<T>& xGm,
        const ManualKey1StreamDesc& desc,
        uint64_t storageOffset,
        uint32_t strideToken,
        uint32_t strideHead,
        uint32_t kOff = 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        auto aL1 =
            resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        // 950 k-outer/m-interleaved staging (the V2 LoadData contract, see
        // the header note): head h's k-fractal j lands at 32B-block 32j+16h
        // -- one 32-block k-group per j holding both heads' 16 rows. Same
        // Nd2Nz contract as the B rotation staging (NStride=1 keeps
        // token rows contiguous, C0Stride = full 32-row tile height).
        for (uint32_t aivSlice = 0; aivSlice < TQ_AIV_SUB_BLOCKS; ++aivSlice) {
            const uint32_t firstHead =
                desc.headTileStart + aivSlice * TQ_MANUAL_HEADS_PER_AIV;
            const uint32_t l1SliceOffset =
#if TQ_PACK_LOADDATA_V2
                aivSlice * TQ_MANUAL_GROUP_ROWS * TQ_CUBE_M_ALIGN;
#else
                aivSlice * TQ_MANUAL_AIV_SLICE_ELEMS;
#endif
            for (uint32_t localHead = 0;
                 localHead < TQ_MANUAL_HEADS_PER_AIV;
                 ++localHead) {
                const uint32_t headIdx = firstHead + localHead;
                if (headIdx >= context_.numHeads_) {
                    continue;
                }
                const uint64_t srcOffset =
                    storageOffset +
                    static_cast<uint64_t>(desc.tokenStart) * strideToken +
                    static_cast<uint64_t>(headIdx) * strideHead +
                    kOff;
                const uint32_t dstRow =
                    localHead * TQ_MANUAL_GROUP_ROWS + desc.startGroupRow;
                AscendC::Nd2NzParams params;
                params.ndNum = 1;
                params.nValue = desc.validRows;
                params.dValue = TQ_ROT_K;
                params.srcDValue = strideToken;
#if TQ_PACK_LOADDATA_V2
                params.dstNzC0Stride = TQ_MANUAL_ROT_TILE_M;  // 32 blocks: k-fractal pitch
#else
                params.dstNzC0Stride = TQ_MANUAL_AIV_SLICE_M;
#endif
                params.dstNzNStride = 1;
                params.srcNdMatrixStride = 0;
                params.dstNzMatrixStride = 0;
                AscendC::DataCopy(
                    aL1[l1SliceOffset + dstRow * TQ_CUBE_M_ALIGN],
                    xGm[srcOffset],
                    params);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
    }

    __aicore__ inline void LoadRawInputToL1(
        TqManualMmadResource& resource,
        const ManualKey1StreamDesc& desc,
        uint32_t kOff = 0U) {
        if (desc.isValue) {
            LoadRawTensorToL1(
                resource,
                context_.valueGm_,
                desc,
                context_.valueStorageOffset_,
                context_.valueStrideToken_,
                context_.valueStrideHead_,
                kOff);
        } else {
            LoadRawTensorToL1(
                resource,
                context_.keyGm_,
                desc,
                context_.keyStorageOffset_,
                context_.keyStrideToken_,
                context_.keyStrideHead_,
                kOff);
        }
    }

    // ── Rotation-only: A × RotB → cWorkGm (merged m=32) ──────────────
    //
    // Single Mmad(m=32, n=128, k=128) + Fixpipe replaces the previous
    // two-slice loop (2 × Mmad(m=16)).  The L1 layout already stores
    // both 16-row NZ slices contiguously (C0 block 0 at offset 0, C0
    // block 1 at offset 2048 elements), so one LoadData of 16 fractals
    // (2 M-direction × 8 K-direction) loads the full 32×128 tile into
    // L0A.  The workspace stride is 32×PackD() (tight; no D_MAX padding).
    //
    // Per-stream event sequence (unitFlag=0b11):
    //
    //   Wait<M_MTE1>  — M done with prev L0A; MTE1 can write
    //   LoadData L1→L0A[0]  (repeatTimes=16, 2×8 fractals)
    //   Set<MTE1_M>   — L0A ready for M
    //   Wait<MTE1_M>  — L0A ready
    //   Mmad L0A×RotB→L0C_rot  (m=32, unitFlag=0b11 → M_FIX)
    //   Fixpipe L0C_rot→GM      (mSize=32, unitFlag=0b11 → FIX_M)
    //   Set<M_MTE1>   — M freed L0A (MMAD done)

    __aicore__ inline void LoadRotationTile(
        TqManualMmadResource& resource,
        uint32_t kOff,
        uint32_t nOff) {
        const uint32_t packD = context_.PackD();
        auto bL1 = resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_B_L1_OFFSET);
        AscendC::Nd2NzParams params;
#if TQ_PACK_LOADDATA_V2
        // K-outer staging, same contract as LoadResidentRotation (GM row
        // stride stays packD for the D=256 sub-tile read).
        params.ndNum = 1;
        params.nValue = TQ_CUBE_TILE_D;
        params.dValue = TQ_ROT_N;
        params.srcDValue = packD;
        params.srcNdMatrixStride = 0;
        params.dstNzC0Stride = TQ_CUBE_TILE_D;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
#else
        params.ndNum = TQ_CUBE_TILE_D / TQ_CUBE_M_ALIGN;
        params.nValue = TQ_CUBE_M_ALIGN;
        params.dValue = TQ_CUBE_TILE_D;
        params.srcDValue = packD;
        params.srcNdMatrixStride = TQ_CUBE_M_ALIGN * packD;
        params.dstNzC0Stride = TQ_CUBE_M_ALIGN;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = TQ_CUBE_M_ALIGN * TQ_CUBE_TILE_D;
#endif
        AscendC::DataCopy(bL1, context_.rotationTGm_[kOff * packD + nOff], params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);

        auto rotationBL0 = resource.l0BBuf.template GetBufferByByte<T>(0);
#if TQ_PACK_LOADDATA_V2
        // Same transposed geometry as LoadResidentRotation with
        // with TQ_CUBE_TILE_D in place of TQ_ROT_K.
        AscendC::LoadData2DParamsV2 bLoad;
        bLoad.mStartPosition = 0;
        bLoad.kStartPosition = 0;
        bLoad.mStep = TQ_CUBE_TILE_D / TQ_CUBE_M_ALIGN;
        bLoad.kStep = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.srcStride = TQ_CUBE_TILE_D / TQ_CUBE_M_ALIGN;
        bLoad.dstStride = TQ_ROT_N / TQ_CUBE_M_ALIGN;
        bLoad.ifTranspose = true;
        bLoad.sid = 0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        AscendC::LoadData(rotationBL0, bL1, bLoad);
#else
        AscendC::LoadData2DParams bLoad;
        bLoad.startIndex = 0;
        bLoad.repeatTimes = TQ_CUBE_TILE_D / TQ_CUBE_M_ALIGN;
        bLoad.srcStride = 1;
        bLoad.sid = 0;
        bLoad.dstGap = 0;
        bLoad.ifTranspose = true;
        bLoad.addrMode = 0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        for (uint32_t i = 0; i < TQ_CUBE_TILE_D / TQ_CUBE_M_ALIGN; ++i) {
            AscendC::LoadData(
                rotationBL0[i * TQ_CUBE_TILE_D * TQ_CUBE_M_ALIGN],
                bL1[i * TQ_CUBE_TILE_D * TQ_CUBE_M_ALIGN], bLoad);
        }
#endif
        // L0B ready. L0A still free; ComputeSlice loads A then mmads.
    }

    __aicore__ inline void ComputeSlice(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<half>& cWorkGm,
        uint32_t streamOrdinal,
        uint32_t cOffset,
        uint32_t nOff = 0U,
        bool initC = true,
        bool doFixpipe = true,
        bool waitMte2 = true) {
        if (waitMte2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(eventListMte2Mte1);
        }
        auto aL1 =
            resource.l1Buf.template GetBufferByByte<T>(TQ_MANUAL_ROT_A_L1_OFFSET);
        auto aL0 = resource.l0ABuf.template GetBufferByByte<T>(0);
        auto bL0_rot = resource.l0BBuf.template GetBufferByByte<T>(0);
        auto cL0_rot = resource.l0CBuf.template GetBufferByByte<float>(0);

        // K-split: partial tiles use unitFlag=2 (MMAD→MMAD). The last K tile
        // uses 3 (MMAD→Fixpipe). unitFlag=3 without Fixpipe leaves M_FIX
        // pending and deadlocks D=256 2×2 (same rule as FIA MM2).
        const uint8_t unitFlag = doFixpipe ? static_cast<uint8_t>(0b11)
                                           : static_cast<uint8_t>(0b10);

        if (waitMte2) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
        }

#if TQ_PACK_LOADDATA_V2
        // A tile is 32(M)×128(K) over the k-outer/m-interleaved L1 staging
        // (LoadRawTensorToL1): head h's k-fractal j at 32B-block 32j+16h.
        // V2 fractal addressing (header note): src(m_i,k_j) =
        // k_j*srcStride*512B + m_i*512B and dst likewise, so the dense tile
        // takes srcStride = dstStride = mStep = 2.
        AscendC::LoadData2DParamsV2 aLoad;
        aLoad.mStartPosition = 0;
        aLoad.kStartPosition = 0;
        aLoad.mStep = TQ_MANUAL_ROT_TILE_M / TQ_CUBE_M_ALIGN;
        aLoad.kStep = TQ_ROT_K / TQ_CUBE_M_ALIGN;
        aLoad.srcStride = TQ_MANUAL_ROT_TILE_M / TQ_CUBE_M_ALIGN;
        aLoad.dstStride = TQ_MANUAL_ROT_TILE_M / TQ_CUBE_M_ALIGN;
        aLoad.ifTranspose = false;
        aLoad.sid = 0;
        AscendC::LoadData(aL0, aL1[0], aLoad);
#else
        AscendC::LoadData2DParams aLoad;
        aLoad.startIndex = 0;
        aLoad.repeatTimes =
            (TQ_MANUAL_ROT_TILE_M / TQ_CUBE_M_ALIGN) *
            (TQ_ROT_K / TQ_CUBE_M_ALIGN);
        aLoad.srcStride = 1;
        aLoad.sid = 0;
        aLoad.dstGap = 0;
        aLoad.ifTranspose = false;
        aLoad.addrMode = 0;
        AscendC::LoadData(aL0, aL1[0], aLoad);
#endif
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M);

        AscendC::MmadParams rotMmParams;
        rotMmParams.m = TQ_MANUAL_ROT_TILE_M;
        rotMmParams.n = TQ_ROT_N;
        rotMmParams.k = TQ_ROT_K;
        rotMmParams.cmatrixInitVal = initC;
        rotMmParams.cmatrixSource = false;
        rotMmParams.unitFlag = unitFlag;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventListMte1M);
        AscendC::Mmad(cL0_rot, aL0, bL0_rot, rotMmParams);
        if (!doFixpipe) {
            AscendC::PipeBarrier<PIPE_M>();
        }

        if (doFixpipe) {
            AscendC::FixpipeParamsV220 rotFixParams;
            rotFixParams.nSize = TQ_ROT_N;
            rotFixParams.mSize = TQ_MANUAL_ROT_TILE_M;
            rotFixParams.srcStride = TQ_MANUAL_ROT_TILE_M;
            rotFixParams.dstStride = context_.PackD();
            rotFixParams.ndNum = 1;
            rotFixParams.unitFlag = static_cast<uint8_t>(0b11);
            rotFixParams.quantPre = QuantMode_t::F322F16;
            rotFixParams.reluEn = false;
            AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(
                cWorkGm[cOffset + nOff],
                cL0_rot,
                rotFixParams);
        }

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventListMmte1);
    }

    __aicore__ inline void ComputeStream(
        TqManualMmadResource& resource,
        AscendC::GlobalTensor<half>& cWorkGm,
        const ManualKey1StreamDesc& desc) {
        const uint16_t flagBase = context_.ManualFlagBase(desc.streamOrdinal);
        const uint32_t bufferOffset =
            context_.ManualBufferOffset(desc.streamOrdinal);
        TqCrossCoreWaitForBothAiv<PIPE_FIX>(
            flagBase + TQ_MANUAL_SYNC_C_FREE);
        const uint32_t nTiles = context_.CubeNTiles();
        if (nTiles == 1U) {
            LoadRawInputToL1(resource, desc, 0U);
            ComputeSlice(resource, cWorkGm, desc.streamOrdinal, bufferOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
        } else {
            // D=256: 2×2 of 32×128×128. Rotation cannot stay resident in L0B.
            for (uint32_t nTile = 0U; nTile < nTiles; ++nTile) {
                const uint32_t nOff = nTile * TQ_CUBE_TILE_D;
                for (uint32_t kTile = 0U; kTile < nTiles; ++kTile) {
                    const uint32_t kOff = kTile * TQ_CUBE_TILE_D;
                    LoadRawInputToL1(resource, desc, kOff);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                        eventListMte2Mte1);
                    LoadRotationTile(resource, kOff, nOff);
                    ComputeSlice(
                        resource, cWorkGm, desc.streamOrdinal, bufferOffset,
                        nOff, /*initC=*/(kTile == 0U),
                        /*doFixpipe=*/(kTile + 1U == nTiles),
                        /*waitMte2=*/false);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventListMte1Mte2);
                }
            }
        }
        TqSyncFixed<AscendC::HardEvent::FIX_MTE3>();
        TqCrossCoreSetForBothAiv<PIPE_FIX>(
            flagBase + TQ_MANUAL_SYNC_C_READY);
    }

    RqUniformPackK8v4Context<T>& context_;
    int8_t eventListMte1M;
    int8_t eventListMmte1;
    int8_t eventListMte1Mte2;
    int8_t eventListMte2Mte1;
};

}  // namespace rq_uniform
