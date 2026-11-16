/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_antiquant_w8.h
 * \brief
 */
#ifndef VF_ANTIQUANT_W8
#define VF_ANTIQUANT_W8

#include "kernel_tensor.h"

namespace FaVectorApi {
// w8转Q_T
static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};
// fp32->Q_T
static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::CAST_RINT};
// fp16 -> bf16
static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_1 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr Reg::CastTrait castTraitFp8_2 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8_4 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8Nz(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubOffsetAddr,
                                     __ubuf__ Q_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vOffset;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    uint32_t rowBaseSize = 8;   // 8行
    uint32_t colBaseSize = 16;  // 16列
    uint32_t dealBaseNum = 128; // 128个元素
    uint32_t colDstStride = dealRowCount * colBaseSize;
    uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) / 32 * 32;                               // 32B对齐
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize); // 8行对齐
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);

    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
        if constexpr (hasOffset) {
            Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_BLK>(vOffset, ubOffsetAddr + colBaseSize * colLoopIdx);
        }
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colBaseSize * colLoopIdx);

        // #pragma unroll(4)
        for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
            uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
            __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + dealBaseNum * rowLoopIdx + colDstStride * colLoopIdx;
            __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseNum * rowLoopIdx;
            ;
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp);
            if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
                Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
                Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
            } else {
                Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
            }
            if constexpr (hasOffset) {
                Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
            }
            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
            Reg::StoreAlign<Q_T, Reg::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8Nz(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize % 16 == 0);
    static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                  "antiquant w8, KV_T must be int8_t or hifloat8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubOffsetAddr = (__ubuf__ Q_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8Nz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr, ubScaleAddr,
                                                                        dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplW8NzD032(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                         __ubuf__ Q_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();

    constexpr uint32_t rowBaseSize = 4;
    constexpr uint32_t colBaseSize = 32;
    constexpr uint32_t dealBaseNum = 128;
    uint32_t colDstStride = dealRowCount * colBaseSize;
    uint32_t colSrcStride = dealRowCount * colBaseSize;
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize);
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);

    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vScale, ubScaleAddr + dealBaseNum * colLoopIdx);

        for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
            uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
            __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx + dealBaseNum * rowLoopIdx;
            __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseNum * rowLoopIdx;

            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp);
            if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
                Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
                Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
            } else {
                Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
            }
            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
            Reg::StoreAlign<Q_T, Reg::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFW8NzD032(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                           LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize % 32 == 0);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant w8 D032, KV_T must be int8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8NzD032<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplFp8Nz(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubScaleAddr,
                                      uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vRes;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    uint32_t rowBaseSize = 8;   // 8行
    uint32_t colBaseSize = 16;  // 16列
    uint32_t dealBaseNum = 128; // 128个元素
    uint32_t colDstStride = colBaseSize * dealRowCount;
    uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) / 32 * 32; // 32B对齐
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + rowBaseSize - 1) / rowBaseSize);
    uint16_t tailRowLoopIdx = rowLoopCnt - 1;
    uint32_t validElements = (dealRowCount - tailRowLoopIdx * rowBaseSize) * colBaseSize;
    Reg::MaskReg storeMask = Reg::UpdateMask<Q_T>(validElements);

    for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
        // 加载 scale
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_BLK>(vScale, ubScaleAddr + colLoopIdx * colBaseSize);

        for (uint16_t rowLoop = 1; rowLoop < rowLoopCnt; rowLoop++) {
            uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
            __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx + dealBaseNum * rowLoopIdx;
            __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseNum * rowLoopIdx;
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B16>((Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp);

            // cast操作, Fp8->Fp32
            Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
            Reg::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);
            // cast操作, Fp32->Fp16/Bf16
            Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
            Reg::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);
            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vCastRes0,
                                                           (Reg::RegTensor<uint16_t> &)vCastRes0,
                                                           (Reg::RegTensor<uint16_t> &)vCastRes1, kvTypeMaskAll);
            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vCastRes0, vScale, qTypeMaskAll);
            // 将输出结果copy到UB
            Reg::StoreAlign<Q_T, Reg::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, qTypeMaskAll);
        }
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + colDstStride * colLoopIdx + dealBaseNum * tailRowLoopIdx;
        __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + colSrcStride * colLoopIdx + dealBaseNum * tailRowLoopIdx;
        Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B16>((Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp);

        // cast操作, Fp8->Fp32
        Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
        Reg::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);
        // cast操作, Fp32->Fp16/Bf16
        Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
        Reg::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);
        Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vCastRes0,
                                                       (Reg::RegTensor<uint16_t> &)vCastRes0,
                                                       (Reg::RegTensor<uint16_t> &)vCastRes1, kvTypeMaskAll);
        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vCastRes0, vScale, qTypeMaskAll);
        // 将输出结果copy到UB
        Reg::StoreAlign<Q_T, Reg::StoreDist::DIST_NORM_B16>(ubDstAddrTmp, vRes, storeMask);
    }
}

template <typename Q_T, typename KV_T, bool hasOffset>
__simd_callee__ static inline void W8NzQuantAndStore(Reg::RegTensor<KV_T> &vKvData, Reg::RegTensor<half> &vCastFp16Res,
                                                     Reg::RegTensor<Q_T> &vRes, __ubuf__ uint8_t *ubSrc,
                                                     Reg::RegTensor<Q_T> &vOffset, Reg::RegTensor<Q_T> &vScale,
                                                     __ubuf__ Q_T *ubDst, Reg::MaskReg &kvTypeMaskAll,
                                                     Reg::MaskReg &qTypeMaskAll)
{
    Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t> &)vKvData, ubSrc);
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
    } else {
        Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
    }
    if constexpr (hasOffset) {
        Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vOffset, vRes, qTypeMaskAll);
    }
    Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
    Reg::StoreAlign<Q_T, Reg::StoreDist::DIST_NORM_B16>(ubDst, vRes, qTypeMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenNz(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                             __ubuf__ Q_T *ubOffsetAddr, __ubuf__ Q_T *ubScaleAddr,
                                             uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vOffsetFirst;
    Reg::RegTensor<Q_T> vOffsetBack;
    Reg::RegTensor<Q_T> vScaleFirst;
    Reg::RegTensor<Q_T> vScaleBack;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    const uint32_t rowBaseSize = 8;        // 8行
    const uint32_t colBaseSize = 16;       // 16列
    const uint32_t dealBaseNum = 128;      // 128个元素
    const uint32_t doubleRowBaseSize = 16; // 每16行交替，防止bank冲突

    const uint32_t rowStride = doubleRowBaseSize * colBaseSize;
    const uint32_t colDstStride = dealRowCount * colBaseSize;
    const uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) >> 5U << 5U; // 32B对齐
    const uint16_t rowLoopCnt =
        static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize); // 16行对齐
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);

    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
        uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
        __ubuf__ Q_T *ubOffsetAddrTmp = ubOffsetAddr + doubleRowBaseSize * rowLoopIdx;
        __ubuf__ Q_T *ubScaleAddrTmp = ubScaleAddr + doubleRowBaseSize * rowLoopIdx;

        if constexpr (hasOffset) {
            Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vOffsetFirst, ubOffsetAddrTmp);
            Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vOffsetBack, rowBaseSize + ubOffsetAddrTmp);
        }
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vScaleFirst, ubScaleAddrTmp);
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vScaleBack, rowBaseSize + ubScaleAddrTmp);
        for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
            __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + rowStride * rowLoopIdx + colSrcStride * colLoopIdx;
            __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + rowStride * rowLoopIdx + colDstStride * colLoopIdx;

            // 后半组
            W8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, ubSrcTemp + dealBaseNum, vOffsetBack,
                                                    vScaleBack, ubDstAddrTmp + dealBaseNum, kvTypeMaskAll,
                                                    qTypeMaskAll);

            // 前半组
            W8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vCastFp16Res, vRes, ubSrcTemp, vOffsetFirst, vScaleFirst,
                                                    ubDstAddrTmp, kvTypeMaskAll, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenNz(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                               LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                               LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize % 16 == 0);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant w4 PerToken, KV_T must be int4_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubOffsetAddr = (__ubuf__ Q_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();
    AntiquantVFImplW8PerTokenNz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr,
                                                                                ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplW8PerTokenNzD032(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                                 __ubuf__ Q_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vScaleLoaded;
    Reg::RegTensor<Q_T> placeHolder;
    Reg::RegTensor<Q_T> vScaleFirst;
    Reg::RegTensor<Q_T> vScaleBack;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();

    const uint32_t colBaseSize = 32;
    const uint32_t dealBaseNum = 128;
    const uint32_t doubleRowBaseSize = 8;

    const uint32_t rowStride = doubleRowBaseSize * colBaseSize;
    const uint32_t colDstStride = dealRowCount * colBaseSize;
    const uint32_t colSrcStride = dealRowCount * colBaseSize;
    const uint16_t rowLoopCnt = static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize);
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);

    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
        uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
        __ubuf__ Q_T *ubScaleAddrTmp = ubScaleAddr + doubleRowBaseSize * rowLoopIdx;
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vScaleLoaded, ubScaleAddrTmp);
        Reg::Interleave<Q_T>(vScaleFirst, vScaleBack, vScaleLoaded, vScaleLoaded);

        for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
            __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + rowStride * rowLoopIdx + colSrcStride * colLoopIdx;
            __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + rowStride * rowLoopIdx + colDstStride * colLoopIdx;

            W8NzQuantAndStore<Q_T, KV_T, false>(vKvData, vCastFp16Res, vRes, ubSrcTemp + dealBaseNum, placeHolder,
                                                vScaleBack, ubDstAddrTmp + dealBaseNum, kvTypeMaskAll, qTypeMaskAll);

            W8NzQuantAndStore<Q_T, KV_T, false>(vKvData, vCastFp16Res, vRes, ubSrcTemp, placeHolder, vScaleFirst,
                                                ubDstAddrTmp, kvTypeMaskAll, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFW8PerTokenNzD032(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                   LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize % 32 == 0);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant w8 PerToken D032, KV_T must be int8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();
    AntiquantVFImplW8PerTokenNzD032<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize>(ubSrcAddr, ubDstAddr, ubScaleAddr,
                                                                         dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFFp8Nz(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                        LocalTensor<Q_T> &antiqScaleFp16Ub, uint32_t dealRowCount)
{
    static_assert(baseSize % 16 == 0);
    static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                  "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleFp16Ub.GetPhyAddr();

    AntiquantVFImplFp8Nz<Q_T, KV_T, baseSize>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8D64(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubDstAddr_,
                                      __ubuf__ Q_T *ubOffsetAddr, __ubuf__ Q_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vOffset;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskLower64 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL64>();
    Reg::MaskReg qTypeMaskLower128 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL128>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
    Reg::MaskReg qTypeMaskHigher64;
    Reg::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll,
             qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

    uint32_t blockStride = 1 + dealRowCount;
    uint32_t repeatStride = 2;
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

    __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr;
    if constexpr (hasOffset) {
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr);
    }
    Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vScale, ubScaleAddr);

    // 对D64优化，相邻2行合并计算；
    for (uint16_t i = 0; i < loopCnt; i++) {
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
            (Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp, baseSize * 2);

        if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
            Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
            Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
        } else {
            Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
        }

        if constexpr (hasOffset) {
            Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskLower128);
        }
        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskLower128);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8D64(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                        LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                        LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize == 64);
    static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                  "antiquant w8, KV_T must be int8_t or hifloat8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ Q_T *ubOffsetAddr = (__ubuf__ Q_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8D64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr,
                                                                         ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T>
__simd_callee__ static inline void LoadCastDupLowHighOr(
    Reg::RegTensor<ANTIQ_PARAMS_T> &vParam, Reg::UnalignRegForLoad &uReg, __ubuf__ ANTIQ_PARAMS_T *&ubAddr,
    Reg::RegTensor<Q_T> &vParamFp16, Reg::RegTensor<Q_T> &vParamFp16Low, Reg::RegTensor<Q_T> &vParamFp16High,
    Reg::MaskReg &maskOne, Reg::MaskReg &qTypeMaskLower64, Reg::MaskReg &qTypeMaskHigher64, Reg::MaskReg &kvTypeMaskAll)
{
    Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vParam, uReg, ubAddr, 1); // 1表示ub自动往后偏移1个float
    Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vParamFp16, vParam, maskOne);
    Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
        (Reg::RegTensor<uint16_t> &)vParamFp16Low, (Reg::RegTensor<uint16_t> &)vParamFp16, qTypeMaskLower64);

    Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vParam, uReg, ubAddr, 1); // 1表示ub自动往后偏移1个float
    Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vParamFp16, vParam, maskOne);
    Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
        (Reg::RegTensor<uint16_t> &)vParamFp16High, (Reg::RegTensor<uint16_t> &)vParamFp16, qTypeMaskHigher64);
    Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vParamFp16,
                                                   (Reg::RegTensor<uint16_t> &)vParamFp16Low,
                                                   (Reg::RegTensor<uint16_t> &)vParamFp16High, kvTypeMaskAll);
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void CastW8KvToRes(Reg::RegTensor<Q_T> &vRes, Reg::RegTensor<KV_T> &vKvData,
                                                 Reg::RegTensor<half> &vCastFp16Res, Reg::MaskReg &kvTypeMaskAll)
{
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
    } else {
        Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD64(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                              __ubuf__ Q_T *ubDstAddr_, __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                              __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vOffsetFp16Low;
    Reg::RegTensor<Q_T> vOffsetFp16High;
    Reg::RegTensor<Q_T> vScaleFp16;
    Reg::RegTensor<Q_T> vScaleFp16High;
    Reg::RegTensor<Q_T> vScaleFp16Low;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;
    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskLower128 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL128>();
    Reg::MaskReg qTypeMaskLower64 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL64>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
    Reg::MaskReg qTypeMaskHigher64;
    Reg::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll,
             qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;

    uint32_t blockStride = 1 + dealRowCount;
    uint32_t repeatStride = 2;

    Reg::LoadUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
        Reg::LoadUnAlignPre(u1, ubOffsetAddr);
    }
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行
    // 对D64优化，相邻2行合并计算；+1兼容奇数行场景
    for (uint16_t i = 0; i < loopCnt; i++) {
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
            (Reg::RegTensor<uint8_t> &)vKvData, ubSrcAddr, baseSize * 2);

        CastW8KvToRes<Q_T, KV_T>(vRes, vKvData, vCastFp16Res, kvTypeMaskAll);

        LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, vScaleFp16, vScaleFp16Low, vScaleFp16High,
                                                  maskOne, qTypeMaskLower64, qTypeMaskHigher64, kvTypeMaskAll);
        if constexpr (hasOffset) {
            LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, vOffsetFp16, vOffsetFp16Low,
                                                      vOffsetFp16High, maskOne, qTypeMaskLower64, qTypeMaskHigher64,
                                                      kvTypeMaskAll);
            Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskLower128);
        }

        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskLower128);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, vRes, blockStride, repeatStride, qTypeMaskLower64);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr_, vRes, blockStride, repeatStride, qTypeMaskHigher64);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD64(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize == 64);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8PerTokenD64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
        ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8Norm(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubOffsetAddr,
                                       __ubuf__ Q_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vOffset;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = static_cast<uint16_t>(baseSize / 128);

    for (uint16_t j = 0; j < loops; j++) {
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
        __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + j * 128;

        if constexpr (hasOffset) {
            Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vOffset, ubOffsetAddr + j * 128);
        }
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vScale, ubScaleAddr + j * 128);

        // #pragma unroll(4)
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
                (Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp, baseSize);

            if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
                Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvTypeMaskAll);
                Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvTypeMaskAll);
            } else {
                Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvTypeMaskAll);
            }

            if constexpr (hasOffset) {
                Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
            }

            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vScale, vRes, qTypeMaskAll);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8Norm(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                         LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                         LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize % 128 == 0);
    static_assert(IsSameType<KV_T, int8_t>::value || IsSameType<KV_T, hifloat8_t>::value,
                  "antiquant w8, KV_T must be int8_t or hifloat8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubOffsetAddr = (__ubuf__ Q_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8Norm<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr,
                                                                          ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD128(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                               __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                               __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vScaleFp16;

    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = baseSize / 128;
    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;
    for (uint16_t j = 0; j < static_cast<uint16_t>(loops); j++) {
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
        __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + j * 128;
        __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddrTemp = ubOffsetAddr;
        __ubuf__ ANTIQ_PARAMS_T *ubScaleAddrTemp = ubScaleAddr;

        Reg::LoadUnAlignPre(u0, ubScaleAddrTemp);
        if constexpr (hasOffset) {
            Reg::LoadUnAlignPre(u1, ubOffsetAddrTemp);
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
                (Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp, baseSize);

            CastW8KvToRes<Q_T, KV_T>(vRes, vKvData, vCastFp16Res, kvTypeMaskAll);
            Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddrTemp, 1); // 1表示ub自动往后偏移1个float
            Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
            Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                (Reg::RegTensor<uint16_t> &)vScaleFp16, (Reg::RegTensor<uint16_t> &)vScaleFp16, qTypeMaskAll);
            if constexpr (hasOffset) {
                Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddrTemp, 1); // 1表示ub自动往后偏移1个float
                Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
                Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                    (Reg::RegTensor<uint16_t> &)vOffsetFp16, (Reg::RegTensor<uint16_t> &)vOffsetFp16, qTypeMaskAll);
                Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vOffsetFp16, vRes, qTypeMaskAll);
            }

            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vScaleFp16, vRes, qTypeMaskAll);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD128(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize == 128);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8PerTokenD128<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr,
                                                                                  ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, bool hasOffset>
__simd_callee__ static inline void LoadPerTokenAntiqParams(
    Reg::RegTensor<ANTIQ_PARAMS_T> &vScale, Reg::UnalignRegForLoad &u0, __ubuf__ ANTIQ_PARAMS_T *&ubScaleAddr,
    Reg::RegTensor<Q_T> &vScaleFp16, Reg::MaskReg &maskOne, Reg::MaskReg &qMaskAll,
    Reg::RegTensor<ANTIQ_PARAMS_T> &vOffset, Reg::UnalignRegForLoad &u1, __ubuf__ ANTIQ_PARAMS_T *&ubOffsetAddr,
    Reg::RegTensor<Q_T> &vOffsetFp16)
{
    Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
    Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
    Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
        (Reg::RegTensor<uint16_t> &)vScaleFp16, (Reg::RegTensor<uint16_t> &)vScaleFp16, qMaskAll);
    if constexpr (hasOffset) {
        Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
        Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
        Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
            (Reg::RegTensor<uint16_t> &)vOffsetFp16, (Reg::RegTensor<uint16_t> &)vOffsetFp16, qMaskAll);
    }
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void CastW8KvToResPair(Reg::RegTensor<Q_T> &vRes, Reg::RegTensor<Q_T> &vRes1,
                                                     Reg::RegTensor<KV_T> &vKvData, Reg::RegTensor<KV_T> &vKvData1,
                                                     Reg::RegTensor<half> &vCastFp16Res,
                                                     Reg::RegTensor<half> &vCastFp16Res1, Reg::MaskReg &kvMaskAll)
{
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvMaskAll);
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res1, vKvData1, kvMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes1, vCastFp16Res1, kvMaskAll);
    } else {
        Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
        Reg::Cast<Q_T, KV_T, castTrait>(vRes1, vKvData1, kvMaskAll);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD256(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ uint8_t *ubSrcAddr1,
                                               __ubuf__ Q_T *ubDstAddr, __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                               __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<Q_T> vScaleFp16;
    Reg::RegTensor<Q_T> vOffsetFp16;

    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<Q_T> vRes1;
    Reg::RegTensor<half> vCastFp16Res;
    Reg::RegTensor<half> vCastFp16Res1;

    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = 1 + dealRowCount;
    uint32_t repeatStride = 1;
    Reg::UnalignRegForLoad u1;
    Reg::UnalignRegForLoad u0;
    Reg::LoadUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
        Reg::LoadUnAlignPre(u1, ubOffsetAddr);
    }
    __ubuf__ Q_T *ubDstAddr1 = ubDstAddr + blockStride * 128;
    for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
        // 读入每行的伪量化参数
        LoadPerTokenAntiqParams<Q_T, ANTIQ_PARAMS_T, hasOffset>(vScale, u0, ubScaleAddr, vScaleFp16, maskOne, qMaskAll,
                                                                vOffset, u1, ubOffsetAddr, vOffsetFp16);
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
            (Reg::RegTensor<uint8_t> &)vKvData, ubSrcAddr, 256); // d=256，自动往后偏移256个数
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
            (Reg::RegTensor<uint8_t> &)vKvData1, ubSrcAddr1, 256); // d=256
        CastW8KvToResPair<Q_T, KV_T>(vRes, vRes1, vKvData, vKvData1, vCastFp16Res, vCastFp16Res1, kvMaskAll);
        if constexpr (hasOffset) {
            Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
            Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes1, vRes1, vOffsetFp16, qMaskAll);
        }

        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes1, vRes1, vScaleFp16, qMaskAll);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD256(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize == 256);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ uint8_t *ubSrcAddr1 = ubSrcAddr + 128;
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8PerTokenD256<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
        ubSrcAddr, ubSrcAddr1, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename KV_T>
__simd_callee__ static inline void LoadAlignW8Quad(Reg::RegTensor<KV_T> &vKvData, Reg::RegTensor<KV_T> &vKvData1,
                                                   Reg::RegTensor<KV_T> &vKvData2, Reg::RegTensor<KV_T> &vKvData3,
                                                   __ubuf__ uint8_t *&ubSrcAddr, __ubuf__ uint8_t *&ubSrcAddr1,
                                                   __ubuf__ uint8_t *&ubSrcAddr2, __ubuf__ uint8_t *&ubSrcAddr3)
{
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
        (Reg::RegTensor<uint8_t> &)vKvData, ubSrcAddr, 512); // d=512 每次往后偏移512
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
        (Reg::RegTensor<uint8_t> &)vKvData1, ubSrcAddr1, 512); // d=512
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
        (Reg::RegTensor<uint8_t> &)vKvData2, ubSrcAddr2, 512); // d=512
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B8>(
        (Reg::RegTensor<uint8_t> &)vKvData3, ubSrcAddr3, 512); // d=512
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void CastW8KvToResQuad(
    Reg::RegTensor<Q_T> &vRes, Reg::RegTensor<Q_T> &vRes1, Reg::RegTensor<Q_T> &vRes2, Reg::RegTensor<Q_T> &vRes3,
    Reg::RegTensor<KV_T> &vKvData, Reg::RegTensor<KV_T> &vKvData1, Reg::RegTensor<KV_T> &vKvData2,
    Reg::RegTensor<KV_T> &vKvData3, Reg::RegTensor<half> &vCastFp16Res, Reg::RegTensor<half> &vCastFp16Res1,
    Reg::RegTensor<half> &vCastFp16Res2, Reg::RegTensor<half> &vCastFp16Res3, Reg::MaskReg &kvMaskAll)
{
    if constexpr (std::is_same<Q_T, bfloat16_t>::value) {
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res, vKvData, kvMaskAll);
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res1, vKvData1, kvMaskAll);
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res2, vKvData2, kvMaskAll);
        Reg::Cast<half, KV_T, castTrait>(vCastFp16Res3, vKvData3, kvMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes, vCastFp16Res, kvMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes1, vCastFp16Res1, kvMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes2, vCastFp16Res2, kvMaskAll);
        Reg::Cast<Q_T, half, castTrait1>(vRes3, vCastFp16Res3, kvMaskAll);
    } else {
        Reg::Cast<Q_T, KV_T, castTrait>(vRes, vKvData, kvMaskAll);
        Reg::Cast<Q_T, KV_T, castTrait>(vRes1, vKvData1, kvMaskAll);
        Reg::Cast<Q_T, KV_T, castTrait>(vRes2, vKvData2, kvMaskAll);
        Reg::Cast<Q_T, KV_T, castTrait>(vRes3, vKvData3, kvMaskAll);
    }
}

template <typename Q_T, bool hasOffset>
__simd_callee__ static inline void AddMulStoreFp8Quad(Reg::RegTensor<Q_T> &vRes, Reg::RegTensor<Q_T> &vRes1,
                                                      Reg::RegTensor<Q_T> &vRes2, Reg::RegTensor<Q_T> &vRes3,
                                                      Reg::RegTensor<Q_T> &vOffsetFp16, Reg::RegTensor<Q_T> &vScaleFp16,
                                                      __ubuf__ Q_T *&ubDstAddr, __ubuf__ Q_T *&ubDstAddr1,
                                                      __ubuf__ Q_T *&ubDstAddr2, __ubuf__ Q_T *&ubDstAddr3,
                                                      uint32_t blockStride, uint32_t repeatStride,
                                                      Reg::MaskReg &qMaskAll)
{
    if constexpr (hasOffset) {
        Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qMaskAll);
        Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes1, vRes1, vOffsetFp16, qMaskAll);
        Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes2, vRes2, vOffsetFp16, qMaskAll);
        Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes3, vRes3, vOffsetFp16, qMaskAll);
    }
    Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vScaleFp16, vRes, qMaskAll);
    Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes1, vScaleFp16, vRes1, qMaskAll);
    Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes2, vRes2, vScaleFp16, qMaskAll);
    Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes3, vRes3, vScaleFp16, qMaskAll);
    Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
    Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
    Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr2, vRes2, blockStride, repeatStride, qMaskAll);
    Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        ubDstAddr3, vRes3, blockStride, repeatStride, qMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplW8PerTokenD512(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ uint8_t *ubSrcAddr1,
                                               __ubuf__ uint8_t *ubSrcAddr2, __ubuf__ uint8_t *ubSrcAddr3,
                                               __ubuf__ Q_T *ubDstAddr, __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                               __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<KV_T> vKvData2;
    Reg::RegTensor<KV_T> vKvData3;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vScaleFp16;

    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<Q_T> vRes1;
    Reg::RegTensor<Q_T> vRes2;
    Reg::RegTensor<Q_T> vRes3;
    Reg::RegTensor<half> vCastFp16Res;
    Reg::RegTensor<half> vCastFp16Res1;
    Reg::RegTensor<half> vCastFp16Res2;
    Reg::RegTensor<half> vCastFp16Res3;

    Reg::MaskReg kvMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg qMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = 1 + dealRowCount;
    uint32_t repeatStride = 1;
    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;
    Reg::LoadUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
        Reg::LoadUnAlignPre(u1, ubOffsetAddr);
    }
    __ubuf__ Q_T *ubDstAddr1 = ubDstAddr + blockStride * 128;
    __ubuf__ Q_T *ubDstAddr2 = ubDstAddr + blockStride * 128 * 2;
    __ubuf__ Q_T *ubDstAddr3 = ubDstAddr + blockStride * 128 * 3;
    for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
        // 读入每行的伪量化参数
        LoadPerTokenAntiqParams<Q_T, ANTIQ_PARAMS_T, hasOffset>(vScale, u0, ubScaleAddr, vScaleFp16, maskOne, qMaskAll,
                                                                vOffset, u1, ubOffsetAddr, vOffsetFp16);
        LoadAlignW8Quad<KV_T>(vKvData, vKvData1, vKvData2, vKvData3, ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3);
        CastW8KvToResQuad<Q_T, KV_T>(vRes, vRes1, vRes2, vRes3, vKvData, vKvData1, vKvData2, vKvData3, vCastFp16Res,
                                     vCastFp16Res1, vCastFp16Res2, vCastFp16Res3, kvMaskAll);
        AddMulStoreFp8Quad<Q_T, hasOffset>(vRes, vRes1, vRes2, vRes3, vOffsetFp16, vScaleFp16, ubDstAddr, ubDstAddr1,
                                           ubDstAddr2, ubDstAddr3, blockStride, repeatStride, qMaskAll);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFW8PerTokenD512(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    static_assert(baseSize == 512);
    static_assert(IsSameType<KV_T, int8_t>::value, "antiquant perToken w8, KV_T must be int8_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ uint8_t *ubSrcAddr1 = ubSrcAddr + 128;
    __ubuf__ uint8_t *ubSrcAddr2 = ubSrcAddr + 128 * 2;
    __ubuf__ uint8_t *ubSrcAddr3 = ubSrcAddr + 128 * 3;
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplW8PerTokenD512<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
        ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false,
          bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<int8_t> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount,
                                       uint32_t headDim, uint32_t copyTotalS)
{
    if constexpr (isKvCacheNz) {
        if constexpr (!isPerToken) {
            AntiquantVFW8Nz<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb,
                                                                              antiqScaleUb, dealRowCount);
        } else {
            AntiquantVFW8PerTokenNz<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
        }
    } else {
        if constexpr (isPerToken) {
            if constexpr (baseSize == 64) {
                AntiquantVFW8PerTokenD64<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            } else if constexpr (baseSize == 128) {
                AntiquantVFW8PerTokenD128<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            } else if constexpr (baseSize == 256) {
                AntiquantVFW8PerTokenD256<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            } else {
                AntiquantVFW8PerTokenD512<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            }
        } else {
            if constexpr (baseSize == 64) {
                AntiquantVFW8D64<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb,
                                                                                   antiqScaleUb, dealRowCount);
            } else {
                AntiquantVFW8Norm<Q_T, int8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            }
        }
    }
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false,
          bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<hifloat8_t> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                       LocalTensor<Q_T> &antiqOffsetUb, LocalTensor<Q_T> &antiqScaleUb,
                                       uint32_t dealRowCount, uint32_t headDim, uint32_t copyTotalS)
{
    if constexpr (isKvCacheNz) {
        AntiquantVFW8Nz<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb,
                                                                              antiqScaleUb, dealRowCount);
    } else {
        if constexpr (baseSize == 64) {
            AntiquantVFW8D64<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(antiqInUb, antiqResUb, antiqOffsetUb,
                                                                                   antiqScaleUb, dealRowCount);
        } else {
            AntiquantVFW8Norm<Q_T, hifloat8_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
        }
    }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplFp8D64(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                       __ubuf__ Q_T *ubDstAddrEven, __ubuf__ Q_T *ubDstAddr_,
                                       __ubuf__ Q_T *ubScalerSrcAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vMulRes;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskLower64 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL64>();
    Reg::MaskReg qTypeMaskLower128 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL128>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskHigher64;
    // NZ:
    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 2;
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行

    __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr;

    // 加载 scale
    Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vScale, ubScalerSrcAddr);
    Reg::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll,
             qTypeMaskAll); // qTypeMaskAll与qTypeMaskLower64异或得到qTypeMaskHigher64

    // D=64时相邻2行合并做伪量化计算，减小循环次数；额外+1是为了处理奇数行时场景
    for (uint16_t i = 0; i < loopCnt; i++) {
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
            (Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp, baseSize * 2);

        // cast操作, Fp8->Fp32
        Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
        Reg::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);

        // cast操作, Fp32->Fp16/Bf16
        Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
        Reg::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);

        Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vCastRes0,
                                                       (Reg::RegTensor<uint16_t> &)vCastRes0,
                                                       (Reg::RegTensor<uint16_t> &)vCastRes1, kvTypeMaskAll);

        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes, vCastRes0, vScale, qTypeMaskLower128);

        // 将输出结果copy到UB
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, vMulRes, blockStride, repeatStride, qTypeMaskLower64);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr_, vMulRes, blockStride, repeatStride, qTypeMaskHigher64);
    }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFFp8D64(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                         LocalTensor<Q_T> &antiqScaleFp16Ub, uint32_t dealRowCount)
{
    static_assert(baseSize == 64);
    static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                  "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddrEven = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ Q_T *ubScalerSrcAddr = (__ubuf__ Q_T *)antiqScaleFp16Ub.GetPhyAddr();

    AntiquantVFImplFp8D64<Q_T, KV_T, baseSize>(ubSrcAddr, ubDstAddr, ubDstAddrEven, ubDstAddr_, ubScalerSrcAddr,
                                               dealRowCount);
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__simd_vf__ void AntiquantVFImplFp8Norm(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                        __ubuf__ Q_T *ubScalerSrcAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vScale;
    Reg::RegTensor<Q_T> vMulRes;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();

    // 目前不支持D泛化, 仅支持D128对齐场景

    // NZ:
    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = static_cast<uint16_t>(baseSize / 128);
    for (uint16_t j = 0; j < loops; j++) {
        __ubuf__ Q_T *ubDstAddrOdd = ubDstAddr + blockStride * 128 * j;
        __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + j * 128;
        // 加载 scale
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_NORM>(vScale, ubScalerSrcAddr + j * 128);

        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) { // 共处理dealRowCount * 128个Fp8元素
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                (Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp, baseSize);

            // cast操作, Fp8->Fp32
            Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
            Reg::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);

            // cast操作, Fp32->Fp16/Bf16
            Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
            Reg::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);

            Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vCastRes0,
                                                           (Reg::RegTensor<uint16_t> &)vCastRes0,
                                                           (Reg::RegTensor<uint16_t> &)vCastRes1, kvTypeMaskAll);

            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes, vCastRes0, vScale, qTypeMaskAll);

            // 将输出结果copy到UB
            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrOdd, vMulRes, blockStride, repeatStride, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, uint32_t baseSize>
__aicore__ inline void AntiquantVFFp8Norm(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                          LocalTensor<Q_T> &antiqScaleFp16Ub, uint32_t dealRowCount)
{
    static_assert(baseSize % 128 == 0);
    static_assert(IsSameType<KV_T, fp8_e4m3fn_t>::value || IsSameType<KV_T, fp8_e5m2_t>::value,
                  "antiquant w8, KV_T must be fp8_e4m3fn_t or fp8_e5m2_t");
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubScalerSrcAddr = (__ubuf__ Q_T *)antiqScaleFp16Ub.GetPhyAddr();

    AntiquantVFImplFp8Norm<Q_T, KV_T, baseSize>(ubSrcAddr, ubDstAddr, ubScalerSrcAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false,
          bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp8_e5m2_t> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount,
                                       uint32_t headDim, uint32_t copyTotalS)
{
    if constexpr (isKvCacheNz) {
        AntiquantVFFp8Nz<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
    } else {
        if constexpr (baseSize == 64) {
            AntiquantVFFp8D64<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
        } else {
            AntiquantVFFp8Norm<Q_T, fp8_e5m2_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
        }
    }
}

template <typename Q_T, typename KV_T>
__simd_callee__ static inline void Fp8CastToRes(Reg::RegTensor<Q_T> &vRes, Reg::RegTensor<KV_T> &vKvData,
                                                Reg::RegTensor<float> &vCastFp32Res0,
                                                Reg::RegTensor<float> &vCastFp32Res1, Reg::RegTensor<Q_T> &vCastRes0,
                                                Reg::RegTensor<Q_T> &vCastRes1, Reg::MaskReg &kvMaskAll)
{
    Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvMaskAll);
    Reg::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvMaskAll);
    Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvMaskAll);
    Reg::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvMaskAll);
    Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vRes,
                                                   (Reg::RegTensor<uint16_t> &)vCastRes0,
                                                   (Reg::RegTensor<uint16_t> &)vCastRes1, kvMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD64(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                               __ubuf__ Q_T *ubDstAddr_, __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                               __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vOffsetFp16High;
    Reg::RegTensor<Q_T> vOffsetFp16Low;
    Reg::RegTensor<Q_T> vScaleFp16;
    Reg::RegTensor<Q_T> vScaleFp16High;
    Reg::RegTensor<Q_T> vScaleFp16Low;
    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskLower64 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL64>();
    Reg::MaskReg qTypeMaskLower128 = Reg::CreateMask<Q_T, Reg::MaskPattern::VL128>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）
    Reg::MaskReg qTypeMaskHigher64;
    Reg::Xor(qTypeMaskHigher64, qTypeMaskLower64, qTypeMaskAll, qTypeMaskAll); // 异或得到higher64

    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 2;

    Reg::LoadUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
        Reg::LoadUnAlignPre(u1, ubOffsetAddr);
    }
    uint16_t loopCnt = static_cast<uint16_t>((dealRowCount + 1) / 2); // +1是为了兼容处理奇数行
    // 对D64优化，相邻2行合并计算；+1兼容奇数行场景
    for (uint16_t i = 0; i < loopCnt; i++) {
        // POST_MODE_UPDATE 表示 UB 地址在搬入后要自动更新
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
            (Reg::RegTensor<uint8_t> &)vKvData, ubSrcAddr, baseSize * 2);
        Fp8CastToRes<Q_T, KV_T>(vCastRes0, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvTypeMaskAll);
        LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, vScaleFp16, vScaleFp16Low, vScaleFp16High,
                                                  maskOne, qTypeMaskLower64, qTypeMaskHigher64, kvTypeMaskAll);
        if constexpr (hasOffset) {
            LoadCastDupLowHighOr<Q_T, ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, vOffsetFp16, vOffsetFp16Low,
                                                      vOffsetFp16High, maskOne, qTypeMaskLower64, qTypeMaskHigher64,
                                                      kvTypeMaskAll);
            Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vCastRes0, vCastRes0, vOffsetFp16, qTypeMaskLower128);
        }

        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vCastRes0, vCastRes0, vScaleFp16, qTypeMaskLower128);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, vCastRes0, blockStride, repeatStride, qTypeMaskLower64);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr_, vCastRes0, blockStride, repeatStride, qTypeMaskHigher64);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD64(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                 LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    ASCENDC_ASSERT((baseSize == 64), { KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 64.", baseSize); });
    ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                   { KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr_ = ubDstAddr + 16 - (dealRowCount + 1) * 32 * 4 / 2;
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplFp8PerTokenD64<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
        ubSrcAddr, ubDstAddr, ubDstAddr_, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD128(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                                __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                                __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vScaleFp16;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<half> vCastFp16Res;

    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    const uint16_t loops = baseSize / 128;
    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;
    for (uint16_t j = 0; j < static_cast<uint16_t>(loops); j++) {
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + blockStride * 128 * j;
        __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + j * 128;
        __ubuf__ ANTIQ_PARAMS_T *ubScaleAddrTemp = ubScaleAddr;
        __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddrTemp = ubOffsetAddr;

        Reg::LoadUnAlignPre(u0, ubScaleAddrTemp);
        if constexpr (hasOffset) {
            Reg::LoadUnAlignPre(u1, ubOffsetAddrTemp);
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                (Reg::RegTensor<uint8_t> &)vKvData, ubSrcTemp, baseSize);

            Fp8CastToRes<Q_T, KV_T>(vRes, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvTypeMaskAll);
            Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddrTemp, 1); // 1表示ub自动往后偏移1个float
            Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
            Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                (Reg::RegTensor<uint16_t> &)vScaleFp16, (Reg::RegTensor<uint16_t> &)vScaleFp16, qTypeMaskAll);

            if constexpr (hasOffset) {
                Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddrTemp, 1); // 1表示ub自动往后偏移1个float
                Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
                Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                    (Reg::RegTensor<uint16_t> &)vOffsetFp16, (Reg::RegTensor<uint16_t> &)vOffsetFp16, qTypeMaskAll);
                Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffsetFp16, qTypeMaskAll);
            }

            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qTypeMaskAll);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vRes, blockStride, repeatStride, qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD128(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                  LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                  LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    ASCENDC_ASSERT((baseSize == 128), { KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 128.", baseSize); });
    ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                   { KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplFp8PerTokenD128<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr,
                                                                                   ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD256(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ uint8_t *ubSrcAddr1,
                                                __ubuf__ Q_T *ubDstAddr, __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                                __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vScaleFp16;

    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<Q_T> vRes1;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;

    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t blockStride = dealRowCount + 1;
    uint32_t repeatStride = 1;
    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;
    Reg::LoadUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
        Reg::LoadUnAlignPre(u1, ubOffsetAddr);
    }
    __ubuf__ Q_T *ubDstAddr1 = ubDstAddr + blockStride * 128;

    for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
        // 读入每行的伪量化参数
        Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vScale, u0, ubScaleAddr, 1); // 1表示ub自动往后偏移1个float
        Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vScaleFp16, vScale, maskOne);
        Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
            (Reg::RegTensor<uint16_t> &)vScaleFp16, (Reg::RegTensor<uint16_t> &)vScaleFp16, qMaskAll);
        if constexpr (hasOffset) {
            Reg::LoadUnAlign<ANTIQ_PARAMS_T>(vOffset, u1, ubOffsetAddr, 1); // 1表示ub自动往后偏移1个float
            Reg::Cast<Q_T, ANTIQ_PARAMS_T, castTrait0>(vOffsetFp16, vOffset, maskOne);
            Reg::Duplicate<uint16_t, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(
                (Reg::RegTensor<uint16_t> &)vOffsetFp16, (Reg::RegTensor<uint16_t> &)vOffsetFp16, qMaskAll);
        }

        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
            (Reg::RegTensor<uint8_t> &)vKvData, ubSrcAddr, 256); // d=256，自动往后偏移256个数
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
            (Reg::RegTensor<uint8_t> &)vKvData1, ubSrcAddr1, 256); // d=256

        Fp8CastToRes<Q_T, KV_T>(vRes, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
        Fp8CastToRes<Q_T, KV_T>(vRes1, vKvData1, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);

        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScaleFp16, qMaskAll);
        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes1, vRes1, vScaleFp16, qMaskAll);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, vRes, blockStride, repeatStride, qMaskAll);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr1, vRes1, blockStride, repeatStride, qMaskAll);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD256(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                  LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                  LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    ASCENDC_ASSERT((baseSize == 256), { KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 256.", baseSize); });
    ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                   { KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ uint8_t *ubSrcAddr1 = ubSrcAddr + 128;
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplFp8PerTokenD256<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
        ubSrcAddr, ubSrcAddr1, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename KV_T>
__simd_callee__ static inline void LoadAlignFp8Quad(Reg::RegTensor<KV_T> &vKvData, Reg::RegTensor<KV_T> &vKvData1,
                                                    Reg::RegTensor<KV_T> &vKvData2, Reg::RegTensor<KV_T> &vKvData3,
                                                    __ubuf__ uint8_t *&ubSrcAddr, __ubuf__ uint8_t *&ubSrcAddr1,
                                                    __ubuf__ uint8_t *&ubSrcAddr2, __ubuf__ uint8_t *&ubSrcAddr3)
{
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
        (Reg::RegTensor<uint8_t> &)vKvData, ubSrcAddr, 512); // d=512 每次往后偏移512
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
        (Reg::RegTensor<uint8_t> &)vKvData1, ubSrcAddr1, 512); // d=512
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
        (Reg::RegTensor<uint8_t> &)vKvData2, ubSrcAddr2, 512); // d=512
    Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
        (Reg::RegTensor<uint8_t> &)vKvData3, ubSrcAddr3, 512); // d=512
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenD512(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ uint8_t *ubSrcAddr1,
                                                __ubuf__ uint8_t *ubSrcAddr2, __ubuf__ uint8_t *ubSrcAddr3,
                                                __ubuf__ Q_T *ubDstAddr, __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr,
                                                __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr, uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<KV_T> vKvData2;
    Reg::RegTensor<KV_T> vKvData3;
    Reg::RegTensor<ANTIQ_PARAMS_T> vOffset;
    Reg::RegTensor<ANTIQ_PARAMS_T> vScale;
    Reg::RegTensor<Q_T> vOffsetFp16;
    Reg::RegTensor<Q_T> vScaleFp16;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;

    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<Q_T> vRes1;
    Reg::RegTensor<Q_T> vRes2;
    Reg::RegTensor<Q_T> vRes3;

    Reg::MaskReg maskOne = Reg::CreateMask<ANTIQ_PARAMS_T, Reg::MaskPattern::VL1>();
    Reg::MaskReg kvMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    uint32_t repeatStride = 1;
    uint32_t blockStride = dealRowCount + 1;
    Reg::UnalignRegForLoad u0;
    Reg::UnalignRegForLoad u1;
    Reg::LoadUnAlignPre(u0, ubScaleAddr);
    if constexpr (hasOffset) {
        Reg::LoadUnAlignPre(u1, ubOffsetAddr);
    }
    __ubuf__ Q_T *ubDstAddr1 = ubDstAddr + blockStride * 128;
    __ubuf__ Q_T *ubDstAddr2 = ubDstAddr + blockStride * 256; // 128*2
    __ubuf__ Q_T *ubDstAddr3 = ubDstAddr + blockStride * 384; // 128*3

    for (uint16_t j = 0; j < static_cast<uint16_t>(dealRowCount); j++) {
        // 读入每行的伪量化参数
        LoadPerTokenAntiqParams<Q_T, ANTIQ_PARAMS_T, hasOffset>(vScale, u0, ubScaleAddr, vScaleFp16, maskOne, qMaskAll,
                                                                vOffset, u1, ubOffsetAddr, vOffsetFp16);

        LoadAlignFp8Quad<KV_T>(vKvData, vKvData1, vKvData2, vKvData3, ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3);

        Fp8CastToRes<Q_T, KV_T>(vRes, vKvData, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
        Fp8CastToRes<Q_T, KV_T>(vRes1, vKvData1, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
        Fp8CastToRes<Q_T, KV_T>(vRes2, vKvData2, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);
        Fp8CastToRes<Q_T, KV_T>(vRes3, vKvData3, vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvMaskAll);

        AddMulStoreFp8Quad<Q_T, hasOffset>(vRes, vRes1, vRes2, vRes3, vOffsetFp16, vScaleFp16, ubDstAddr, ubDstAddr1,
                                           ubDstAddr2, ubDstAddr3, blockStride, repeatStride, qMaskAll);
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenD512(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                  LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                  LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    ASCENDC_ASSERT((baseSize == 512), { KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 512.", baseSize); });
    ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                   { KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ uint8_t *ubSrcAddr1 = ubSrcAddr + 128;
    __ubuf__ uint8_t *ubSrcAddr2 = ubSrcAddr + 256; // 128*2
    __ubuf__ uint8_t *ubSrcAddr3 = ubSrcAddr + 384; // 128*3
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ ANTIQ_PARAMS_T *ubOffsetAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ ANTIQ_PARAMS_T *ubScaleAddr = (__ubuf__ ANTIQ_PARAMS_T *)antiqScaleUb.GetPhyAddr();

    AntiquantVFImplFp8PerTokenD512<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(
        ubSrcAddr, ubSrcAddr1, ubSrcAddr2, ubSrcAddr3, ubDstAddr, ubOffsetAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T, bool hasOffset>
__simd_callee__ static inline void Fp8NzQuantAndStore(Reg::RegTensor<KV_T> &vKvData, Reg::RegTensor<Q_T> &vRes,
                                                      __ubuf__ uint8_t *ubSrc, Reg::RegTensor<Q_T> &vOffset,
                                                      Reg::RegTensor<Q_T> &vScale, __ubuf__ Q_T *ubDst,
                                                      Reg::RegTensor<float> &vCastFp32Res0,
                                                      Reg::RegTensor<float> &vCastFp32Res1,
                                                      Reg::RegTensor<Q_T> &vCastRes0, Reg::RegTensor<Q_T> &vCastRes1,
                                                      Reg::MaskReg &kvTypeMaskAll, Reg::MaskReg &qTypeMaskAll)
{
    Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B16>((Reg::RegTensor<uint8_t> &)vKvData, ubSrc);
    Reg::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData, kvTypeMaskAll);
    Reg::Cast<float, KV_T, castTraitFp8_2>(vCastFp32Res1, vKvData, kvTypeMaskAll);
    Reg::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, kvTypeMaskAll);
    Reg::Cast<Q_T, float, castTraitFp8_4>(vCastRes1, vCastFp32Res1, kvTypeMaskAll);
    Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((Reg::RegTensor<uint16_t> &)vRes,
                                                   (Reg::RegTensor<uint16_t> &)vCastRes0,
                                                   (Reg::RegTensor<uint16_t> &)vCastRes1, kvTypeMaskAll);
    if constexpr (hasOffset) {
        Reg::Add<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vOffset, qTypeMaskAll);
    }
    Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vRes, vRes, vScale, qTypeMaskAll);
    Reg::StoreAlign<Q_T, Reg::StoreDist::DIST_NORM_B16>(ubDst, vRes, qTypeMaskAll);
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__simd_vf__ void AntiquantVFImplFp8PerTokenNz(__ubuf__ uint8_t *ubSrcAddr, __ubuf__ Q_T *ubDstAddr,
                                              __ubuf__ Q_T *ubOffsetAddr, __ubuf__ Q_T *ubScaleAddr,
                                              uint32_t dealRowCount)
{
    Reg::RegTensor<KV_T> vKvData;
    Reg::RegTensor<Q_T> vOffsetFirst;
    Reg::RegTensor<Q_T> vOffsetBack;
    Reg::RegTensor<Q_T> vScaleFirst;
    Reg::RegTensor<Q_T> vScaleBack;
    Reg::RegTensor<Q_T> vRes;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg qTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>(); // Q_T 所有元素（共128个）

    // UB总共dealRowCount行 * baseSize列，每次处理8行 * 16列 = 128个元素
    const uint32_t rowBaseSize = 8;        // 8行
    const uint32_t colBaseSize = 16;       // 16列
    const uint32_t dealBaseNum = 128;      // 128个元素
    const uint32_t doubleRowBaseSize = 16; // 每16行交替，防止bank冲突

    const uint32_t rowStride = doubleRowBaseSize * colBaseSize; // 16 * 16
    const uint32_t colDstStride = dealRowCount * colBaseSize;
    const uint32_t colSrcStride = (dealRowCount * colBaseSize + 31) >> 5U << 5U; // 32B对齐
    const uint16_t colLoopCnt = static_cast<uint16_t>(baseSize / colBaseSize);
    const uint16_t rowLoopCnt =
        static_cast<uint16_t>((dealRowCount + doubleRowBaseSize - 1) / doubleRowBaseSize); // 16行对齐

    for (uint16_t rowLoop = 0; rowLoop < rowLoopCnt; rowLoop++) {
        uint16_t rowLoopIdx = rowLoopCnt - 1 - rowLoop;
        __ubuf__ Q_T *ubOffsetAddrTmp = ubOffsetAddr + rowLoopIdx * doubleRowBaseSize;
        __ubuf__ Q_T *ubScaleAddrTmp = ubScaleAddr + rowLoopIdx * doubleRowBaseSize;

        if constexpr (hasOffset) {
            Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vOffsetFirst, ubOffsetAddrTmp);
            Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vOffsetBack, ubOffsetAddrTmp + rowBaseSize);
        }
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vScaleFirst, ubScaleAddrTmp);
        Reg::LoadAlign<Q_T, Reg::LoadDist::DIST_E2B_B16>(vScaleBack, ubScaleAddrTmp + rowBaseSize);
        for (uint16_t colLoopIdx = 0; colLoopIdx < colLoopCnt; colLoopIdx++) {
            __ubuf__ uint8_t *ubSrcTemp = ubSrcAddr + rowStride * rowLoopIdx + colSrcStride * colLoopIdx;
            __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + rowStride * rowLoopIdx + colDstStride * colLoopIdx;

            // 后半组
            Fp8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vRes, ubSrcTemp + dealBaseNum, vOffsetBack, vScaleBack,
                                                     ubDstAddrTmp + dealBaseNum, vCastFp32Res0, vCastFp32Res1,
                                                     vCastRes0, vCastRes1, kvTypeMaskAll, qTypeMaskAll);

            // 前半组
            Fp8NzQuantAndStore<Q_T, KV_T, hasOffset>(vKvData, vRes, ubSrcTemp, vOffsetFirst, vScaleFirst, ubDstAddrTmp,
                                                     vCastFp32Res0, vCastFp32Res1, vCastRes0, vCastRes1, kvTypeMaskAll,
                                                     qTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false>
__aicore__ inline void AntiquantVFFp8PerTokenNz(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                                LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                                LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount)
{
    ASCENDC_ASSERT((baseSize % 16 == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "baseSize is %d, which must be 16 aligned.", baseSize); });
    ASCENDC_ASSERT((IsSameType<KV_T, fp8_e4m3fn_t>::value),
                   { KERNEL_LOG(KERNEL_ERROR, "Antiquant fp8 PerToken, KV_T must be fp8_e4m3."); });
    __ubuf__ uint8_t *ubSrcAddr = (__ubuf__ uint8_t *)(antiqInUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(antiqResUb.GetPhyAddr());
    __ubuf__ Q_T *ubOffsetAddr = (__ubuf__ Q_T *)antiqOffsetUb.GetPhyAddr();
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)antiqScaleUb.GetPhyAddr();
    AntiquantVFImplFp8PerTokenNz<Q_T, KV_T, ANTIQ_PARAMS_T, baseSize, hasOffset>(ubSrcAddr, ubDstAddr, ubOffsetAddr,
                                                                                 ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false, bool isPerToken = false,
          bool isKvCacheNz = false>
__aicore__ inline void AntiquantVFImpl(LocalTensor<fp8_e4m3fn_t> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                       LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount,
                                       uint32_t headDim, uint32_t copyTotalS)
{
    if constexpr (isPerToken) {
        if constexpr (isKvCacheNz) {
            AntiquantVFFp8PerTokenNz<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
        } else {
            if constexpr (baseSize == 64) {
                AntiquantVFFp8PerTokenD64<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            } else if constexpr (baseSize == 128) {
                AntiquantVFFp8PerTokenD128<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            } else if constexpr (baseSize == 256) {
                AntiquantVFFp8PerTokenD256<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            } else if constexpr (baseSize == 512) {
                AntiquantVFFp8PerTokenD512<Q_T, fp8_e4m3fn_t, ANTIQ_PARAMS_T, baseSize, hasOffset>(
                    antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount);
            }
        }
    } else {
        if constexpr (isKvCacheNz) {
            AntiquantVFFp8Nz<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
        } else {
            if constexpr (baseSize == 64) {
                AntiquantVFFp8D64<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
            } else {
                AntiquantVFFp8Norm<Q_T, fp8_e4m3fn_t, baseSize>(antiqInUb, antiqResUb, antiqScaleUb, dealRowCount);
            }
        }
    }
}

template <typename Q_T, typename KV_T, typename ANTIQ_PARAMS_T, uint32_t baseSize, bool hasOffset = false,
          bool isPerToken = false, bool isKvCacheNz = false>
__aicore__ inline void AntiquantVF(LocalTensor<KV_T> &antiqInUb, LocalTensor<Q_T> &antiqResUb,
                                   LocalTensor<ANTIQ_PARAMS_T> &antiqOffsetUb,
                                   LocalTensor<ANTIQ_PARAMS_T> &antiqScaleUb, uint32_t dealRowCount, uint32_t headDim,
                                   uint32_t copyTotalS = 0)
{
    AntiquantVFImpl<Q_T, ANTIQ_PARAMS_T, baseSize, hasOffset, isPerToken, isKvCacheNz>(
        antiqInUb, antiqResUb, antiqOffsetUb, antiqScaleUb, dealRowCount, headDim, copyTotalS);
}

}; // namespace FaVectorApi

#endif
