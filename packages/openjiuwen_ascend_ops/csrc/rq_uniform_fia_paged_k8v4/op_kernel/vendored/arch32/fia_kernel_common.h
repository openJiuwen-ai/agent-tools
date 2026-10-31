/*!
 * \file fia_kernel_common.h
 * \brief FIA helpers (renamed from kernel_common.h to avoid shadowing AscendC).
 */
#ifndef FIA_KERNEL_COMMON_H
#define FIA_KERNEL_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../fia_public_define.h"

using namespace AscendC;
using namespace AttentionCommon;
using AscendC::GlobalTensor;

namespace fa_base_kernel {

__aicore__ inline uint64_t GetActualQSeqLength(GlobalTensor<uint64_t> &actualSeqLengths, uint32_t bIdx, const ConstInfo &constInfo)
{
    if (constInfo.actualLenQDims == 0) {
        return constInfo.qSeqSize;
    } 
    if (constInfo.actualLenQDims == 1 || bIdx == 0) {
        return actualSeqLengths.GetValue(0);
    }
    if (constInfo.accumQSeqFlag) {
        return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
    } else {
        return actualSeqLengths.GetValue(bIdx);
    }
    return 0;
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline uint64_t GetActualKVSeqLength(GlobalTensor<uint64_t> &actualSeqLengths, uint32_t bIdx, const ConstInfo &constInfo, __gm__ uint8_t *keyPtr)
{
    if (constInfo.actualLenDims == 0) {
        if (!constInfo.batchContinuous) {
            return SeqLenFromTensorList<LAYOUT_T>(keyPtr, bIdx);
        }
        return constInfo.kvSeqSize;
    } 
    if (constInfo.actualLenDims == 1 || bIdx == 0) {
        return actualSeqLengths.GetValue(0);
    }
    if (constInfo.accumKVSeqFlag) {
        return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
    } else {
        return actualSeqLengths.GetValue(bIdx);
    }
    return 0;
}

} // namespace fa_base_kernel
#endif