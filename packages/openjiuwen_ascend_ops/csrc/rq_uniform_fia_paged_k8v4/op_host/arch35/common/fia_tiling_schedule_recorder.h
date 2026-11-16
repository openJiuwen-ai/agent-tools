/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FIA_TILING_SCHEDULE_RECORDER_H
#define FIA_TILING_SCHEDULE_RECORDER_H

#include <cstdint>
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
// TilingContext has no schedule-mode getter. Observe only the active context;
// scopes restore the previous recorder, including when tiling exits early.
class FiaTilingScheduleRecorder {
public:
    class Scope {
    public:
        explicit Scope(gert::TilingContext *context)
            : context_(context),
              previous_(Current())
        {
            Current() = this;
        }
        ~Scope()
        {
            Current() = previous_;
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
        bool Get(uint32_t &mode) const
        {
            mode = mode_;
            return set_;
        }

    private:
        friend class FiaTilingScheduleRecorder;
        gert::TilingContext *context_;
        Scope *previous_;
        bool set_ = false;
        uint32_t mode_ = 0;
    };

    static ge::graphStatus Set(gert::TilingContext *context, uint32_t mode)
    {
        const auto status = context->SetScheduleMode(mode);
        auto *scope = Current();
        if (status == ge::GRAPH_SUCCESS && scope != nullptr && scope->context_ == context) {
            scope->set_ = true;
            scope->mode_ = mode;
        }
        return status;
    }

private:
    static Scope *&Current()
    {
        static thread_local Scope *scope = nullptr;
        return scope;
    }
};
} // namespace optiling
#endif
