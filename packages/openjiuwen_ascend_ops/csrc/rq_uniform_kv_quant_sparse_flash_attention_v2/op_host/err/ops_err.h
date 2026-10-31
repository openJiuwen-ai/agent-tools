/**
 * Local host-log shim for Schema A kv_quant tiling.
 * Matches ops-tf / newer log.h OP_LOGE_FOR_* call arities used by QSFA tiling.
 */
#ifndef RQ_UNIFORM_KV_QUANT_SFA_OPS_ERR_H
#define RQ_UNIFORM_KV_QUANT_SFA_OPS_ERR_H

#include <string>
#include "exe_graph/runtime/shape.h"

#include "log/ops_log.h"

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                  \
        if (cond) {                       \
            log_func;                     \
            expr;                         \
        }                                 \
    } while (0)
#endif

#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                    \
    do {                                                            \
        if ((ptr) == nullptr) {                                     \
            OPS_LOG_E((context)->GetNodeName(), "%s is null", #ptr); \
            return ge::GRAPH_FAILED;                                \
        }                                                           \
    } while (0)
#endif

#ifndef OP_LOGI
#define OP_LOGI(opname, ...) OPS_LOG_I(opname, __VA_ARGS__)
#endif
#ifndef OP_LOGW
#define OP_LOGW(opname, ...) OPS_LOG_W(opname, __VA_ARGS__)
#endif
#ifndef OP_LOGE
#define OP_LOGE(opname, ...) OPS_LOG_E(opname, __VA_ARGS__)
#endif

#ifndef OPS_REPORT_VECTOR_INNER_ERR
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, __VA_ARGS__)
#endif
#ifndef OPS_REPORT_CUBE_INNER_ERR
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, __VA_ARGS__)
#endif

// Local stand-in for Ops::Base::ToString used by QSFA tiling (CANN log.h not present).
namespace Ops {
namespace Base {
inline std::string ToString(const gert::Shape& shape)
{
    std::string out = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape.GetDim(i));
    }
    out += "]";
    return out;
}
}  // namespace Base
}  // namespace Ops

// Helpers: accept const char* or std::string temporaries.
#define _SFA_CSTR(x) (std::string(x).c_str())

#ifndef OP_LOGE_FOR_INVALID_SHAPE
#define OP_LOGE_FOR_INVALID_SHAPE(opName, paramName, incorrectShape, correctShape) \
    OPS_LOG_E(opName, "param %s shape %s (expect %s)", _SFA_CSTR(paramName),       \
              _SFA_CSTR(incorrectShape), _SFA_CSTR(correctShape))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, paramName, incorrectShape, reason) \
    OPS_LOG_E(opName, "param %s shape %s: %s", _SFA_CSTR(paramName),                     \
              _SFA_CSTR(incorrectShape), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, paramNames, incorrectShapes, reason) \
    OPS_LOG_E(opName, "params %s shapes %s: %s", _SFA_CSTR(paramNames),                     \
              _SFA_CSTR(incorrectShapes), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, paramName, incorrectDim, reason) \
    OPS_LOG_E(opName, "param %s dim %s: %s", _SFA_CSTR(paramName),                        \
              _SFA_CSTR(incorrectDim), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, paramName, incorrectSize, reason) \
    OPS_LOG_E(opName, "param %s size %s: %s", _SFA_CSTR(paramName),                          \
              _SFA_CSTR(incorrectSize), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, paramName, incorrectDtype, reason) \
    OPS_LOG_E(opName, "param %s dtype %s: %s", _SFA_CSTR(paramName),                      \
              _SFA_CSTR(incorrectDtype), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, paramNames, incorrectDtype, reason) \
    OPS_LOG_E(opName, "params %s dtype %s: %s", _SFA_CSTR(paramNames),                      \
              _SFA_CSTR(incorrectDtype), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opName, paramName, incorrectValue, correctValue) \
    OPS_LOG_E(opName, "param %s value %s (expect %s)", _SFA_CSTR(paramName),       \
              _SFA_CSTR(incorrectValue), _SFA_CSTR(correctValue))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, paramName, incorrectValue, reason) \
    OPS_LOG_E(opName, "param %s value %s: %s", _SFA_CSTR(paramName),                      \
              _SFA_CSTR(incorrectValue), _SFA_CSTR(reason))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramNames, incorrectValues, reason) \
    OPS_LOG_E(opName, "params %s values %s: %s", _SFA_CSTR(paramNames),                      \
              _SFA_CSTR(incorrectValues), _SFA_CSTR(reason))
#endif

// 3-arg form used by QSFA tiling: (opName, paramName, reason)
#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, paramName, reason) \
    OPS_LOG_E(opName, "arg %s: %s", _SFA_CSTR(paramName), _SFA_CSTR(reason))
#endif

#endif  // RQ_UNIFORM_KV_QUANT_SFA_OPS_ERR_H
