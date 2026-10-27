#include <pybind11/pybind11.h>
#include "../registration/registration.h"

namespace py = pybind11;

PYBIND11_MODULE(_openjiuwen_ascend_ops, m) {
    m.def("register_ops", &openjiuwen_ascend_ops::register_all_ops,
          "Register all AscendC operators to PyTorch");
}
