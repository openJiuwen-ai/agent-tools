/*
 * Same-shape BenchUs for rq_uniform_pack_k8v4.
 *
 * A2: --quant-mode 0 (UNIFORM, rotation required)
 * A5: --quant-mode 1 (TQ_K8_FP8, no rotation)
 *
 * The timed region is GetWorkspaceSize + aclnn launch, warmup excluded,
 * one stream sync after the repeat loop. Printed as avg_us.
 */

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include "aclnn_rq_uniform_pack_k8v4.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kBlockRows = 16;
constexpr int64_t kQuantUniform = 0;
constexpr int64_t kQuantFp8 = 1;

struct Shape {
    int64_t tokens = 128;
    int64_t heads = 8;
    int64_t head_size = 128;
    int64_t block_size = 16;
};

struct Options {
    int device = 0;
    int64_t quant_mode = -1;
    int warmup = 10;
    int repeat = 100;
    bool all_perf_shapes = false;
    Shape shape {};
};

[[noreturn]] void Fail(const std::string& message)
{
    throw std::runtime_error(message);
}

void CheckAcl(aclError ret, const std::string& what)
{
    if (ret != ACL_SUCCESS) {
        Fail(what + " failed, ret=" + std::to_string(ret));
    }
}

void CheckAclnn(aclnnStatus ret, const std::string& what)
{
    if (ret != ACL_SUCCESS) {
        const char* recent = aclGetRecentErrMsg();
        std::string msg = what + " failed, ret=" + std::to_string(ret);
        if (recent != nullptr && recent[0] != '\0') {
            msg += ", recent=";
            msg += recent;
        }
        Fail(msg);
    }
}

int64_t ParseInt(const char* text, const char* name)
{
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        Fail(std::string("invalid integer for ") + name + ": " + text);
    }
    return static_cast<int64_t>(value);
}

void PrintUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --quant-mode 0|1 [options]\n"
        << "  --quant-mode 0   A2 UNIFORM (fp16 key/value + [D,D] rotation)\n"
        << "  --quant-mode 1   A5 TQ_K8_FP8 (fp16 key/value, no rotation)\n"
        << "  --tokens N       default 128\n"
        << "  --heads N        default 8\n"
        << "  --head-size N    128 or 256, default 128\n"
        << "  --block-size N   multiple of 16, default 16\n"
        << "  --all-perf-shapes  run (32,2), (128,8), (256,16) at head-size/block-size\n"
        << "  --warmup N       default 10\n"
        << "  --repeat N       default 100\n"
        << "  --device N       default 0\n";
}

Options ParseArgs(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                Fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--quant-mode") {
            opt.quant_mode = ParseInt(need("--quant-mode"), "--quant-mode");
        } else if (arg == "--tokens") {
            opt.shape.tokens = ParseInt(need("--tokens"), "--tokens");
        } else if (arg == "--heads") {
            opt.shape.heads = ParseInt(need("--heads"), "--heads");
        } else if (arg == "--head-size") {
            opt.shape.head_size = ParseInt(need("--head-size"), "--head-size");
        } else if (arg == "--block-size") {
            opt.shape.block_size = ParseInt(need("--block-size"), "--block-size");
        } else if (arg == "--warmup") {
            opt.warmup = static_cast<int>(ParseInt(need("--warmup"), "--warmup"));
        } else if (arg == "--repeat") {
            opt.repeat = static_cast<int>(ParseInt(need("--repeat"), "--repeat"));
        } else if (arg == "--device") {
            opt.device = static_cast<int>(ParseInt(need("--device"), "--device"));
        } else if (arg == "--all-perf-shapes") {
            opt.all_perf_shapes = true;
        } else {
            Fail("unknown argument: " + arg);
        }
    }
    if (opt.quant_mode != kQuantUniform && opt.quant_mode != kQuantFp8) {
        Fail("--quant-mode must be 0 (A2 UNIFORM) or 1 (A5 FP8)");
    }
    if (opt.shape.tokens <= 0 || opt.shape.heads <= 0 || opt.warmup < 0 || opt.repeat <= 0) {
        Fail("tokens/heads/repeat must be positive; warmup must be >= 0");
    }
    if (opt.shape.head_size != 128 && opt.shape.head_size != 256) {
        Fail("--head-size must be 128 or 256");
    }
    if (opt.shape.block_size <= 0 || opt.shape.block_size % kBlockRows != 0) {
        Fail("--block-size must be a positive multiple of 16");
    }
    return opt;
}

uint16_t FloatToHalfBits(float value)
{
    union {
        float f;
        uint32_t u;
    } in {};
    in.f = value;
    const uint32_t sign = (in.u >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((in.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = in.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    explicit DeviceBuffer(size_t nbytes) { Reset(nbytes); }
    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void Reset(size_t nbytes)
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
        bytes = nbytes;
        if (bytes != 0) {
            CheckAcl(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        }
    }

    void CopyFromHost(const void* host, size_t nbytes)
    {
        CheckAcl(aclrtMemcpy(ptr, bytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy");
    }

    void Zero()
    {
        if (bytes != 0) {
            CheckAcl(aclrtMemset(ptr, bytes, 0, bytes), "aclrtMemset");
        }
    }
};

struct Tensor {
    aclTensor* tensor = nullptr;
    ~Tensor()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor() = default;

    void Reset(void* data, aclDataType dtype, const std::vector<int64_t>& dims)
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        std::vector<int64_t> strides(dims.size(), 1);
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = dims[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
        }
        int64_t storage = 1;
        for (int64_t d : dims) {
            storage *= d;
        }
        tensor = aclCreateTensor(
            dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
            &storage, 1, data);
        if (tensor == nullptr) {
            Fail("aclCreateTensor returned null");
        }
    }
};

int64_t VecPerCore(int64_t n_vec)
{
    if (n_vec >= 128) {
        return 128;
    }
    int64_t v = n_vec <= 16 ? 2 : ((n_vec + 15) / 16) * 16;
    return v == 0 ? 16 : v;
}

void Launch(const Shape& shape, int64_t quant_mode, Tensor& key, Tensor& value, Tensor* rotation,
            Tensor& slots, Tensor& qsl, Tensor& key_cache, Tensor& value_cache, int64_t num_blocks,
            DeviceBuffer& workspace, aclrtStream stream)
{
    const int64_t n_vec = shape.tokens * shape.heads;
    const int64_t key_stride_token = shape.heads * shape.head_size;
    const int64_t key_stride_head = shape.head_size;
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CheckAclnn(
        aclnnRqUniformPackK8v4GetWorkspaceSize(
            key.tensor,
            value.tensor,
            rotation == nullptr ? nullptr : rotation->tensor,
            slots.tensor,
            qsl.tensor,
            n_vec,
            VecPerCore(n_vec),
            shape.heads,
            shape.block_size,
            num_blocks,
            1,
            key_stride_token,
            key_stride_head,
            key_stride_token,
            key_stride_head,
            0,
            0,
            quant_mode,
            key_cache.tensor,
            value_cache.tensor,
            &workspace_size,
            &executor),
        "aclnnRqUniformPackK8v4GetWorkspaceSize");
    if (workspace.bytes < workspace_size) {
        workspace.Reset(workspace_size);
    }
    CheckAclnn(
        aclnnRqUniformPackK8v4(workspace.ptr, workspace_size, executor, stream),
        "aclnnRqUniformPackK8v4");
}

double BenchUs(const Shape& shape, int64_t quant_mode, Tensor& key, Tensor& value, Tensor* rotation,
               Tensor& slots, Tensor& qsl, Tensor& key_cache, Tensor& value_cache, int64_t num_blocks,
               int warmup, int repeat, aclrtStream stream)
{
    DeviceBuffer workspace(0);
    for (int i = 0; i < warmup; ++i) {
        Launch(shape, quant_mode, key, value, rotation, slots, qsl, key_cache, value_cache,
               num_blocks, workspace, stream);
    }
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream warmup");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        Launch(shape, quant_mode, key, value, rotation, slots, qsl, key_cache, value_cache,
               num_blocks, workspace, stream);
    }
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream repeat");
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / static_cast<double>(repeat);
}

void RunShape(const Options& opt, const Shape& shape, aclrtStream stream)
{
    const int64_t num_blocks = (shape.tokens + shape.block_size - 1) / shape.block_size;
    const int64_t key_row = opt.quant_mode == kQuantFp8 ? shape.head_size : (shape.head_size + 4);
    const int64_t key_span = (shape.block_size / kBlockRows) * (kBlockRows * key_row);
    const int64_t val_span = (shape.block_size / kBlockRows) * (kBlockRows * (shape.head_size / 2 + 4));
    const int64_t n_elem = shape.tokens * shape.heads * shape.head_size;

    std::mt19937 rng(20261112);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<uint16_t> key_host(static_cast<size_t>(n_elem));
    std::vector<uint16_t> value_host(static_cast<size_t>(n_elem));
    for (int64_t i = 0; i < n_elem; ++i) {
        key_host[static_cast<size_t>(i)] = FloatToHalfBits(dist(rng));
        value_host[static_cast<size_t>(i)] = FloatToHalfBits(dist(rng));
    }
    std::vector<int32_t> slots_host(static_cast<size_t>(shape.tokens));
    for (int64_t i = 0; i < shape.tokens; ++i) {
        slots_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    const int32_t qsl_host[2] = {0, static_cast<int32_t>(shape.tokens)};

    DeviceBuffer key_dev(key_host.size() * sizeof(uint16_t));
    DeviceBuffer value_dev(value_host.size() * sizeof(uint16_t));
    DeviceBuffer slots_dev(slots_host.size() * sizeof(int32_t));
    DeviceBuffer qsl_dev(sizeof(qsl_host));
    DeviceBuffer key_cache_dev(static_cast<size_t>(num_blocks * shape.heads * key_span));
    DeviceBuffer value_cache_dev(static_cast<size_t>(num_blocks * shape.heads * val_span));
    key_dev.CopyFromHost(key_host.data(), key_dev.bytes);
    value_dev.CopyFromHost(value_host.data(), value_dev.bytes);
    slots_dev.CopyFromHost(slots_host.data(), slots_dev.bytes);
    qsl_dev.CopyFromHost(qsl_host, qsl_dev.bytes);
    key_cache_dev.Zero();
    value_cache_dev.Zero();

    Tensor key;
    Tensor value;
    Tensor slots;
    Tensor qsl;
    Tensor key_cache;
    Tensor value_cache;
    key.Reset(key_dev.ptr, ACL_FLOAT16, {shape.tokens, shape.heads, shape.head_size});
    value.Reset(value_dev.ptr, ACL_FLOAT16, {shape.tokens, shape.heads, shape.head_size});
    slots.Reset(slots_dev.ptr, ACL_INT32, {shape.tokens});
    qsl.Reset(qsl_dev.ptr, ACL_INT32, {2});
    key_cache.Reset(key_cache_dev.ptr, ACL_UINT8, {num_blocks, shape.heads, key_span});
    value_cache.Reset(value_cache_dev.ptr, ACL_UINT8, {num_blocks, shape.heads, val_span});

    DeviceBuffer rotation_dev(0);
    Tensor rotation;
    Tensor* rotation_ptr = nullptr;
    if (opt.quant_mode == kQuantUniform) {
        std::vector<uint16_t> rot(static_cast<size_t>(shape.head_size * shape.head_size), 0);
        for (int64_t i = 0; i < shape.head_size; ++i) {
            rot[static_cast<size_t>(i * shape.head_size + i)] = FloatToHalfBits(1.0f);
        }
        rotation_dev.Reset(rot.size() * sizeof(uint16_t));
        rotation_dev.CopyFromHost(rot.data(), rotation_dev.bytes);
        rotation.Reset(rotation_dev.ptr, ACL_FLOAT16, {shape.head_size, shape.head_size});
        rotation_ptr = &rotation;
    }

    const double avg_us = BenchUs(
        shape, opt.quant_mode, key, value, rotation_ptr, slots, qsl, key_cache, value_cache,
        num_blocks, opt.warmup, opt.repeat, stream);
    std::cout << "rq_uniform_pack_k8v4"
              << " quant_mode=" << opt.quant_mode
              << " tokens=" << shape.tokens
              << " heads=" << shape.heads
              << " head_size=" << shape.head_size
              << " block_size=" << shape.block_size
              << " avg_us=" << avg_us
              << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = ParseArgs(argc, argv);
        CheckAcl(aclInit(nullptr), "aclInit");
        CheckAcl(aclrtSetDevice(opt.device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
        if (opt.all_perf_shapes) {
            const Shape cases[] = {
                {32, 2, opt.shape.head_size, opt.shape.block_size},
                {128, 8, opt.shape.head_size, opt.shape.block_size},
                {256, 16, opt.shape.head_size, opt.shape.block_size},
            };
            for (const Shape& shape : cases) {
                RunShape(opt, shape, stream);
            }
        } else {
            RunShape(opt, opt.shape, stream);
        }
        CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
        CheckAcl(aclrtResetDevice(opt.device), "aclrtResetDevice");
        CheckAcl(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "rq_uniform_pack_k8v4_perf failed: " << ex.what() << "\n";
        return 1;
    }
}
