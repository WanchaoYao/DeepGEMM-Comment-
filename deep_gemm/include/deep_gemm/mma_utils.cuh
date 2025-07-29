#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

#include "utils.cuh"

namespace deep_gemm
{

    template <typename dtype_t>
    struct SM90_U32x2_STSM_N
    {
        __device__ __forceinline__ static void
        copy(dtype_t src_0, dtype_t src_1, void *smem_dst)
        {
            const uint32_t src[2] = {*reinterpret_cast<uint32_t *>(&src_0), *reinterpret_cast<uint32_t *>(&src_1)};
            asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"l"(smem_dst), "r"(src[0]), "r"(src[1]));
        }
    };

    template <typename dtype_t>
    struct SM90_U32x4_STSM_N
    {
        __device__ __forceinline__ static void
        copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void *smem_dst)
        {
            const uint32_t src[4] = {*reinterpret_cast<uint32_t *>(&src_0), *reinterpret_cast<uint32_t *>(&src_1),
                                     *reinterpret_cast<uint32_t *>(&src_2), *reinterpret_cast<uint32_t *>(&src_3)};
            asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(smem_dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
        }
    };

    // 告诉编译器：
    // 1. 这条指令可能改变内存
    // 2. 不要重排这条指令周围的内存操作
    // 3. 需要刷新对内存状态的所有假设
    __forceinline__ __device__ void warpgroup_arrive()
    {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    }

    __forceinline__ __device__ void warpgroup_commit_batch()
    {
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    }

    __forceinline__ __device__ void warpgroup_fence_operand(float &reg)
    {
        asm volatile("" : "+f"(reg)::"memory"); // 空指令但有内存屏障效果，同时约束寄存器和内存
    }

    // 获取当前线程在 warp 中的索引
    __forceinline__ __device__ uint32_t get_lane_id()
    {
        // uint32_t lane_id;
        // asm：内联汇编关键字
        // mov.u32：32位无符号移动指令
        // %laneid：特殊寄存器，包含当前线程的 lane ID（0-31）
        // %0：第一个输出操作数的占位符
        // "=r"：表示输出到寄存器
        // (lane_id)：存储结果的 C++ 变量
        asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
        return lane_id;
    }

    __device__ __forceinline__ uint32_t ld_shared(const uint32_t *__restrict__ ptr)
    {
        uint32_t ret;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
        return ret;
    }

    __device__ __forceinline__ int4 ld_shared(const int4 *__restrict__ ptr)
    {
        int4 ret;
        asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
        return ret;
    }

    __device__ __forceinline__ float ld_shared(const float *__restrict__ ptr)
    {
        float ret;
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
        return ret;
    }

    __device__ __forceinline__ float2 ld_shared(const float2 *__restrict__ ptr)
    {
        float2 ret;
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(ptr));
        return ret;
    }

    __device__ __forceinline__ void st_shared(const float *ptr, float val)
    {
        asm volatile("st.shared.f32 [%0], %1;" ::"l"(ptr), "f"(val));
    }

    __device__ __forceinline__ void st_shared(const uint32_t *ptr, uint32_t val)
    {
        asm volatile("st.shared.u32 [%0], %1;" ::"l"(ptr), "r"(val));
    }

    __device__ __forceinline__ void st_shared(const float2 *ptr, float2 val)
    {
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"l"(ptr), "f"(val.x), "f"(val.y));
    }

    template <int N>
    __device__ void warpgroup_wait()
    {
        DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
        // 等待指定批次的 WGMMA 操作完成
        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
    }

    union GmmaDescriptor
    {
        __host__ __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}

        __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}

        __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept : desc_(t.desc_) {}

        __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept : desc_(t.desc_) {}

        __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor const &t) noexcept
        {
            desc_ = t.desc_;
            return *this;
        }

        __host__ __device__ constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept
        {
            desc_ = t.desc_;
            return *this;
        }

        uint64_t desc_;
        uint32_t reg32_[2];
        uint16_t reg16_[4];

        // 位域结构，精确控制每个字段的位数，节省内存空间，用于硬件级别的控制
        // GMMA 指令期望的 uint64_t 布局
        // |63    54|53  52|51  48|47    32|31    16|15     0|
        // |--------+-------+-------+---------+--------+--------|
        // |reserved|layout |base   |stride  |leading |start   |
        // |        |type   |offset |offset  |offset  |address |
        // CUTE 不需要解析，直接传给硬件指令
        // asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.f16 {%0,%1}, %2, %3;\n"
        //     : "=r"(d[0]), "=r"(d[1])  // 输出
        //     : "l"(desc_a), "l"(desc_b) // 输入的 uint64_t
        // );
        struct
        {
            uint16_t start_address_ : 14;       // 起始地址
            uint16_t leading_byte_offset_ : 14; // 前导字节偏移
            uint16_t stride_byte_offset_ : 14;  // 步长字节偏移
            uint8_t base_offset_ : 3;           // 基础偏移
            uint8_t layout_type_ : 2;           // 布局类型
        } bitfield;

        // Decay to an `uint64_t`
        // 定义了隐式转换操作符到 uint64_t
        __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
    };

    // 这是一个用于 FP8 (8位浮点) WGMMA 操作的模板类。
    template <class PointerType>
    __device__ GmmaDescriptor make_smem_desc(PointerType smem_ptr, int layout_type,
                                             int leading_byte_offset = 0,
                                             int stride_byte_offset = 1024)
    {
        GmmaDescriptor desc;
        auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        // 表示以16字节为单位的地址
        // 为什么要除以16：Hopper 的 GMMA 操作要求16字节对齐，地址必须是16的倍数，节省描述符中的比特位，硬件级别的要求
        // 实际例子：
        //     uint_ptr = 0x1000;    // 假设共享内存地址是：0x1000
        //     start_address_ = 0x1000 >> 4; // = 0x100
        //     real_address = start_address_ << 4; // 实际访问时硬件自动乘以16
        // 简单理解：
        //     GmmaDescriptor 就像填表
        //     表格格式由硬件定义
        //     CUTE 只负责传递表格
        //     硬件知道如何读表格
        desc.bitfield.start_address_ = uint_ptr >> 4;
        desc.bitfield.layout_type_ = layout_type;
        desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
        desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
        desc.bitfield.base_offset_ = 0;
        return desc;
    }

    template <int N_, typename MMA>
    struct FP8MMA
    {

        template <size_t... Idx>
        __forceinline__ __device__ static void call_fma_impl(uint64_t const &desc_a, uint64_t const &desc_b, float *d, bool scale_d, std::index_sequence<Idx...>)
        {
            using namespace cute::SM90::GMMA;
            // 展开为：MMA::fma(desc_a, desc_b, d[0], d[1], d[2], ..., scale)
            MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
        }

        __forceinline__ __device__ static void wgmma(uint64_t const &desc_a, uint64_t const &desc_b, float *d, bool scale_d)
        {
            // std::make_index_sequence 是 C++ 的模板元编程工具，用于生成编译时的整数序列。在这个例子中，它用于展开矩阵乘法累加器的访问。
            // make_index_sequence<4> 生成序列 <0,1,2,3>
            // std::make_index_sequence<4> // 等价于 std::index_sequence<0,1,2,3>
            call_fma_impl(desc_a, desc_b, d, scale_d, std::make_index_sequence<N_ / 2>{});
        }

        static constexpr int M = 64;                  // 矩阵高度
        static constexpr int N = N_;                  // 矩阵宽度
        static constexpr int K = 32;                  // 内积维度
        static constexpr int kNumAccum = M * N / 128; // 累加器数量
    };

    // 一个模板结构体，根据不同的 N 值选择相应的 CUDA MMA (Matrix Multiply-Accumulate) 操作
    template <int N>
    struct FP8MMASelector
    {
        // 每个 MMA 操作都是 64xNx32 大小的矩阵乘法，这些 MMA 操作是 cutlass 定义的
        static constexpr auto select_mma()
        {
            using namespace cute::SM90::GMMA;
            if constexpr (N == 16)
                return MMA_64x16x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 24)
                return MMA_64x24x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 32)
                return MMA_64x32x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 40)
                return MMA_64x40x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 48)
                return MMA_64x48x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 56)
                return MMA_64x56x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 64)
                return MMA_64x64x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 72)
                return MMA_64x72x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 80)
                return MMA_64x80x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 88)
                return MMA_64x88x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 96)
                return MMA_64x96x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 104)
                return MMA_64x104x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 112)
                return MMA_64x112x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 120)
                return MMA_64x120x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 128)
                return MMA_64x128x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 136)
                return MMA_64x136x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 144)
                return MMA_64x144x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 152)
                return MMA_64x152x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 160)
                return MMA_64x160x32_F32E4M3E4M3_SS_TN();
            if constexpr (N == 192)
                return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        }

        static constexpr auto select_type()
        {
            // decltype 是用来推导表达式类型的关键字
            // 获取 select_mma() 返回值的类型
            return FP8MMA<N, decltype(select_mma())>();
        }

        // 获取 select_type() 返回值的类型
        using type = decltype(select_type());
    };

    enum class Layout
    {
        RowMajor,
        ColMajor
    };

    // warp group数目
    __device__ __host__ constexpr int get_num_math_warpgroups(int block_m)
    {
        return block_m == 64 ? 1 : 2;
    }

    template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
    __device__ __host__ constexpr int get_num_threads_per_sm(int block_m)
    {
        DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
        // 总的thread数目 = warp_group数目 * 每个group计算thread数目 + TMA_thread数目
        return get_num_math_warpgroups(block_m) * kNumMathThreadsPerGroup + kNumTMAThreads;
    }

} // namespace deep_gemm
