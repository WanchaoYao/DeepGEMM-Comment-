#pragma once

#ifdef __CLION_IDE__

__host__ __device__ __forceinline__ void host_device_printf(const char *format, ...)
{
    // volatile 关键字的作用：
    // 防止编译器优化
    // 保证指令顺序
    // 确保内存访问
    // 维持原始语义
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                             \
    do                                                                                     \
    {                                                                                      \
        if (not(cond))                                                                     \
        {                                                                                  \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;");                                                                  \
        }                                                                                  \
    } while (0)
#endif

// DG_DEVICE_ASSERT(value > 0);
// 如果条件失败：
// 1. 打印错误信息
// 2. 触发 trap 中断
// 3. 停止执行

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b)
{
    return b == 0 ? a : constexpr_gcd(b, a % b);
}
