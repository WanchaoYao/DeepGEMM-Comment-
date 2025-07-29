#pragma once

#include "utils.cuh"

namespace deep_gemm
{

    enum class GemmType
    {
        Normal,
        GroupedContiguous,
        GroupedMasked
    };

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
    // template <
    //     GemmType kGemmType,           // GEMM类型
    //     uint32_t SHAPE_N,             // 矩阵N维度大小
    //     uint32_t BLOCK_M,             // 块的M维度大小
    //     uint32_t BLOCK_N,             // 块的N维度大小
    //     uint32_t kNumGroups,          // 组数
    //     uint32_t kNumTMAMulticast,    // TMA多播数量
    //     bool kIsTMAMulticastOnA,      // 是否在A矩阵上进行TMA多播
    //     uint32_t kNumNBlocks,         // N维度的块数
    //     uint32_t kNum1DBlocksPerGroup // 每组1D块数
    //     >
    template <GemmType kGemmType,
              uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
              uint32_t kNumGroups,
              uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
              uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
              uint32_t kNum1DBlocksPerGroup = 16>
    struct Scheduler
    {
        // int current_iter = -1;         // 当前迭代次数
        // uint32_t num_aligned_m_blocks; // 对齐的M维度块数
        // uint32_t num_blocks;           // 总块数
        // uint32_t num_blocks_in_group;  // 组内块数
        // bool is_peer_cta_alive;        // 相邻CTA是否有效
        // int *grouped_layout;           // 分组布局数组
        // uint32_t curr_group_idx;       // 当前组索引
        // uint32_t curr_cumsum;          // 当前累积和

        int current_iter = -1;
        uint32_t num_aligned_m_blocks;

        // For normal GEMM
        // Maybe not used in the masked grouped GEMM
        uint32_t num_blocks;
        uint32_t num_blocks_in_group;
        bool is_peer_cta_alive = true;

        // For grouped GEMM
        int *grouped_layout;

        // Only used for masked layout
        uint32_t curr_group_idx, curr_cumsum;

        __device__ __forceinline__ explicit Scheduler(const uint32_t &shape_m,       // M维度大小
                                                      int *grouped_layout = nullptr) // 分组布局
        {
            // 计算对齐的M维度块数
            num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
            if constexpr (kGemmType == GemmType::Normal)
            {
                num_blocks = num_aligned_m_blocks * kNumNBlocks;
            }
            else if (kGemmType == GemmType::GroupedContiguous)
            {
                num_blocks = num_aligned_m_blocks * kNumNBlocks;
                this->grouped_layout = grouped_layout;
            }
            else if (kGemmType == GemmType::GroupedMasked)
            {
                curr_group_idx = curr_cumsum = 0;
                this->grouped_layout = grouped_layout;
            }
        }

        // ReSharper disable once CppNotAllPathsReturnValue
        __device__ __forceinline__ bool is_computation_valid(const uint32_t &m_block_idx, const uint32_t &m_offset) const
        {
            if constexpr (kGemmType == GemmType::Normal)
            {
                return true;
            }
            else if constexpr (kGemmType == GemmType::GroupedContiguous)
            {
                // grouped_layout 存储每个位置的组索引
                // 负值表示该位置不参与计算
                // 非负值表示该位置属于某个组
                return __ldg(grouped_layout + m_offset + m_block_idx * BLOCK_M) >= 0;
            }
            else if constexpr (kGemmType == GemmType::GroupedMasked)
            {
                // grouped_layout 存储每组的结束位置
                // curr_group_idx 指示当前处理的组
                // 检查当前位置是否在组的范围内
                return m_offset + m_block_idx * BLOCK_M < __ldg(grouped_layout + curr_group_idx);
            }
        }

        __device__ __forceinline__ bool is_tma_multicast_valid(const uint32_t &m_block_idx) const
        {
            if (num_blocks_in_group == 1)
                return false;
            if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::GroupedMasked)
            {
                return true;
            }
            else
            {
                DG_STATIC_ASSERT(kGemmType == GemmType::GroupedContiguous, "Invalid Gemm type");
                if constexpr (kIsTMAMulticastOnA)
                {
                    return true;
                }
                else
                {
                    // __ldg：Load Global Memory Through Cache
                    // 通过 L1 缓存读取全局内存
                    // 可以提高读取性能
                    // 适用于只读数据
                    auto group_idx = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                    auto peer_group_idx = __ldg(grouped_layout + (m_block_idx ^ 1) * BLOCK_M);
                    // 检查相邻块是否在同一组
                    return group_idx == peer_group_idx;
                }
            }
        }

        __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t &num_m_blocks, const uint32_t &block_idx,
                                                               uint32_t &m_block_idx, uint32_t &n_block_idx)
        {
            DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

            // Swizzle for better L2 usages
            // 根据TMA多播模式选择主次维度
            auto primary_num_blocks = kIsTMAMulticastOnA ? kNumNBlocks : num_m_blocks;
            auto secondary_num_blocks = kIsTMAMulticastOnA ? num_m_blocks : kNumNBlocks;
            // 计算每组块数
            auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
            // 计算组索引和组内位置
            auto group_idx = block_idx / num_blocks_per_group;       // 当前组号
            auto first_block_idx = group_idx * kNum1DBlocksPerGroup; // 组起始块号
            auto in_group_idx = block_idx % num_blocks_per_group;    // 组内偏移
            // 计算组内实际块数（处理边界情况）
            num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

            // Fix unaligned TMA multicast
            // 当需要多播且组内块数为奇数时
            if (kNumTMAMulticast > 1 and num_blocks_in_group % 2 != 0)
            {
                if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks)
                {
                    // 在主要部分：调整到偶数块
                    num_blocks_in_group = num_blocks_in_group ^ 1;
                }
                else
                {
                    // 在剩余部分：调整索引和大小
                    in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                    first_block_idx += num_blocks_in_group ^ 1;
                    num_blocks_in_group = 1;
                }
            }

            // Convert to final M/N block indices
            if constexpr (kIsTMAMulticastOnA)
            {
                // A矩阵多播模式
                m_block_idx = in_group_idx / num_blocks_in_group;
                n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            }
            else
            {
                // B矩阵多播模式
                m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
                n_block_idx = in_group_idx / num_blocks_in_group;
            }
        }

        template <bool kIgnoreGroupedForGroupedContiguous = true>
        __device__ __forceinline__ uint32_t get_global_idx(const uint32_t &shape_dim, const uint32_t &block_size,
                                                           const uint32_t &block_idx, const uint32_t &m_block_idx = 0)
        {
            // 最简单的情况，直接将块索引乘以块大小，线性映射，无偏移量
            if constexpr (kGemmType == GemmType::Normal)
            {
                return block_idx * block_size;
            }
            else if constexpr (kGemmType == GemmType::GroupedContiguous)
            {
                // 计算偏移量
                auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M));
                // 计算全局索引
                return offset * shape_dim + block_idx * block_size;
            }
            else if constexpr (kGemmType == GemmType::GroupedMasked)
            {
                // curr_group_idx：当前组索引
                // shape_dim：维度大小
                // block_idx：块索引
                // block_size：块大小
                return curr_group_idx * shape_dim + block_idx * block_size;
            }
        }

        __device__ __forceinline__ bool get_next_block(uint32_t &m_block_idx, uint32_t &n_block_idx)
        {
            // current_iter: 当前迭代次数
            // gridDim.x: 网格维度
            // blockIdx.x: 当前块索引
            const auto next_block_idx = (++current_iter) * gridDim.x + blockIdx.x;

            if constexpr (kGemmType == GemmType::GroupedMasked)
            {
                uint32_t num_m_blocks;
                while (true)
                {
                    // End of the task
                    // 检查是否处理完所有组
                    if (curr_group_idx == kNumGroups)
                        return false;

                    // Within the current group
                    // 计算当前组的M维度块数
                    num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                    // 计算累积块数
                    auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                    // 检查是否在当前组内
                    if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                        break;

                    // Move to check the next group
                    // 移动到下一组
                    curr_group_idx++, curr_cumsum = current_m_block_cumsum;
                }

                // 计算交错块索引
                get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
            }
            else
            {
                // 检查是否超出总块数
                if (next_block_idx >= num_blocks)
                    return false;

                // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
                // 检查相邻CTA是否有效
                is_peer_cta_alive = kNumNBlocks % kNumTMAMulticast == 0 or          // Always aligned on N (constant bypass) // N维度对齐
                                    num_aligned_m_blocks % kNumTMAMulticast == 0 or // Always aligned on M (constant bypass) // M维度对齐
                                    (next_block_idx ^ 1) < num_blocks;              // Peer CTA in bound // 相邻块在范围内
                // 计算交错块索引
                get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
            }
            return true;
        }
    };

#pragma clang diagnostic pop

} // namespace deep_gemm
