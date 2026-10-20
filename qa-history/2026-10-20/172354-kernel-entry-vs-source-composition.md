# kernel 入口与源码组织

## 用户问题

测例真正调用的 kernel 并不是一个单独的函数或者文件，我的理解对吗

## 实际交付的完整回答

基本对，但“不是一个单独的函数”需要修正：**运行时确实启动一个具体的 kernel 函数实例；这个函数的实现由多个模板、类和头文件共同组成。**

这条路径分为三层：

| 层次 | 本例对应内容 | 作用 |
|---|---|---|
| 真正启动的函数 | `device_kernel<GemmKernel>` | GPU 的 `__global__` 入口 |
| kernel 主体 | `GemmKernel::operator()(params, smem)` | 组织线程角色、共享存储、调度、流水线和 TMEM |
| 具体计算组件 | `CollectiveMainloop`、`CollectiveEpilogue` | 完成输入搬运、MMA、融合和输出 |

入口源码其实很短，核心就是：

```cpp
// device_kernel<Operator> 内部
extern __shared__ char smem[];
Operator op;          // 本例 Operator = GemmKernel
op(params, smem);     // 进入 GemmUniversal 的编排主体
```

可直接查看[真正的入口函数](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/device_kernel.h:118)和[SM100 kernel 主体](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:405)。

所以，你找不到一个单独的 `.cu` 函数，把完整 GEMM 从头写到尾，是正常的：**测例通过模板组合出 kernel 类型，编译器再生成具体的入口与设备代码。两个 Collective 是这个 kernel 内部的协作组件。**

## 来源与验证

本轮重新读取工作树 device_kernel.h:108-125、sm100_gemm_tma_warpspecialized.hpp:399-410、432-437、gemm_universal_adapter.h:455-459，确认 Adapter 指向 device_kernel<GemmKernel>、入口构造 Operator 并调用 operator()，以及 kernel 中构造两个 Collective。

仅静态源码核对，未编译、未执行 GPU。未修改源文件。本轮没有新增图片。问答本地保存；跨电脑同步未完成。

