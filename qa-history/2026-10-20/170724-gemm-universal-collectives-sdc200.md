# GEMM Collective 编排与 SDC200 适配

记录时间：2026-10-20T17:07:24.681642+08:00

## 用户问题

详细介绍 GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue> 把两个 Collective 组织成可启动 kernel（int×4 是运行期 M/N/K/ 批次数）；GemmUniversalAdapter 提供 device 级参数打包与 launch；最后 TestAll<Gemm>() 跑多组问题规模 × 布局组合的标准测试，EXPECT_TRUE(pass) 断言 这个完整的过程，以及这俩collective 是如何组织详细具体过程，生成对应的流程图片和卡片直观说明。同时给出若需适配一个对标SM100 的私有架构SDC200，整个流程中有哪些文件和API需要适配，给出一张完备详细的适配参考表格，包含整个流程中可以复用的文件和API，以及需要适配的API和文件

## 实际交付的完整回答

已完成源码核对、详细说明、四组图稿和 **68 项 SDC200 适配参考表**。

建议从[完整阅读页](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/index.html)开始；另有[详细 Markdown](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/report.md)、[完整适配表](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/sdc200-adaptation.md)和[可筛选 CSV](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/sdc200-adaptation.csv)。

核心过程是：Builder 生成两个 Collective 类型 → `GemmUniversal` 组织参数、共享存储、warp 分工、调度与同步 → Adapter 转换参数并启动 `device_kernel` → Mainloop 将结果累加到 TMEM → Epilogue 等待结果、融合并写 D → Testbed 同步、计算主机参考并比较 → `EXPECT_TRUE(pass)` 登记结果。

两个 Collective 的交接核心是 **相同 tile 坐标、同一 TMEM 累加器视图，以及 producer/consumer 流水线协议**。文档逐项展开了初始化、A/B 搬运、MMA、累加器交接、输出与退出过程，并附实际源码位置。

| 图稿 | 图片 | 可编辑源文件 |
|---|---|---|
| 完整调用流程 | [PNG](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/01-end-to-end.png) | [Draw.io](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/01-end-to-end.drawio) |
| 两个 Collective 的协作 | [PNG](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/02-collective-handoff.png) | [Draw.io](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/02-collective-handoff.drawio) |
| 九张核心卡片 | [PNG](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/03-concept-cards.png) | [Draw.io](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/03-concept-cards.drawio) |
| SDC200 适配地图 | [PNG](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/04-sdc200-porting.png) | [Draw.io](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/04-sdc200-porting.drawio) |

适配表逐项列出文件、API、可复用部分、需要适配的实现、建议落点与验收条件，覆盖 Builder、两个 Collective、Kernel、Adapter、调度、TMA、TMEM、MMA、同步、运行时、编译器和测试。

源码核对还确认了三点：

- 单次 `TestAll<Gemm>()` 固定矩阵布局类型；多布局由不同 Gemm 实例覆盖。
- 当前工作树的 `TestAll` 定义缺失函数名；而 Testbed 对某些不支持输入会返回 `true`，所以需要单独统计实际执行与跳过。
- 已有 SDC200 MMA builtin 包装，但 cluster 文件仍含 NVIDIA PTX，整条路径的适配尚不能据此认定完成。

[下载完整资料包](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/gemm-universal-sdc200.zip)。本轮完成静态源码与文档图稿检查，未进行 CUDA/SDC 编译、设备运行、数值或性能验证。图与源文件主要结构一致，字体和抗锯齿可能因渲染器略有差异。问答已本地保存，跨电脑同步未完成。

<oai-mem-citation>
<citation_entries>
MEMORY.md:50-63|note=[prior test context and delivery conventions rechecked against live source]
MEMORY.md:171-184|note=[SDC source and builtin locations rechecked live]
skills/cxx-cute-documentation-delivery/SKILL.md:17-33|note=[source grounded Chinese documentation and editable diagrams]
</citation_entries>
<rollout_ids>
01a15532-fae2-7c80-a55f-d659e5ed5ff5
01a05b16-be32-78b1-9419-bfc7044070f2
</rollout_ids>
</oai-mem-citation>

## 详细成果

- [01-end-to-end.drawio](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/01-end-to-end.drawio)
- [01-end-to-end.png](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/01-end-to-end.png)
- [02-collective-handoff.drawio](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/02-collective-handoff.drawio)
- [02-collective-handoff.png](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/02-collective-handoff.png)
- [03-concept-cards.drawio](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/03-concept-cards.drawio)
- [03-concept-cards.png](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/03-concept-cards.png)
- [04-sdc200-porting.drawio](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/04-sdc200-porting.drawio)
- [04-sdc200-porting.png](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/04-sdc200-porting.png)
- [final-response.md](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/final-response.md)
- [index.html](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/index.html)
- [qa-record-path.txt](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/qa-record-path.txt)
- [report.md](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/report.md)
- [sdc200-adaptation.csv](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/sdc200-adaptation.csv)
- [sdc200-adaptation.html](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/sdc200-adaptation.html)
- [sdc200-adaptation.md](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/sdc200-adaptation.md)
- [source-evidence.json](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/source-evidence.json)
- [source-excerpts.txt](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/source-excerpts.txt)
- [validation.json](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/gemm-universal-sdc200/validation.json)

## 完整技术说明

# 从两个 Collective 到可启动 GEMM：SM100 源码链与 SDC200 适配指南

本文对应本项目此前的 FP16 输入、FP32 累加/输出、`ElementC=void`、tile `64×128×64`、cluster `1×4×1`、1SM 测例。先解释这条具体路径，再区分可泛化的接口与 SM100 专用实现。完整逐项适配清单见 [SDC200 适配参考表](sdc200-adaptation.md)，可筛选版本见 [CSV 表格](sdc200-adaptation.csv)。

## 0. 最重要的结论与证据边界

`GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue>` 通过模板特化选中 kernel 编排器。编排器把两个 Collective 的类型、参数、共享存储、线程角色、调度器和同步协议组合起来。真正的 `__global__` 入口是 `device_kernel<GemmKernel>`，入口内部构造 kernel functor 并调用其 `operator()(Params, smem)`。

两个 Collective 运行于同一次 kernel launch。Mainloop 内部又分搬运和 MMA 角色；Epilogue 内部又分可选 C/辅助输入搬运和输出处理角色。它们通过 **相同的输出 tile 坐标、同一份 TMEM 累加器视图、AccumulatorPipeline 的生产/消费协议** 衔接，整个过程不是一次 host 端 `Mainloop(); Epilogue();` 的串行调用。

四个 `int` 是运行时 `(M,N,K,L)` 的类型，L 是批次数。主循环 tile、CTA cluster、Atom shape、epilogue subtile 均为另一层概念。

**需要修正原问题的一点：单次 `TestAll<Gemm>()` 不枚举 A/B/C/D 的 RowMajor/ColumnMajor 类型组合。** 这些布局已在 `Gemm` 的类型中固定。不同布局组合由其他 `TEST`/类型实例化覆盖；本次 `TestAll` 枚举问题规模、raster order、swizzle 和 splits 等候选参数。

### 本次证据来源

- SM100 源码：`D:/000_SDC/001TASK/02_Research/cutlass`，HEAD `f94ec46f4f63f96003d6cfdf2014731e7672c281`。工作树存在既有修改，本文行号以本轮读取的工作树为准。
- SDC200 参考树：`D:/lvz/sdc/cutlass`，HEAD `dc45f979ae336a235da1676b311f35efeb30149a`。仅抽查相关包装和描述符，未做全仓适配完成度审计。
- SDC builtin 声明：`D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h`。
- 当前 `gemm_testbed_3x.hpp:3828` 为 `bool  (...)`，缺失 `TestAll` 名称。已对照同一 HEAD 提交内容确认原函数为 `bool TestAll(...)`。以下描述其函数体的预期语义，**不表示当前工作树可直接编译**，本次没有修复或改动源代码。
- 本机 PATH 未发现 `nvcc`、`nvidia-smi`。本次只完成源码核对、文档、图稿与文件检查，没有 CUDA/SDC 编译、GPU/模拟器运行、数值通过、机器码或性能验证。
- NVIDIA 官方资料也将 kernel 层定义为两个 Collective 的组合，再由 host adapter 包装；具体 SM100 实现和行号仍以本地版本为准：[CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/4.5.0/media/docs/cpp/gemm_api_3x.html)、[内核 API 与兼容性](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html)。

## 1. 先看完整调用链

![完整流程](01-end-to-end.png)

[PNG](01-end-to-end.png) · [可编辑 Draw.io](01-end-to-end.drawio)

```text
编译期
  CollectiveBuilder(epilogue) → CollectiveEpilogue
  CollectiveBuilder(mainloop, epilogue SMEM carveout) → CollectiveMainloop
  GemmUniversal<ProblemShape, Mainloop, Epilogue> → GemmKernel 类型
  GemmUniversalAdapter<GemmKernel> → Gemm 类型

主机运行期，每个候选问题
  TEST → TestAll<Gemm> → Testbed3x::run → TestbedImpl::run
    → sufficient / initialize inputs
    → Gemm::Arguments{mode, problem_shape, mainloop, epilogue, hw_info, scheduler}
    → get_workspace_size → allocate → can_implement
    → adapter.initialize
         → kernel.initialize_workspace
         → kernel.to_underlying_arguments
             → mainloop.to_underlying_arguments：A/B tensor + TMA descriptor
             → epilogue.to_underlying_arguments：fusion + C/D TMA descriptor
             → scheduler.to_underlying_arguments
         → 保存 Params，必要时设置 dynamic SMEM 属性
    → adapter.run → grid/block/cluster/SMEM
         → ClusterLauncher::launch_with_fallback_cluster（本例非 1×1×1）
         → cudaLaunchKernelExC

设备运行期，一次 kernel launch 内
  device_kernel<GemmKernel>(Params) → GemmKernel::operator()(Params, smem)
    → 建立共享存储、Collective 对象、pipeline、scheduler、TMEM
    → scheduler 持续给出 tile
    → load warp：GMEM A/B → SMEM 多级缓冲
    → MMA warp：SMEM descriptor → UMMA → FP32 TMEM
    → accumulator pipeline：通知完成、等待消费、允许复用
    → epilogue warp：TMEM → registers → fusion → SMEM D → TMA store → GMEM D
    → tail/drain → 安全释放 TMEM → kernel 完成

主机验证
  synchronize → host::Gemm3x/Gett reference → compare_reference
  → bool pass → 内层 EXPECT_TRUE → 外层 EXPECT_TRUE(pass)
```

`run()` 的成功通常说明 launch 阶段没有报告错误，不意味着异步设备计算已完成。Testbed 后续同步并比较结果，才覆盖设备运行和数值检查。

## 2. 编译期：为什么只写三个模板实参就能组成 kernel

### 2.1 Builder 产出的是类型

```cpp
using Epilogue = typename EpilogueBuilder</*输出、融合、tile、架构*/>::CollectiveOp;
using Mainloop = typename MainloopBuilder<
    /*输入、MMA、tile、cluster、stage、schedule*/>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

这些 `using` 不申请 GPU 内存，也不执行 GEMM。Builder 根据元素类型、布局、对齐、tile、cluster 和 schedule 选择 `CollectiveMma`/`CollectiveEpilogue` 的模板特化。通常先构建 Epilogue，是因为 Mainloop 的 `StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>` 需要预留输出侧存储，随后还要扣除 kernel 同步、调度等空间。

本例 epilogue Auto 选 1SM TMA 路径；Mainloop Auto 选 TMA + UMMA warp-specialized 路径。Mainloop builder 将 `TiledMma`、A/B 全局拷贝、SMEM layout、stage 数和 `DispatchPolicy` 打包成 `CollectiveMma` 类型。

### 2.2 `GemmUniversal` 不是把任意两个类粘在一起

声明在 `include/cutlass/gemm/kernel/gemm_universal_decl.h`，聚合头 `gemm_universal.hpp` include 多个架构实现。本例的实现是 `sm100_gemm_tma_warpspecialized.hpp:61` 起的偏特化，其使能条件检查 `Mainloop::DispatchPolicy::Schedule` 是否属于 `KernelTmaWarpSpecializedSm100` 或其 block-scaled 类别。

它从 Mainloop 提取 `TileShape/TiledMma/ArchTag/StrideA/B/ElementAccumulator/ClusterShape/Arguments/Params`，从 Epilogue 提取 `EpilogueTile/ElementC/D/StrideC/D/Arguments/Params/ThreadCount`。随后建立 scheduler 和多个 pipeline。接口必须在这些类型和同步约定上匹配；仅仅同名、继承了某个类或输出类型相同都不够。

省略的第四个模板参数默认 `void`。`tile_scheduler.hpp:207` 的 SM100 selector 把它映射到 `PersistentTileSchedulerSm100<ClusterShape, SchedulerPipelineStageCount>`。该实现包含硬件 CLC 工作获取，不能把它当作完全与架构无关的整数循环。

### 2.3 形状与地址，必须分四层

| 层 | 本例 | 谁决定 | 用途 |
|---|---|---|---|
| 完整问题 | `(M,N,K,L)` | 运行时 `Arguments.problem_shape` | 整个 GEMM 和 batch 范围 |
| Mainloop tile | `64×128×64` | 编译期 Builder | 输出 tile 和 K 分块 |
| CTA cluster | `1×4×1` | 编译期配置，本例静态 | 4 个 CTA 沿 N 协同，完整覆盖 64×512 |
| MMA Atom | `64×128×16`，对应本例 FP16 | MMA Traits | 每个完整 K tile 有 4 个 K=16 子操作 |
| Epilogue subtile | Auto 推导 | epilogue Builder | 分片加载 TMEM、融合并存回 D |

CuTe 的 A 视图为 `(M,K,L)`，B 为 `(N,K,L)`；B 的坐标顺序不同于数学记法 `B[K,N]`，不代表物理执行转置。batch 是通过 stride 的第 3 分量访问同一问题的第 l 批。

## 3. Kernel 在结构上怎样容纳两个 Collective

### 3.1 两套参数嵌入一套 kernel 参数

本例实现 `sm100_gemm_tma_warpspecialized.hpp:215` 的 host 接口为：

```cpp
struct Arguments {
  GemmUniversalMode mode;
  ProblemShape problem_shape;
  MainloopArguments mainloop;
  EpilogueArguments epilogue;
  KernelHardwareInfo hw_info;
  TileSchedulerArguments scheduler;
};
struct Params {
  GemmUniversalMode mode;
  ProblemShape problem_shape;
  MainloopParams mainloop;
  EpilogueParams epilogue;
  TileSchedulerParams scheduler;
  KernelHardwareInfo hw_info;
};
```

`Arguments` 面向调用者：A/B/C/D 指针、stride、alpha/beta 和调度选项。`Params` 面向设备：包含下层已经转换的 TMA/融合/调度描述。两者不是简单的同结构拷贝，也不是把整块矩阵复制进参数。

Kernel 的 `to_underlying_arguments` 是总调度者，分别调用两个 Collective 和 scheduler 的转换函数。Adapter 调用这个总转换并保存 `params_`；描述符的实际构建知识属于各个 Collective/CuTe 后端。

### 3.2 参数转换的具体数据流

Mainloop 的 `Arguments` 包括 `ptr_A, dA, ptr_B, dB` 及运行时数据类型字段。转换函数读取 `(M,N,K,L)`，构造 `tensor_a(M,K,L,dA)` 与 `tensor_b(N,K,L,dB)`，结合 SMEM layout、TiledMma 与 cluster layout 调用 `make_tma_atom_A_sm100` 和 `make_tma_atom_B_sm100`。该实现还构造 fallback 描述符字段；静态 cluster 的构造函数选择常规路径。

Epilogue 的转换函数建立 C/D 视图及 TMA 参数，并调用 `FusionCallbacks::to_underlying_arguments`。C=void 时实际 C 数据路径禁用，D 描述符仍然必要。alpha/beta 的数值与是否读取 C，受 fusion 配置及其参数共同控制。

注意三种“描述符”不同：

| 描述符 | 构建/使用位置 | 描述内容 | 移植注意 |
|---|---|---|---|
| TMA TensorMap | host 参数转换 → device copy | 全局 tensor、尺寸、stride、tile、swizzle 等 | NVIDIA 路径可调用 `cuTensorMapEncodeTiled`；SDC 需确认等价 ABI |
| UMMA SMEM descriptor | device MMA fragment/Traits | 共享内存地址、步长和 swizzle 等 | 不是 A/B 的数据，也不是普通 64 位指针 |
| UMMA instruction descriptor | Traits → fma | 输入/累加类型、M/N、major、取负等指令模式 | 不能默认与 SDC 位编码一致 |

### 3.3 共享存储不是中间矩阵

`SharedStorage` 在 kernel 文件 `:182` 处组合：

- `pipelines.mainloop`：A/B 缓冲的空/满及异步事务同步。
- `pipelines.epi_load`：可选 C/辅助输入搬运同步。
- `load_order`、`clc`、`clc_throttle`：搬运顺序和工作分发。
- `pipelines.accumulator`：Mainloop 与 Epilogue 的 TMEM 所有权交接。
- `tmem_dealloc`：退出和释放 TMEM 的参与者同步。
- `clc_response[]`、`tmem_base_ptr`：调度响应和 TMEM 基地址。
- `tensors.epilogue`、`tensors.mainloop`：输出侧 SMEM 暂存和 A/B 多级 SMEM 缓冲。

该版本的 `TensorStorage` 包含并列成员，不能泛称为两个 Collective 的 SMEM 总是 `union` 复用。TMEM 累加器本体也不在上述 SMEM 数组中，SMEM 只保存其地址及同步元数据。

## 4. 设备端：两个 Collective 的详细协作

![Collective 分工与交接](02-collective-handoff.png)

[PNG](02-collective-handoff.png) · [可编辑 Draw.io](02-collective-handoff.drawio)

### 4.1 线程分工

本例 kernel 的 `WarpCategory` 为：warp 0=MMA，warp 1=Sched，warp 2=MainloopLoad，warp 3=EpilogueLoad，warp 4 及之后=Epilogue。线程数公式为 `4×NumThreadsPerWarp + Epilogue::ThreadCount`；图上角色表示源码分工，不把总线程数硬编码为所有配置都相同。

| 角色 | 主要工作 | 所属层 |
|---|---|---|
| Sched | 获取/广播下一工作 tile，维护 CLC pipeline | Kernel + TileScheduler |
| MainloopLoad | 取 A/B tile，经 TMA 放入 SMEM stage | CollectiveMainloop |
| MMA | 等待 A/B ready、发射 MMA、写 TMEM、发布 accumulator ready | CollectiveMainloop + kernel 的交接代码 |
| EpilogueLoad | 按需搬 C/辅助输入 | CollectiveEpilogue；本例默认 C=void，无需 C load |
| Epilogue | 等待累加完成、TMEM→寄存器、融合、存 D、释放 accumulator stage | CollectiveEpilogue |

角色划分不代表所有代码每个线程都执行。warp/lane predicate 控制具体发射者；一个 warp 的角色也不等于硬件操作必须由 32 个 lane 各自发射一遍。

### 4.2 初始化：先建立共同“坐标系”和同步对象

1. 读取 warp、lane、cluster rank，映射角色；将动态 SMEM 缓冲解释为 `SharedStorage`。
2. 以各自 `Params` 构造 `CollectiveMainloop` 和 `CollectiveEpilogue`，它们引用同一次 launch 的参数/存储。
3. 初始化 MainloopPipeline、EpiLoadPipeline、EpiStorePipeline、AccumulatorPipeline、CLC pipeline 和 barrier，并执行所需同步，使初始化结果对参与者可见。
4. `load_init` 根据 tensor 与 cluster 分区生成后续 TMA copy 使用的视图和 multicast mask。
5. `init_tmem_tensors` 构造累加器 tensor 的布局；MMA 角色申请 TMEM，写入 `tmem_base_ptr`，通过分配结果 barrier 通知输出角色；双方 `set_tmem_offsets` 令自己的 tensor 视图指向同一物理 TMEM。
6. 每个角色从 scheduler 对应的工作流得到相同逻辑输出 tile，并分别维护本角色的 pipeline state。不是所有 warp 去抢同一个任务后自行猜测归属。

### 4.3 A/B 搬运协议：防止提前读取和提前覆盖

`CollectiveMainloop::load`，`sm100_mma_warpspecialized.hpp:589`：

```text
producer_acquire(write_state)          等这个 stage 可写
producer_get_barrier(write_state)      获得异步事务完成对象
write_stage = write_state.index()
copy(TMA_A.with(barrier, maskA), A_tile, smemA[write_stage])
copy(TMA_B.with(barrier, maskB), B_tile, smemB[write_stage])
推进 K iterator / state
```

MMA 侧 `consumer_wait(read_state)` 等待 TMA 数据完成，再从相同 stage 的 SMEM descriptor 读取。完成对应异步 MMA 读取所要求的同步后，通过 `consumer_release` 路径允许生产者复用 SMEM。底层 SM100 pipeline 含 UMMA arrival/commit 语义，不能将 release 机械替换为一个普通计数器写。

本例 cluster 沿 N 有 4 个 CTA，它们需要同一 M/K 范围的 A、不同 N 范围的 B，Builder 因而可为 A 选择 multicast。multicast 是 cluster 搬运能力，不改变 1SM MMA 的协作粒度。

### 4.4 MMA 与 TMEM：两个 Collective 的真正交接点

`CollectiveMainloop::mma`，同文件 `:652` 起：

1. `accumulator_pipeline.producer_acquire(acc_state)`，等待有一个可写的 TMEM accumulator stage。
2. 对新输出 tile 设置 `tiled_mma.accumulate_ = ScaleOut::Zero`。
3. 对每个 K tile 等 `mainloop_pipeline.consumer_wait`，选择 `read_stage`。
4. 逐个 Atom K 子块执行 `cute::gemm(tiled_mma, descA_fragment, descB_fragment, accumulators)`；首次之后设置 `ScaleOut::One`，继续累计部分和。
5. 释放对应 A/B stage，推进到下一个 K tile。
6. 返回 kernel 后，kernel 在 `sm100_gemm_tma_warpspecialized.hpp:770` 调用 `accumulator_pipeline.producer_commit(acc_state)`，将异步 UMMA 完成纳入消费者可等待的协议，然后推进 accumulator state。

`slice_accumulator(tmem_storage, acc_stage)` 传递的是指定 stage 的 tensor 视图，并没有 host 拷贝或先写一份中间 GMEM 矩阵。Epilogue 取得同一地址/布局的 accumulator 视图。

实际 MMA 调用链：

```text
CollectiveMainloop::mma
  → cute::gemm
  → MMA_Atom::call
  → MMA_Traits<SM100_MMA_F16BF16_SS>::mma_unpack（ADL hidden friend）
      desc_a = A[0]; desc_b = B[0]
      tmem_c = raw_pointer_cast(D.data())
      idesc = make_runtime_instr_desc(traits.idesc_)
  → SM100_MMA_F16BF16_SS::fma
  → elect_one_sync 的发射 lane
  → tcgen05.mma.cta_group::1.kind::f16
```

`ElementC=void` 是禁用 epilogue 的全局源矩阵 C；`scaleC`/`accumulate_` 则控制 MMA 是否保留旧 TMEM 部分和。它们含义不同，因此 C=void 仍有跨 K 的累加。

### 4.5 Epilogue 消费协议与输出

Kernel 在 `:897` 取得 accumulator 视图，调用 `scheduler.fixup`；本例默认 scheduler 不切分 K，不需要 split-K 归并。随后把 accumulator、对应 pipeline/state、完整 problem shape、tile 坐标和 epilogue SMEM 一并交给 `CollectiveEpilogue::store`。

Epilogue 在相关 subtile 上执行：

```text
acc_pipeline.consumer_wait             等 UMMA 的结果可读
copy(tiled_t2r, tmem_accumulator, rAcc)  TMEM → 寄存器
fence_view_async_tmem_load              需要释放时确保相关 TMEM 读取满足完成约束
acc_pipeline.consumer_release          该 stage 不再需要时，交还给 MMA
cst_callbacks.visit(...)               alpha/beta/融合计算
NumericArrayConverter                  转成输出 SMEM 元素类型
copy(tiled_r2s, rD, sD[stage])          寄存器 → SMEM
fence_view_async_shared + synchronize  让 TMA 看见所有输出写入
copy(tma_store_d, sD, gD)               SMEM → GMEM D
store pipeline commit/acquire/tail     等待、复用与排空输出缓冲
```

TMEM stage 的 release 可以发生在相关累加器数据已读取到寄存器之后，不必等待整个 D 都写完；具体时机由 subtile、`do_acc_release` 和 overlapping 配置决定。不能把 epilogue 的每个 subtile 都画成“释放整个 TMEM”。

默认计算为 `D = alpha*Acc + beta*C`；本例 C=void，等效 `D = alpha*Acc`。输出 FP32，默认 alpha=1，因而验证的是每批 FP16 A/B 形成的 FP32 GEMM 结果。

### 4.6 持续调度与退出

load、MMA、epilogue 可以处理不同流水阶段，持续推进多个输出 tile。Mainloop 的 A/B stage 数、TMEM accumulator stage 数、scheduler pipeline stage 数是不同的容量，不应统称一个 `Stages`。

退出时需要 load_tail、accumulator 的完成/消费同步、epilogue load/store tail、TMEM 分配许可释放和最终 free。TMEM 的释放必须满足全部参与者不再访问；单看 `mma()` 返回不能证明可立即 free。

## 5. Adapter 怎样把这个 C++ 类型启动到设备

`GemmUniversalAdapter` 是有状态的 host handle，3.x 特化在 `gemm_universal_adapter.h:123`。它保存 `Params params_`，提供稳定接口，但内部 launch 仍有 NVIDIA 架构与运行时依赖。

| 调用 | 实际负责的事 | 不应混淆 |
|---|---|---|
| `get_workspace_size(args)` | 查询 kernel/epilogue/scheduler 的额外全局工作区 | 不是 `SharedStorageSize`，也不含调用者的全部 A/B/D 内存 |
| `can_implement(args)` | 聚合 mode、Collective、scheduler、cluster 等约束 | 不执行 kernel，不保证后续申请/launch 成功 |
| `initialize(args, workspace, stream)` | 初始化 workspace，调用 kernel 参数转换，保存 params_，必要时设置动态 SMEM 属性 | 3.x 这个实现不会自动替调用者执行一遍 `can_implement` |
| `update(args, workspace)` | 重新转换/更新参数，检查所需 workspace | 不能概括为已重新完成所有 workspace 初始化/函数属性配置 |
| `run(stream)` | 使用保存的 params_ 计算 launch 配置并提交 kernel | 是异步提交，尚未 host reference 比较 |
| `run(args, workspace, stream)` / 对应 `operator()` | 先 initialize 再 run | 仍需要调用者提供正确生命周期的内存和参数 |

本例 `ArchTag=Sm100`、静态 cluster 大小为 4，所以普通 host launch 路径进入 `ClusterLauncher::launch_with_fallback_cluster`，内部用 `cudaLaunchKernelExC`。静态 `1×1×1`、动态 cluster、其他架构、启用 `CudaHostAdapter` 等配置会走不同分支，不能把这一条路径推广为所有 Gemm 的唯一 launch。

`device_kernel.h:118` 中的入口收到 `Params`，声明动态共享内存 `smem[]`，构造 `Operator op`，调用 `op(params, smem)`。所以严格来说 `GemmUniversal` 提供 kernel 的实现类型/编排，而可被 runtime launch 的函数实例是这个入口包装。

## 6. TestAll、Testbed、EXPECT_TRUE 的完整语义

### 6.1 一次 TestAll 的覆盖范围

布局来自已经固定的 `Gemm`；函数体读取其 alignment、TileK 和 Stages 来生成尺寸。本例：

| 测试轴 | 候选/行为 |
|---|---|
| M | `{8, 488}` |
| N | `{8, 496}` |
| K | `{8, 64*(S+1)-8}`，S=`GemmKernel::DispatchPolicy::Stages` |
| L | 常规循环为 1；最后另测 `(264,264,168,3)` |
| raster order | AlongM、AlongN，属于 tile 遍历次序，不是矩阵存储布局 |
| max swizzle size | 1、4，属于 scheduler tile 重排参数，不是 SMEM swizzle 位编码 |
| decomposition | 默认 heuristic；显式 StreamK scheduler 才增加其他模式 |
| splits | 内层依据 K tile 数生成多个候选；默认 persistent scheduler 的 args 只接收 raster/swizzle，不实际启用 split-K |
| alpha / beta / equality | 默认 1 / 0（C=void）/ RELATIVE |

举例：**若编译结果 S=8**，较大 K=568，K tile 数为 9。小 K 的 splits 列表为 `{1,1,2}`，保留重复项；大 K 为 `{1,2,3,9,10}`。候选 Testbed 调用数是 `2*2*2*2*(3+5)+1 = 129`。这是条件化的静态枚举推导，不能据此声称本轮真的启动了 129 次 kernel。因为 `can_implement` 跳过、异常或提前失败都会改变实际执行数，且 S 在本轮未重新编译求值。

### 6.2 每次 Testbed 做什么

1. `sufficient()` 查询设备和每 block 可用动态 SMEM，本版本不满足会返回 false。
2. HostCollectiveMainloop 初始化 A/B，HostCollectiveEpilogue 初始化 C/D 与标量/参考数据，并完成所需 host/device 数据传输。
3. 构建 kernel `Arguments`：`mode=kGemm`、problem size、mainloop args、epilogue args、hw_info、scheduler args。注意本测试 rank=4 的 batch 也使用此实现接受的 `kGemm` 路径，不能把所有 L>1 都机械解释成测试设置了 `kBatched`。
4. 查询并申请 workspace；检查 `can_implement`；初始化 adapter；提交 run。
5. `cudaDeviceSynchronize()` 捕捉异步错误。
6. `verify()` 调用 `cutlass::reference::host::Gemm3x`，底层 `Gett` 做主机参考计算，再比较 D。
7. 内部断言给出具体问题信息；失败返回 false，`TestAll` 退出；最外层 `EXPECT_TRUE(pass)` 将返回结果登记到 GoogleTest。

### 6.3 通过标志的实际限制，移植时尤其重要

当前 `gemm_testbed_3x.hpp:2991–2996` 在 `can_implement` 不成功时打印“不支持”并返回 **true**。因此 TestAll 的 true 可能包含未执行候选。`EXPECT_TRUE(pass)` 断言的是函数返回值，不是“所有理论配置都完成了计算”。GoogleTest 还会保留其他内层断言失败，不能只看最终 bool。

当前 Testbed 在 initialize 错误时仅打印，随后仍调用 run，并覆盖 `status`。移植验证版建议明确区分初始化失败、launch 失败、设备失败、数值失败、能力跳过，而不是依赖这个宽松控制流。

SDC200 测试报告应至少记录：候选数、明确不支持数、initialize 成功数、launch 成功数、同步成功数、数值比较成功数，以及每个 `(M,N,K,L,layout,scheduler)`。强制验收的目标配置如果跳过，应判为未完成，不能算数值通过。本次只给出建议，没有修改测试框架。

## 7. 卡片：每层该记住什么

![九张核心卡片](03-concept-cards.png)

[PNG](03-concept-cards.png) · [可编辑 Draw.io](03-concept-cards.drawio)

| 卡片 | 一句话 | 最容易混淆的边界 |
|---|---|---|
| ProblemShape | int×4 表示运行期 M/N/K/L | 不决定 tile、warp 或 cluster |
| Builders | 选择并生成两个 Collective 类型 | `CollectiveOp` 是类型，不是一次运算结果 |
| Mainloop | 搬 A/B 并跨 K 累加到 TMEM | 一份 Mainloop 内部已经有 load/MMA 两个角色 |
| Kernel | 组织角色、存储、同步和任务坐标 | 不只是两个函数调用的顺序 |
| Handoff | accumulator view + pipeline 交接 | ready 信号必须对应异步完成与可见性 |
| Epilogue | 读累加器、融合、转换并存 D | C=void 不影响 MMA 内部累加 |
| Adapter | host 参数转换入口与 launch handle | launch 成功不等于数值通过 |
| TestAll | 对固定 Gemm 枚举尺寸/调度 | 多布局需要多个类型/测试实例 |
| SDC200 | 保留抽象契约，重建硬件语义 | 替换 MMA builtin 只覆盖链条中的一环 |

## 8. SDC200：适配边界与现有代码的真实状态

![SDC200 适配分层](04-sdc200-porting.png)

[PNG](04-sdc200-porting.png) · [可编辑 Draw.io](04-sdc200-porting.drawio)

“对标 SM100”是能力目标，不自动说明 SDC200 与 NVIDIA 的 ABI、warp、cluster、TMEM、descriptor、CLC、异步内存模型和 CUDA runtime 相同。建议保留 `GemmUniversal/CollectiveBuilder/GemmUniversalAdapter` 的用户接口形状，新增 SDC 架构策略和后端，按已验证能力启用配置。

### 8.1 当前能确认的 SDC 参考实现

现有 `D:/lvz/sdc/cutlass/include/cute/arch/mma_sdc200_umma.hpp:72` 定义 `SDC200_MMA_F16BF16_SS`，`fma` 在 `:108` 接受两个 64 位 descriptor、32 位 TMEM 地址、scaleC 和组合 instruction descriptor；在 `:117` 调用：

```cpp
__sdc_uten_mma_bf16(tmem_c, desc_a, desc_b,
                    uint32_t(idescE >> 32), scaleC);
```

实际 compiler header `sdc_builtins.h:897` 声明：

```cpp
void __sdc_uten_mma_bf16(uint32_t dTmem, uint64_t aDesc,
                        uint64_t bDesc, uint32_t iDesc,
                        uint32_t inputD);
```

这证明包装层和 builtin 形参形式已有对应；**不证明 FP16 模式、descriptor 字段、K 粒度、完成事件或数值已正确**。builtin 名含 bf16，包装类型支持 F16/BF16，实际类型选择是否由 iDesc 控制，需要以 SDC 编译器/ISA 契约和 FP16/BF16 两套运行测试确认。

`mma_sdc200_desc.hpp:109` 具有 `SmemDescriptor` union，属于 ABI 敏感数据结构。不能因为字段名/位宽和 SM100 相似就将它作为兼容结论；需要 golden bit-pattern、合法/非法布局与硬件读取验证。

`cluster_sdc200.hpp:180` 的 `elect_one_sync` 仍含 `elect.sync`；该文件还有 `barrier.cluster`、`mapa.shared::cluster` 和 NVIDIA 宏条件。可能依赖编译器兼容支持，也可能尚未替换，当前不能认定已原生适配。include 应放在全局命名空间，并通过后端选择避免同一翻译单元重复定义 `cute::elect_one_sync` 或出现 `cute::cute`。

### 8.2 两条适配路径与边界

| 能力条件 | 可以保留什么 | 必须改变什么 |
|---|---|---|
| SDC 有等价异步 TMA、可寻址 TMEM、完成 barrier、cluster 语义 | 可考虑保留类似 Mainloop/Epilogue 的协议和 kernel 分工 | 每个 descriptor、builtin、completion、runtime launch 和资源常量仍需验证/替换 |
| 有 MMA/TMEM，但无 CLC | 可保留 GEMM 数据链和部分流水线 | 使用静态 tile 调度或软件队列；scheduler 实现、selector 与 kernel 调度角色需适配 |
| 无 cluster 或 multicast | 单 CTA GEMM 抽象仍可保留 | 先用 cluster=1；A/B 独立搬运；删除/重做 cluster 同步与发射约束 |
| 无等价 TMA | Tensor/布局与分块思想可保留 | 用协作 load 或其他 DMA，重做 Mainloop load、同步和输出 store 路径 |
| MMA 结果在寄存器、无等价 TMEM | GEMM 上层类型接口与计算语义可保留 | 重建 accumulator fragment/Traits、Kernel handoff、Epilogue T2R，不能照搬 TMEM kernel |
| CUDA runtime/launch ABI 不兼容 | Adapter 用户方法命名可保留 | runtime shim 或新 launcher、设备属性/分配/拷贝/同步/错误、host descriptor encoder |

`CudaHostAdapter` 可作为 host launch 扩展点之一，但不能修复 device 内联 PTX、descriptor ABI 或同步语义。现 Adapter 使用 `kMinComputeCapability >= 90`、`==100/101/103/120` 等分支；即便给新架构设置“200”也不等于自动走正确 launch。应该用明确后端/能力选择替代借用 NVIDIA 编号来蒙混分发。

### 8.3 完整参考表与阅读顺序

[打开完整适配表](sdc200-adaptation.md)。表格覆盖：构建与宏、架构标签、类型/布局、Builder、dispatch、两个 Collective、kernel、scheduler、pipeline、barrier、TMA host/device、UMMA Traits/descriptor/Operation、TMEM、epilogue fusion、adapter/runtime、测试/reference、诊断和扩展范围。每行列出实际文件/API、可复用部分、需要适配部分、候选 SDC 落点及验收标准。

表中“建议新增”是设计建议，不是声称文件已经存在；“条件复用”是接口/算法在满足契约后可复用，不表示整个文件可以原封不动照搬。完整性针对本例 dense FP16/FP32 1SM 链以及扩展接入边界，不代表未经硬件规格就能列尽整个 CUTLASS 仓库的所有架构功能。

### 8.4 建议的落地顺序与验收门槛

1. **P0：固化契约。** 确认 lane/warp、MMA 形状、TMEM 地址/容量、三类 descriptor、异步完成、barrier scope、cluster 和 runtime；为每项标记已支持/可兼容/需模拟/不支持。
2. **P1：底层闭环。** descriptor 编码 → 单次 MMA → TMEM 读回 → 与主机参考比较；覆盖 overwrite/accumulate、FP16/BF16、合法 tile 与拒绝条件。先证明一个 Atom，不直接跳入复杂 persistent kernel。
3. **P2：单 CTA。** 固定支持的 tile，cluster=1，静态调度，最简单搬运和无融合 epilogue，跑多 K tile、尾块和 batch。该阶段改变了调度/搬运策略，不能将通过结果视为原 SM100 路径全兼容。
4. **P3：异步流水线。** 增加多级 load/MMA/accumulator/store；用大 K、多 tile、重复运行压力验证 stage 所有权、phase wrap、尾部 drain 与可见性。
5. **P4：高级功能。** 依次启用 TMA、multicast/cluster、CLC 或软件替代、融合、Stream-K/2SM 等。每项独立列覆盖矩阵，能力不支持要明确拒绝。
6. **P5：集成验收。** 修复测试入口问题后接入 SDC unit target，强制记录实际执行与 skip；数值通过后再查机器码和性能。优化前记录 baseline，不能用编译成功替代正确性或吞吐达标。

## 9. 关键源码导航

下列行号来自本次工作树，后续修改会漂移；完整快照与 SHA-256 见 `source-evidence.json`，关键片段见 `source-excerpts.txt`。

| 目的 | 文件 | 位置 |
|---|---|---|
| 本例 Builder/Kernel/Adapter/TestAll | `test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu` | 197、215、234、244、253 |
| GemmUniversal 声明/聚合入口 | `include/cutlass/gemm/kernel/gemm_universal_decl.h`；`gemm_universal.hpp` | 模板声明；SM100 include |
| 偏特化与类型拼装 | `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp` | 61–179 |
| shared storage / args / params | 同上 | 182、215、225 |
| 参数转换/检查/workspace/grid | 同上 | 257、301、339、384 |
| warp 角色与对象建立 | 同上 | 234、405、435 |
| TMEM、MMA、发布完成 | 同上 | 727、762、770 |
| Epilogue 调用 | 同上 | 897、911 |
| Adapter initialize/run | `include/cutlass/gemm/device/gemm_universal_adapter.h` | 313、375、509 |
| 真正 global 入口 | `include/cutlass/device_kernel.h` | 118 |
| 扩展 launch | `include/cutlass/cluster_launch.hpp` | 294 |
| Mainloop host 转换/load/mma | `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp` | 355、589、652 |
| MMA Traits→fma | `include/cute/atom/mma_traits_sm100.hpp` | 284、323 |
| MMA 指令落点 | `include/cute/arch/mma_sm100_umma.hpp` | 86、111 |
| Epilogue T2R/fusion/store | `include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp` | 574、866、882、895、922、771 |
| Testbed 执行与 skip 语义 | `test/unit/gemm/device/gemm_testbed_3x.hpp` | 2877、2991、3014、3023、3032、3047 |
| TestAll 当前损坏与枚举 | 同上 | 3828、3840、3853、3878、3966 |
| CPU 参考 | `tools/util/include/cutlass/util/reference/host/gett.hpp` | 491、837 |

## 10. 交付与验证

- 本文、独立适配表、UTF-8 CSV 表、4 组 PNG/Draw.io、离线阅读页均保存在当前项目 `deliverables/gemm-universal-sdc200/`。
- 所有 PNG 和 `.drawio` 来自同一节点/连线/文本/坐标数据；Draw.io 中节点、箭头和文字可单独编辑，没有用整张位图替代结构。字体字形、抗锯齿和箭头尖端在不同渲染器中可能略有区别。
- 检查范围：文件存在、源链接与锚点、UTF-8、Draw.io XML、节点/连线、PNG 尺寸、主要布局和文字视觉检查；详细结果见 `validation.json`。
- 本轮没有改动 CUTLASS/SDC 源文件，也没有进行编译或设备数值测试。
- 本轮问答完整记录保存在 `qa-history/2026-10-20/`。本地持久保存已完成；未配置并验证跨电脑同步渠道，本轮跨电脑同步未完成。


## 完整适配表

# SDC200 完整流程适配参考表

基线：dense FP16 A/B → FP32 accumulator/D，1SM，含 Mainloop、Epilogue、Kernel、Adapter 与测试链。以下 68 项为源码入口级清单；SDC 候选文件标为“建议”的均是设计建议。分类含“复用”不意味着整个文件无硬件依赖。

相对路径的根目录为 `D:/000_SDC/001TASK/02_Research/cutlass`；第 68 项显式列出外部 SDC compiler header 的绝对路径。链接与行号来自本轮工作树。实际 SDC 包装位于另一棵 `D:/lvz/sdc/cutlass`，两者版本不同，不能混作同一源码快照。

优先读 11–17（Mainloop）、41–50（Epilogue/Kernel）、51–57（调度与启动）、60–64（验收）；底层必须补齐 19–40。本文给出架构条件下的适配建议，没有声称 SDC 硬件已支持这些能力。

整条链至少有四种独立 ABI：kernel 参数、TMA TensorMap、UMMA SMEM/指令 descriptor、TMEM 地址/fragment 布局。MMA builtin 可编译不能证明其余 ABI 兼容。

| 阶段 / 分类 | 实际文件与源码入口 | API | 可复用部分 | 需要适配或核验 | SDC 落点 | 验收条件 |
|---|---|---|---|---|---|---|
| 01 构建；条件复用 / 必须适配后端 | [CMakeLists.txt:58](D:/000_SDC/001TASK/02_Research/cutlass/CMakeLists.txt:58) | CMake project / CUDA language | 目标分层、源码组织、CTest 思路 | SDC 编译器识别、device 编译/链接、目标架构参数；不能沿用 sm_100a 标志 | 新增 SDC toolchain / 独立构建选项 | host/device 最小程序编译链接运行；确认实际后端 |
| 02 能力宏；条件复用 / 必须适配后端 | [include/cute/config.hpp:42](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/config.hpp:42) | CUTE_HOST_DEVICE / CUTE_DEVICE / invalid-control-path | 接口注解宏的调用位置 | SDC host/device 标注、forceinline、RTC、诊断；未实现设备操作必须显式拒绝 | 现配置增加明确 SDC 分支 | host/device 双模式编译；禁用能力触发预期错误 |
| 03 指令能力；条件复用 / 必须适配后端 | [include/cute/arch/config.hpp:56](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/config.hpp:56) | CUTE_ARCH_* feature macros | 按能力开关分发的模式 | 将 SDC 能力和 NVIDIA __CUDA_ARCH__ 分离；核对 fallback 是否会静默进入空路径 | 新增 SDC feature 宏 | 支持/不支持矩阵对应正确特化及诊断 |
| 04 架构标签；条件复用 / 必须适配后端 | [include/cutlass/arch/arch.h:100](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/arch/arch.h:100) | arch::Sm100 / sm100_smem_capacity_bytes | arch tag 抽象和资源查询接口 | 新增 SDC tag 与真实 SMEM/TMEM/warp 资源；禁止用假 compute capability 驱动 CUDA 分支 | 建议新增 arch::Sdc200 | 静态资源计算符合规格，超限配置拒绝 |
| 05 架构 SIMD 配套（按需）；条件复用 / 必须适配后端 | [include/cutlass/arch/mma_sm100.h:68](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/arch/mma_sm100.h:68) | arch::Mma SIMD specialization / architecture includes | 聚合头和能力暴露方式 | 清理 NVIDIA 编译器保护与目标指令依赖，接入 SDC 后端 | 建议 mma_sdc200.h 或统一后端入口 | include 自洽；不意外依赖 NVIDIA PTX |
| 06 问题与模式；优先复用语义 / 审核边界 | [include/cutlass/gemm/gemm_enumerated_types.h:57](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/gemm_enumerated_types.h:57) | GemmUniversalMode / problem semantics | GEMM/strided batch 数学语义 | mode 支持范围明确，指针数组模式不能当 strided batch 处理 | 优先复用；能力检查限制模式 | M/N/K/L、batch stride、mode 对应正确 |
| 07 类型元编程；优先复用 | [include/cute/int_tuple.hpp:371](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/int_tuple.hpp:371) | Shape / Stride / tuple utilities | 编译期整数与运行期 tuple 算法 | 编译器模板兼容、整数宽度和溢出边界 | 优先原接口复用 | 静态 shape 断言与运行时坐标单测 |
| 08 Layout；优先复用算法 / 核验编译器 | [include/cute/layout.hpp:231](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/layout.hpp:231) | Layout / make_layout / compose / partition | 坐标映射和分块代数 | 设备注解/编译器兼容；硬件依赖布局仍由 Traits/Builder 重选 | 复用通用层；隔离硬件 layout | 索引到地址映射与边界参考一致 |
| 09 Tensor；条件复用 / 必须适配后端 | [include/cute/tensor_impl.hpp:263](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/tensor_impl.hpp:263) | Tensor / make_tensor / local_tile | tensor 视图与坐标组合 | 地址空间、指针标记、TMEM engine 和设备注解 | 通用视图复用；新增 SDC memory engine | 同一坐标映射正确地址空间，无错误 cast |
| 10 Swizzle；条件复用 / 必须适配后端 | [include/cute/swizzle.hpp:64](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/swizzle.hpp:64) | Swizzle / composed layout | 通用位变换算法 | 合法 bank/swizzle 与 descriptor 编码由 SDC 重新定义 | 算法复用 + SDC layout policy | golden 地址序列、bank 分布、编码符合规格 |
| 11 主循环入口；条件复用 / 必须适配后端 | [include/cutlass/gemm/collective/collective_builder.hpp:39](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/collective/collective_builder.hpp:39) | gemm::collective::CollectiveBuilder | 调用者模板接口与 CollectiveOp 契约 | include/register SDC 特化；不支持组合明确报错 | 建议 builders/sdc200_umma_builder.inl | 目标元素/布局/alignment/shape 选中正确 Collective |
| 12 Mainloop builder；条件复用 / 必须适配后端 | [include/cutlass/gemm/collective/builders/sm100_umma_builder.inl:329](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl:329) | TiledMma / GmemTiledCopyA/B / PipelineStages / DispatchPolicy | 根据配置组装 Collective 的流程 | MMA 形状、cluster copy、SMEM layout、资源 carveout、stage、overlap 策略全部核验 | 建议 sdc200_umma_builder.inl | 资源不超限；自动策略符合能力而非名称相似 |
| 13 主循环 helper；条件复用 / 必须适配后端 | [include/cutlass/gemm/collective/builders/sm100_common.inl:444](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:444) | sm100_make_trivial_tiled_mma / cluster-to-TMA / layout selectors | 类型选择的组织方式 | 选择 SDC Operation/Traits 和 copy atom；改变 tile/major 限制时重选布局 | 建议 sdc200_common.inl | 类型静态断言 + 代表布局实际数据验证 |
| 14 Kernel carveout；条件复用 / 必须适配后端 | [include/cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl:45](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl:45) | SMEM carveout helpers | 预留同步元数据后计算可用缓冲的策略 | 实际 barrier 大小、CLC/软件队列、对齐、SMEM 容量重新计算 | SDC carveout helper | sizeof 实例总量与 launch SMEM 一致 |
| 15 Mainloop policy；条件复用 / 必须适配后端 | [include/cutlass/gemm/dispatch_policy.hpp:1029](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/dispatch_policy.hpp:1029) | MainloopSm100TmaUmmaWarpSpecialized / KernelTmaWarpSpecializedSm100 | Stages/ClusterShape/Schedule 的类型契约 | 新增 SDC policy 与 kernel 匹配条件，分离 load/accumulator/scheduler stage | 建议 MainloopSdc200* / Kernel*Sdc200 | 默认与显式 schedule 均分发到预期实现 |
| 16 主循环主体；条件复用 / 必须适配后端 | [include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:81](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:81) | Arguments / Params / load_init / load / mma / load_tail | 分区、K 循环、producer/consumer 接口 | 替换 TMA、TMEM 视图、pipeline、cluster mask 与异步完成；必要时重构分工 | 建议 collective/sdc200_mma_warpspecialized.hpp | 多 K tile、phase wrap、尾块、持续多 tile 无误 |
| 17 主循环 host 转换；条件复用 / 必须适配后端 | [include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:355](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:355) | to_underlying_arguments / make_tma_atom_A/B_sm100 | 从 ptr/stride/problem shape 构造 tensor 的流程 | SDC host tensor descriptor encoder；fallback/cluster 描述符的支持范围 | SDC Collective Params + descriptor factory | 描述符字节与硬件寻址对照；拒绝不合法 stride |
| 18 MMA 通用算法；优先复用 / 核验实例化 | [include/cute/algorithm/gemm.hpp:77](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/algorithm/gemm.hpp:77) | cute::gemm | tile/Atom 调用展开和接口 | 模板限制与 fragment/布局一致性；确认没有进入错误默认路径 | 优先复用 + SDC Traits | 编译期形状检查，Atom 次序/累加与参考一致 |
| 19 MMA Atom；条件复用 / 必须适配后端 | [include/cute/atom/mma_atom.hpp:117](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/mma_atom.hpp:117) | MMA_Atom::call / TiledMMA / ADL mma_unpack | Operation→Traits→Atom→TiledMMA 框架 | SDC fragment engine、hidden friend 解包与返回约定 | 通用 Atom 复用；新增架构 Traits | 能解析到 SDC fma，且不存在错误 namespace/ADL |
| 20 MMA Traits；条件复用 / 必须适配后端 | [include/cute/atom/mma_traits_sm100.hpp:284](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/mma_traits_sm100.hpp:284) | MMA_Traits / Shape_MNK / FrgType / mma_unpack | Traits 描述形式和解包流程 | SDC Atom K/M/N、lane/value layout、TMEM fragment、scale 与 descriptor 编码 | 建议 atom/mma_traits_sdc200.hpp | 静态 fragment 映射 + 每元素数值对照 |
| 21 TMEM fragment；条件复用 / 必须适配后端 | [include/cute/atom/mma_traits_sm100_frag.hpp:78](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/mma_traits_sm100_frag.hpp:78) | UMMA::tmem_frg_* / TMEM fragment layouts | 表达 accumulator tensor 的抽象 | TMEM 行列、lane/data-path、地址加法、overlap 范围 | SDC fragment traits/engine | T2R 每 lane 映射和多 stage 互不覆盖 |
| 22 MMA 指令；必须适配 / 已有包装未验证 | [include/cute/arch/mma_sm100_umma.hpp:86](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/mma_sm100_umma.hpp:86) | SM100_MMA_F16BF16_SS::fma / tcgen05.mma | fma 接口形状，受控覆盖/累加语义 | SDC builtin lowering、FP16/BF16 模式、发射粒度、完成事件、2SM 限制 | 已见 mma_sdc200_umma.hpp；需验收 | 单 Atom FP16/BF16 + overwrite/accumulate + 真正生成机器指令 |
| 23 MMA descriptor；必须确认 ABI / 不可按名称复用 | [include/cute/arch/mma_sm100_desc.hpp:131](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/mma_sm100_desc.hpp:131) | SmemDescriptor / InstrDescriptor / smem_desc / make_instr_desc | 有类型的编码器接口 | 地址单位、bitfield ABI、stride/swizzle、idesc 高低位、合法值 | 已见 mma_sdc200_desc.hpp；逐字段核验 | golden bit-pattern、反解与硬件读数一致 |
| 24 TMEM 分配；条件复用 / 必须适配后端 | [include/cute/arch/tmem_allocator_sm100.hpp:84](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/tmem_allocator_sm100.hpp:84) | TMEM::Allocator1Sm / allocate / free / release_allocation_lock | allocator 接口和生命周期管理方式 | SDC 分配/释放 builtin、粒度/容量、参与 lane、返回位置与许可语义 | 建议 tmem_allocator_sdc200.hpp | 分配→写→读→drain→释放；重复运行无泄漏/死锁 |
| 25 TMEM helper；条件复用 / 必须适配后端 | [include/cutlass/detail/sm100_tmem_helper.hpp:49](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/detail/sm100_tmem_helper.hpp:49) | TMEM partition/accumulator helpers | 将 tile 切成 accumulator stage 的思路 | 硬件 TMEM 行列容量与别名、overlap、寻址布局 | SDC TMEM helper | 所有 stage/subtile 映射合法且不发生意外别名 |
| 26 Copy Atom；优先复用框架 / 新增后端 | [include/cute/atom/copy_atom.hpp:45](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/copy_atom.hpp:45) | Copy_Atom / TiledCopy / copy_unpack | copy 分区和调用框架 | 新增 SDC copy Traits，核验线程参与/向量宽度约定 | 通用 Atom + SDC traits | 每 lane 源/目的覆盖正确，无缺失或重复 |
| 27 TMEM copy Traits；条件复用 / 必须适配后端 | [include/cute/atom/copy_traits_sm100.hpp:274](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/copy_traits_sm100.hpp:274) | Copy_Traits / make_tmem_copy / T2R | 表达线程到 TMEM 的 copy 接口 | lane/register layout、load 粒度、合法 shape、完成语义 | 建议 copy_traits_sdc200.hpp | 非平凡 TMEM pattern 完整读回 |
| 28 TMEM copy ISA；必须适配 | [include/cute/arch/copy_sm100.hpp:628](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/copy_sm100.hpp:628) | SM100_TMEM_LOAD* / tcgen05.ld / copy operations | copy Operation 的参数边界 | SDC TMEM load/store builtin、wait、寄存器布局 | 建议 copy_sdc200.hpp | 不同 lane/subpartition 读回及 fence 压测 |
| 29 TMA Traits SM100；条件复用 / 必须适配后端 | [include/cute/atom/copy_traits_sm100_tma.hpp:729](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/copy_traits_sm100_tma.hpp:729) | make_tma_atom_A/B_sm100 / multicast partition | 按 A/B tensor 和 cluster 分区的算法意图 | SDC descriptor、multicast mask、CTA group、tensor shape 限制 | 建议 copy_traits_sdc200_tma.hpp | A 广播/B 分片/边界 tensor 的数据一致 |
| 30 TMA host 编码；必须适配或证明兼容 | [include/cute/atom/copy_traits_sm90_tma.hpp:1047](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1047) | make_tma_copy / cuTensorMapEncodeTiled | tensor→copy 参数的接口意图 | NVIDIA Driver TensorMap ABI 和 driver wrapper 需替换或有证据兼容 | SDC tensor-map encoder / runtime shim | 跨 shape/stride/swizzle 编码和实际传输正确 |
| 31 TMA descriptor 类型；条件复用 / 必须适配后端 | [include/cute/arch/copy_sm90_desc.hpp:292](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/copy_sm90_desc.hpp:292) | TmaDescriptor / swizzle enums / descriptor helpers | 描述符抽象与工厂接口 | 结构大小、对齐、enum 值、host/device 可见性 | 建议 SDC TMA descriptor | sizeof/alignof + 编码 golden + kernel 参数 ABI |
| 32 TMA device load/store；必须适配 | [include/cute/arch/copy_sm90_tma.hpp:61](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/copy_sm90_tma.hpp:61) | SM90_TMA_LOAD/STORE/MULTICAST / cp.async.bulk | 全局↔共享 tensor copy 意图 | SDC DMA/bulk copy、OOB 行为、barrier transaction bytes、commit/wait | SDC TMA Operations；无 TMA 时换协作 copy | 异步完成、越界 fill/store 边界与事务计数 |
| 33 TMA SM100 扩展；条件复用 / 必须适配后端 | [include/cute/arch/copy_sm100_tma.hpp:52](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/copy_sm100_tma.hpp:52) | SM100 TMA 1SM/2SM Operations | 按能力选择 Operation 的形式 | CTA-group 和 multicast 新模式；只适配实际支持分支 | SDC copy Operation 能力表 | 支持模式运行；不支持模式编译/运行明确拒绝 |
| 34 pipeline；必须适配同步语义 | [include/cutlass/pipeline/sm100_pipeline.hpp:119](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp:119) | PipelineTmaUmmaAsync / PipelineUmmaAsync / PipelineCLCFetchAsync | acquire/commit/wait/release、index/phase 契约 | UMMA completion、TMA expected-bytes、cluster scope、barrier 存储与 arrival counts | 建议 sdc200_pipeline.hpp | load/MMA/epilogue 高并发和 phase wrap 压测 |
| 35 通用 pipeline；条件复用 / 必须适配后端 | [include/cutlass/pipeline/sm90_pipeline.hpp:1015](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:1015) | PipelineAsync / PipelineTmaAsync / PipelineTmaStore | 环形 state 与空满管理 | 内部仍依赖 barrier/TMA 原语；核对每种 commit/tail | 优先接口复用；替换底层同步 | 生产/消费多阶段和 tail 不早退 |
| 36 barrier/fence；必须适配 | [include/cutlass/arch/barrier.h:797](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/arch/barrier.h:797) | ClusterBarrier / ClusterTransactionBarrier / umma_arrive / fence_view_* | 同步接口与角色表达 | mbarrier phase、字节计数、UMMA arrival、TMEM load fence、SMEM async proxy 可见性 | SDC barrier/fence 后端 | 跨操作可见性 litmus、参与者计数、scope 与进展保证 |
| 37 named/order barrier；条件复用 / 必须适配后端 | [include/cutlass/pipeline/sm90_pipeline.hpp:1255](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:1255) | OrderedSequenceBarrier / barrier helpers | 顺序化 load 与角色协议 | 内部 named barrier 与线程集合/warp 大小 | SDC barrier traits | 可选 C load、load order、退出路径无死锁 |
| 38 cluster/election；必须核验 / 现文件非完成证明 | [include/cute/arch/cluster_sm90.hpp:180](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/cluster_sm90.hpp:180) | cluster_arrive/wait / block_rank_in_cluster / mapa / elect_one_sync | 统一 election/cluster API | warp 大小、active mask、cluster shared 寻址和内存屏障；替换内联 PTX | 已见 cluster_sdc200.hpp，但仍含 NVIDIA PTX | 部分 active mask、cluster rank、跨 CTA 可见性；include 不重定义 |
| 39 cluster SM100；条件复用 / 必须适配后端 | [include/cute/arch/cluster_sm100.hpp:49](D:/000_SDC/001TASK/02_Research/cutlass/include/cute/arch/cluster_sm100.hpp:49) | SM100 cluster helpers | 能力隔离入口 | 核对相关扩展指令与能力宏是否由 SDC 编译器支持 | SDC cluster extension 或显式禁用 | 仅启用已验收的 cluster 操作 |
| 40 register 调度；条件复用 / 必须适配后端 | [include/cutlass/arch/reg_reconfig.h:74](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/arch/reg_reconfig.h:74) | warpgroup_reg_alloc / warpgroup_reg_dealloc | 按角色调整资源的策略意图 | SDC 是否存在等价动态寄存器重分配；无则重算静态资源/占用率 | SDC resource policy / 合法无操作分支 | 编译资源报告、角色资源足够、性能基线 |
| 41 epilogue Builder；条件复用 / 必须适配后端 | [include/cutlass/epilogue/collective/collective_builder.hpp:70](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/epilogue/collective/collective_builder.hpp:70) | epilogue::collective::CollectiveBuilder | 输出/fusion/布局接口及 CollectiveOp 契约 | 接入 SDC 特化，确保 accumulator 和 Mainloop 匹配 | 建议 builders/sdc200_builder.inl | C=void、有 C、各 D layout 选择正确实现 |
| 42 epilogue 策略生成；条件复用 / 必须适配后端 | [include/cutlass/epilogue/collective/builders/sm100_builder.inl:593](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/epilogue/collective/builders/sm100_builder.inl:593) | EpilogueTileAuto / TmaWarpSpecialized1Sm / sm100_get_tmem_load_op | 按类型/tile 选择输出流程的结构 | TMEM T2R、subtile、SMEM swizzle、vector width、reuse、delay store 和资源 | 建议 sdc200_builder.inl | subtile 覆盖、C/D alias 和数据转换正确 |
| 43 epilogue dispatch；条件复用 / 必须适配后端 | [include/cutlass/epilogue/dispatch_policy.hpp:226](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/epilogue/dispatch_policy.hpp:226) | Sm100TmaWarpSpecialized / TmaWarpSpecialized1Sm | policy 提供 stage 与 schedule 常量 | 新增 SDC load/store stage 与访问策略 | 建议 Sdc200* policy | dispatch 与 kernel/Collective 相容 |
| 44 epilogue 主体；条件复用 / 必须适配后端 | [include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp:84](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp:84) | load / store / store_tail / FusionCallbacks / CopyOpT2R | 融合生命周期、分片输出流程 | TMEM ownership、T2R、C DMA、R2S、TMA store/fence/drain；无 TMEM 时改数据通路 | 建议 sdc200_epilogue_* | alpha/beta、C=void、残块、batch、输出类型与 reference 一致 |
| 45 fusion 架构回调；条件复用 / 必须适配后端 | [include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp:68](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp:68) | FusionCallbacks / sm100 visitor adapters | visit 生命周期和融合树组合 | 架构专用 broadcast/load/store、TMEM/SMEM 访问、归约和同步 | SDC callbacks；纯算术节点尽量复用 | bias/activation/辅助输出独立覆盖后组合 |
| 46 fusion 通用算术；优先复用 / 核验数值 | [include/cutlass/epilogue/fusion/operations.hpp:116](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/epilogue/fusion/operations.hpp:116) | FusionOperation / LinearCombination | alpha*acc+beta*C 等数学定义 | 数值精度、舍入、NaN/Inf、设备内建算术兼容 | 优先复用数学层 | 与明确参考精度/容差一致 |
| 47 数值转换；条件复用 / 必须适配后端 | [include/cutlass/numeric_conversion.h:850](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/numeric_conversion.h:850) | NumericConverter / NumericArrayConverter | 转换 API 和纯 C++ fallback | 内联 PTX、半精度/低位宽 pack、round/saturate 的目标实现 | SDC numeric builtin specialization | 边界值、舍入、饱和和向量 pack 位模式 |
| 48 Kernel 声明；优先复用公共接口 | [include/cutlass/gemm/kernel/gemm_universal_decl.h:57](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/gemm_universal_decl.h:57) | GemmUniversal<ProblemShape,Mainloop,Epilogue,Scheduler> | 用户模板签名与 composition 概念 | 声明本身少改；通过 SDC dispatch 接入特化 | 保留入口，新增 SDC kernel 特化 | 同样三参数组合可实例化 SDC kernel |
| 49 Kernel 聚合；条件复用 / 必须适配后端 | [include/cutlass/gemm/kernel/gemm_universal.hpp:67](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:67) | architecture kernel includes | 架构实现聚合方式 | 安全 include SDC 特化，避免 namespace 嵌套和 ODR 冲突 | include sdc200_gemm_* | 同时包含 NVIDIA/SDC 后端的编译隔离验证 |
| 50 Kernel 编排；必须适配或完整证明硬件契约兼容 | [include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:67](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:67) | Arguments / Params / SharedStorage / operator() / TMEM handoff | 参数聚合、tile 协作与所有权交接设计 | warp/thread/cluster 分工、SMEM/TMEM 容量、pipeline、scheduler、退出/释放重新核对 | 建议 sdc200_gemm_*；协议不同时重构 | 多 tile、多 stage、tail/异常/重复启动数值与进展 |
| 51 scheduler selector；条件复用 / 必须适配后端 | [include/cutlass/gemm/kernel/tile_scheduler.hpp:207](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:207) | TileSchedulerSelector / default void mapping | tag→scheduler 的接口 | 新增 SDC 默认映射；不让 arch::Sdc200 误走 Sm100 或缺失特化 | SDC static/software/CLC scheduler selector | 默认/显式 scheduler 都可构建且行为正确 |
| 52 persistent scheduler；必须适配硬件 CLC / 可换算法 | [include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:394](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:394) | PersistentTileSchedulerSm100 / fetch_next_work / CLC | WorkTileInfo、raster/swizzle 和坐标转换思想 | clusterlaunchcontrol.try_cancel/query_cancel 的 SDC 等价或软件替代；修复 workspace/参与角色 | 建议 SDC scheduler；先静态后动态 | tile 覆盖无遗漏/重复，batch 与退出正确 |
| 53 device adapter；条件复用 / 必须适配后端 | [include/cutlass/gemm/device/gemm_universal_adapter.h:123](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:123) | can_implement / get_workspace_size / initialize / update / run | 公共方法和有状态 Params handle | 架构数值分支、cudaFuncSetAttribute、occupancy、launch/error/stream；CudaHostAdapter 只是扩展点 | 公共接口 + SDC runtime/launcher 后端 | initialize/update/run 生命周期、错误传播和实际完成 |
| 54 全局入口；条件复用 / 必须适配后端 | [include/cutlass/device_kernel.h:118](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/device_kernel.h:118) | device_kernel<Operator> / Params / dynamic shared memory | wrapper 调 functor 的结构 | SDC kernel ABI、launch_bounds、grid_constant、动态共享内存对齐 | SDC device entry wrapper | 参数 layout 与动态 SMEM 正确传入 |
| 55 普通 launcher；条件复用 / 必须适配后端 | [include/cutlass/kernel_launch.h:76](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/kernel_launch.h:76) | kernel_launch / runtime launch APIs | 从 grid/block/SMEM/stream 提交 kernel 的接口 | SDC runtime launch/参数指针/异步错误与 PDL 支持边界 | SDC launcher 或有证据兼容 runtime shim | 单 CTA 与多 block 启动、属性/参数及错误正确 |
| 56 cluster launcher；必须适配或证明 runtime 兼容 | [include/cutlass/cluster_launch.hpp:248](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/cluster_launch.hpp:248) | ClusterLauncher::launch_with_fallback_cluster / cudaLaunchKernelExC | cluster/fallback 选择接口 | SDC cluster launch ABI、上限、grid 整除、preferred/fallback 和驱动支持 | SDC cluster launcher；不支持则禁用配置 | 1×4×1 可实际启动，fallback/非法配置正确处理 |
| 57 硬件信息；条件复用 / 必须适配后端 | [include/cutlass/kernel_hardware_info.h:43](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/kernel_hardware_info.h:43) | KernelHardwareInfo / query_device_multiprocessor_count | 信息结构和调用位置 | 设备 ID、SM/计算单元、cluster 范围与 runtime 查询；单位不盲目对应 SM | SDC hardware info provider | 查询值与真实资源及 scheduler 解释一致 |
| 58 host/device 内存；条件复用 / 必须适配后端 | [tools/util/include/cutlass/util/device_memory.h:61](D:/000_SDC/001TASK/02_Research/cutlass/tools/util/include/cutlass/util/device_memory.h:61) | device_memory::allocation / memcpy / synchronization | RAII、元素容量管理 | SDC malloc/free/copy/stream/error 与地址空间 ABI | SDC runtime shim / utility backend | 分配拷贝回读、零长度/错误路径与生命周期 |
| 59 HostTensor；条件复用 / 必须适配后端 | [tools/util/include/cutlass/util/host_tensor.h:403](D:/000_SDC/001TASK/02_Research/cutlass/tools/util/include/cutlass/util/host_tensor.h:403) | HostTensor / sync_host / sync_device | host 数据与布局组织、随机输入工具 | 底层 device allocation/copy API、格式表示 | 保留测试结构，接 SDC 内存工具 | A/B/D 数据传输与 batch stride 正确 |
| 60 Testbed；条件复用 / 必须适配后端 | [test/unit/gemm/device/gemm_testbed_3x.hpp:3828](D:/000_SDC/001TASK/02_Research/cutlass/test/unit/gemm/device/gemm_testbed_3x.hpp:3828) | TestAll / Testbed3x::run / TestbedImpl::verify / sufficient | 尺寸枚举、host reference 对照、GoogleTest 断言 | 修复当前缺失函数名；替换 CUDA API；将 skip 与通过分开；initialize 失败立即终止 | SDC testbed backend 或受控公共抽象 | 记录候选/skip/launch/数值数，强制配置不得静默跳过 |
| 61 具体测试；条件复用 / 必须适配后端 | [test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu:133](D:/000_SDC/001TASK/02_Research/cutlass/test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu:133) | TEST / Builders / GemmUniversal / EXPECT_TRUE | 测试表达形式和计算语义 | SDC arch/schedule/能力 guard/测试 target；多布局建立多个实例 | 建议 sdc200_tensorop_gemm/* | 每个承诺布局、tail、batch、K 长度均实际执行 |
| 62 主机参考；优先复用 / 保持参考独立 | [tools/util/include/cutlass/util/reference/host/gett.hpp:837](D:/000_SDC/001TASK/02_Research/cutlass/tools/util/include/cutlass/util/reference/host/gett.hpp:837) | host::Gemm3x / Gett | CPU 参考 GEMM 和 tensor 坐标语义 | 元素格式与 host 编译兼容；明确累加/舍入容差，防止跟随错误设备布局 | 优先复用独立 host reference | 小矩阵手算/独立参考 + SDC 比较 |
| 63 数值比较；优先复用 / 配置验收 | [tools/util/include/cutlass/util/reference/host/tensor_compare.h:179](D:/000_SDC/001TASK/02_Research/cutlass/tools/util/include/cutlass/util/reference/host/tensor_compare.h:179) | TensorEquals / relative comparison helpers | 相等/容差比较方法 | 调用方明确 tolerance、NaN/Inf、零附近误差及精度，禁止放宽掩盖错误 | 优先复用比较工具 | 故意错误数据能被检测，容差符合精度目标 |
| 64 构建测试 target；条件复用 / 必须适配后端 | [test/unit/gemm/device/sm100_tensorop_gemm/CMakeLists.txt:32](D:/000_SDC/001TASK/02_Research/cutlass/test/unit/gemm/device/sm100_tensorop_gemm/CMakeLists.txt:32) | SM100 unit-test target registration | 按架构组织目标和 GTest filter 的方式 | SDC toolchain、链接库、硬件 capability guard、测试运行器 | 建议 SDC 单独 target | CI/本机运行到真实 SDC kernel，非空测试或全 skip |
| 65 扩展范围；按需扩展 / 非本例必需 | [include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp:52](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp:52) | Stream-K / split-K reduction interfaces | 分解工作和归并的数学需求 | workspace、全局原子/内存序、fixup barrier、精度；本例默认路径不启用 | 后续 SDC StreamK backend | 独立验收后才能加入支持清单 |
| 66 CUTLASS 能力宏；条件复用 / 必须适配后端 | [include/cutlass/arch/config.h:87](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/arch/config.h:87) | CUTLASS_ARCH_MMA_*_SUPPORTED / *_ENABLED | 编译器/目标能力分离的结构 | SDC 编译器识别、目标 feature、host/device 两阶段宏，不让 host supported 误表示设备指令可用 | 明确 SDC 宏并与 cute/arch/config.hpp 联动 | host 预处理、device 预处理和实际机器指令三者一致 |
| 67 HostAdapter / Driver 接入；条件复用 / 必须适配后端 | [include/cutlass/cuda_host_adapter.hpp:152](D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/cuda_host_adapter.hpp:152) | CudaHostAdapter / CUTLASS_CUDA_DRIVER_WRAPPER_CALL | 可注入 launch/query 的接口思想 | Driver 动态解析、TensorMap 编码、cluster launch、occupancy；不能覆盖 device PTX 的语义 | SDC host adapter / driver wrapper backend | 所有被调用方法有实现，descriptor 与 kernel runtime ABI 相容 |
| 68 编译器 builtin 声明；外部编译器依赖 / 必须验收 | [D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h:897](D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h:897) | __sdc_uten_mma_bf16 / compiler lowering | 现有 builtin 声明形参契约 | 核验前端识别、后端 lowering、FP16/BF16 idesc、inputD 和完成语义；声明存在不证明生成指令正确 | 现有 SDC compiler header / 编译器后端 | 编译诊断 + 机器码 + 单 Atom overwrite/accumulate 数值 |

## 现有 SDC200 证据

- [MMA 包装](D:/lvz/sdc/cutlass/include/cute/arch/mma_sdc200_umma.hpp:108)：存在 `SDC200_MMA_F16BF16_SS::fma`，调用 `__sdc_uten_mma_bf16`。
- [builtin 声明](D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h:897)：形参是 `dTmem/aDesc/bDesc/iDesc/inputD`。声明对应不构成数值或完成语义验证。
- [SDC descriptor](D:/lvz/sdc/cutlass/include/cute/arch/mma_sdc200_desc.hpp:109)：存在 ABI 敏感 union，仍需位级对照。
- [SDC cluster/election](D:/lvz/sdc/cutlass/include/cute/arch/cluster_sdc200.hpp:180)：仍有 NVIDIA `elect.sync` 与 cluster PTX；本轮无法据此确认原生 SDC 支持。

## 能力扩展，按实际需求另立验收项

2SM MMA、sparse、FP8/FP6/FP4、block-scaled、mixed-input、pointer-array/grouped GEMM、复杂 fusion、PDL、RTC、graph capture、distributed GEMM 和 profiler dispatch 均不因本例通过而自动获得支持。扩展时从其独立 Builder/Collective/Kernel/Traits/Operation/测试入口继续审计；只实现本例时应明确拒绝不支持能力。

## 测试通过口径

当前 Testbed 对 `can_implement` 失败返回 true。SDC 验收应单独列出 requested、unsupported/skip、initialized、launched、synchronized、numerically_passed；承诺支持的配置跳过视为未完成。编译、运行、数值、机器码与性能是五种不同证据。本轮只有静态源码和文档图稿验证。

[返回完整说明](report.md) · [下载 UTF-8 CSV](sdc200-adaptation.csv)


## 验证与未完成项

{
  "table_rows": 68,
  "source_files": 70,
  "all_source_files_and_regex_anchors_exist": true,
  "diagrams": [
    {
      "name": "01-end-to-end",
      "nodes": 19,
      "edges": 12,
      "png": [
        2610,
        2130
      ]
    },
    {
      "name": "02-collective-handoff",
      "nodes": 21,
      "edges": 13,
      "png": [
        2850,
        2490
      ]
    },
    {
      "name": "03-concept-cards",
      "nodes": 12,
      "edges": 0,
      "png": [
        2700,
        2085
      ]
    },
    {
      "name": "04-sdc200-porting",
      "nodes": 14,
      "edges": 11,
      "png": [
        2700,
        2160
      ]
    }
  ],
  "xml_parse": true,
  "png_visual_review": "reviewed all four PNGs; revised launch routing and ready/empty label routing",
  "utf8": true,
  "compiled": false,
  "device_run": false,
  "numerical_validation": false,
  "sass_or_sdc_machine_code": false,
  "performance": false,
  "source_modified": false,
  "cross_computer_sync": false,
  "html_structure_and_local_image_paths": true,
  "csv_rows": 68,
  "all_source_hashes_unchanged_at_final_verification": true,
  "drawio_nodes_are_editable": true,
  "visual_differences": "Font glyphs, antialiasing, arrow tip geometry may differ between Pillow and diagrams.net."
}

未执行编译、设备运行、数值、机器码、性能验证；未更改 CUTLASS/SDC 源码；没有已验证授权的跨电脑同步渠道，故同步未完成。
