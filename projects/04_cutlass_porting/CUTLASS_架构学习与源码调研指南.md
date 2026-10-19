# CUTLASS 架构学习与源码调研指南

核实日期：2026-10-18。路线：全局架构 → 核心抽象 → GEMM 执行链 → GPU 硬件映射 → 性能优化 → 后端移植。

## 0. 使用范围、版本和推进方法

本指南以 NVIDIA 官方仓库提交 [`78e46228a1bfc5c972e2f9a634223af784cd6949`](https://github.com/NVIDIA/cutlass/commit/78e46228a1bfc5c972e2f9a634223af784cd6949) 为源码基线。它是本次核实时取得的 main HEAD，不是某个稳定发行版承诺。文档中的 latest 链接会更新；发生差异时，以本指南固定提交中的实现为核查对象。

三个问题贯穿全程：CUTLASS 如何设计？一个 GEMM 如何执行？如何修改、优化并迁移到另一硬件实现？

先明确四个边界：

- **3.x GEMM API 是接口架构名称。** 当前仓库继续使用这套 C++ 分层；不能从名称推断当前仓库主版本是 3。
- **五级抽象是 Device、Kernel、Collective、Tiled MMA/Copy、Atom。** Tensor、Layout 是贯穿底层的表达基础，不应机械添加为第六级调用层。[官方 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
- **CuTe C++ 与 CuTe DSL 分开追踪。** 前者进入 C++ 模板、traits 和 PTX wrapper；后者有 Python DSL、IR 和编译流程。CuTe DSL 不只是 Python 调用 C++ GemmUniversalAdapter 的包装。[官方 Python/DSL 概览](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- **编译期组合图与运行期执行图分开画。** Builder 在编译期选择类型；运行期不是依次调用五层普通虚函数。模板展开、内联和异步硬件执行会改变最终可见的调用形态。

每题使用一个独立研究记录，填齐后再勾选：

```text
题号 / 状态：未开始、阅读中、已定位、已验证
一句话问题：
当前假设与适用架构：
源码版本 / 文件 / 符号 / 行号：
输入 → 输出 / 前置条件：
编译期发生什么，运行期发生什么：
最小算例、图或实验：
证据与结论；未解决事项：
```

下面的“完成标志”是学习检查点，不表示已经完成了 GPU 实测。本次核实包含官方文档、仓库树及关键源文件；没有编译、运行或测量 GPU kernel。

## 1. 全局架构：先确定模块边界（Q01–Q08）

阅读入口：[仓库 README][R0]、[GEMM API 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)、[Example 49][R1]。

| 完成 | 题号与可直接发起的问题 | 源码调研动作与完成标志 |
|---|---|---|
| [ ] | **Q01：CUTLASS、CuTe、CUDA/PTX 分别解决什么问题？与 cuBLAS 的使用方式有何区别？** | 画出库、张量表达、编译工具链和硬件的关系；说明用户在编译期配置什么、在运行期提交什么。不要把 CUTLASS 模板本体、实例库和 profiler 混成一个运行时调度器。 |
| [ ] | **Q02：官方仓库的目录如何映射到产品与执行路径？** | 标出 `include/cutlass`、`include/cute`、`examples`、`test/unit`、`tools/library`、`tools/profiler`、`tools/util`、`python/CuTeDSL`；每个目录各找一个真实入口。[R0] |
| [ ] | **Q03：五级 GEMM 抽象的职责、输入输出和组合关系是什么？** | 用下方模块图解释每层；分别给出类型名称、负责的空间范围和同步责任。避免把软件层与 GPU 硬件层做一一对应。 |
| [ ] | **Q04：2.x 与 3.x GEMM API 在当前仓库中如何共存？** | 对照 `gemm_universal.h`、`gemm_universal.hpp` 和 Adapter 特化；从模板参数而非后缀猜测确认所走接口。[R2][R3] |
| [ ] | **Q05：一个 GEMM 类型是如何由两个 Collective 组合出来的？** | 在 Example 49 摘录 epilogue builder、mainloop builder、GemmUniversal、Adapter 四个类型别名；画出依赖图。[R1] |
| [ ] | **Q06：Builder、DispatchPolicy、KernelSchedule 和 TileScheduler 各决定什么？** | 分别标出类型构造、mainloop 分派、kernel 内部工作组织、工作 tile 分配；证明 Schedule 与 TileScheduler 是不同维度。[R4][R5][R6] |
| [ ] | **Q07：CuTe DSL 与 CuTe C++ 共享哪些概念，编译路线又在哪里分开？** | 分别画 C++ 模板路线和 Python DSL 路线，定位 DSL 的 `cute`、`nvgpu`、编译相关目录；不假定 Python DSL 必经 C++ Builder。[R20][R21] |
| [ ] | **Q08：先选哪个 GEMM 实例作为贯穿全文的标本？** | 冻结提交、GPU、工具链、数据类型、布局、ProblemShape、TileShape、ClusterShape、Schedule。选 Example 49 中一个显式 schedule 实例；不要同时追它的所有 runner。[R1] |

五级结构的阅读地图：

```text
Device      GemmUniversalAdapter       Host 参数、workspace、launch
   ↓ 选择的 kernel 类型
Kernel      GemmUniversal 特化         工作 tile、角色、collective 编排
   ↓ mainloop 与 epilogue
Collective  CollectiveMma / Epilogue    K tile 循环、数据搬运、融合、协作
   ↓ tiled 操作
Tiled       TiledMMA / TiledCopy        thread/value 与逻辑 tile 的映射
   ↓ atom 与 traits
Atom        MMA_Atom / Copy_Atom       指令操作及其布局、类型元信息
   ↓ 架构操作
arch wrapper → PTX → 编译后的机器指令 → GPU

CuTe Tensor / Layout / layout algebra 贯穿上述底层表达。
Pipeline、barrier、scheduler 是协作机制，不是新增的顺序调用层。
```

源码符号的准确拼写是 `cute::TiledMMA`、`cute::MMA_Atom`、`cute::TiledCopy`、`cute::Copy_Atom`；部分说明文字会使用 TiledMma/Mma_Atom 的概念性写法，搜索源码时应使用实际符号。[R8][R9]

**阶段产物：** 一张架构图、一张目录索引、一份标本配置。能够回答“新增功能应从哪层开始调查”。

## 2. 核心抽象：先把数据和线程映射算明白（Q09–Q16）

阅读入口：[CuTe Layout 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)、[CuTe MMA Atom 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html)、[CuTe C++ 教程目录][R10]。

| 完成 | 题号与可直接发起的问题 | 源码调研动作与完成标志 |
|---|---|---|
| [ ] | **Q09：Shape、Stride、Layout 分别表示什么？Layout 如何把坐标映射成索引？** | 手算 `Shape=(2,3), Stride=(3,1)` 的全部坐标与偏移；再构造嵌套 shape。解释 shape、rank、size、cosize 的区别。[R7] |
| [ ] | **Q10：Tensor 的 engine 与 Layout 怎样组合？视图什么时候分配或共享存储？** | 比较全局内存 view、shared-memory view、register fragment；追踪创建方式与存储所有权，不把所有 Tensor 都理解为拥有内存的容器。[R7] |
| [ ] | **Q11：composition、logical_divide、tile、slice 如何改变索引关系？** | 从一个小矩阵推导 tile 坐标、tile 内坐标及偏移；把输出 shape/stride 写在每一步旁边。以具体算例验证，而非只背 API 名称。[R7][R10] |
| [ ] | **Q12：线程、lane 和每线程 value 如何共同拥有一个 tile？** | 展示 thread/value 到逻辑坐标的映射，列出两个线程持有的元素；区分存储布局与线程所有权映射。[R8][R9] |
| [ ] | **Q13：TiledCopy 如何组合 Copy_Atom、线程布局和 value 布局？** | 沿 `get_slice`、`partition_S`、`partition_D`、`copy` 查看源/目标 partition；说明向量宽度、对齐和连续访问的前置条件。[R9][R10] |
| [ ] | **Q14：TiledMMA 如何把单条 MMA 指令扩展到更大的 tile？** | 从 `get_slice`、`partition_A/B/C`、fragment 构造追到 `gemm`；列出 atom 的 MNK、参与线程与重复维度。[R8][R11] |
| [ ] | **Q15：MMA operation、MMA_Traits、MMA_Atom 三者如何分工？** | 把指令 wrapper 的参数、traits 中的 ThrID/A/B/C layout 和 atom 接口对应起来；证明只写 asm 还不能完成可组合 atom。[R8][R12][R13] |
| [ ] | **Q16：逻辑 GEMM 布局、shared-memory swizzle、指令 descriptor 有什么联系？** | 对照一个具体 A/B tile，说明 global layout 到 SMEM layout 再到指令操作数的转换；解释合法性约束与 bank conflict 分别在哪里体现。[R12][R14] |

**阶段产物：** 一个带坐标的 tensor 分块图、一份 thread/value 映射表。能够手算数据属于哪个线程、位于何处、以何种方式交给指令。

## 3. GEMM 执行链：把类型选择和运行时数据流贯通（Q17–Q24）

阅读入口：[Adapter][R2]、[kernel 总入口][R3]、[Hopper cooperative kernel][R6]、[Hopper TMA/GMMA collective][R14]。

| 完成 | 题号与可直接发起的问题 | 源码调研动作与完成标志 |
|---|---|---|
| [ ] | **Q17：Host 侧如何表示 M/N/K/L、A/B/C/D、stride、alpha/beta？** | 从 Example 49 的 `Arguments` 开始逐字段追到 Kernel/Collective；区分逻辑 B 的 K×N 记法与内核常见的 N×K mode 组织。[R1][R6] |
| [ ] | **Q18：can_implement、workspace 和 initialize 各检查或准备什么？** | 查看返回状态、空间大小、workspace 初始化、`to_underlying_arguments`；说明 workspace 由谁分配、何时可复用，不假设 Adapter 自动拥有分配空间。[R2] |
| [ ] | **Q19：Arguments 如何变成 Params？TMA descriptor 在哪里形成？** | 同时追 Kernel 和 Collective 的 lowering；为指针、stride、descriptor、scheduler 参数画字段流向。记录哪些参数是 Host 预计算。[R2][R6][R14] |
| [ ] | **Q20：Adapter 如何确定 grid、block、cluster 和动态共享内存并启动 kernel？** | 从 `run` 追踪实际 launch 分支和 `device_kernel` wrapper；按选定架构解释 extended/cluster launch，避免把某一分支当成全部实现。[R2][R15] |
| [ ] | **Q21：GemmUniversal 如何划分 producer/consumer 等角色并分配工作 tile？** | 在选定特化中找到线程角色、scheduler 查询、mainloop 和 epilogue 调用；说明 persistent 与非 persistent 的工作获取方式。[R6] |
| [ ] | **Q22：Collective mainloop 如何推进 K tile 并重叠搬运与计算？** | 画至少三个 K tile 的时序，标出 producer acquire、TMA 发起、consumer wait、MMA、release；检查尾部 drain。明确 pipeline stage 是复用槽位，不能简单等同于某个固定 K tile。[R14][R16] |
| [ ] | **Q23：cute::gemm/copy 如何落到 atom、traits 和 PTX wrapper？** | 从实际模板参数出发找重载/特化，记录最终 operation 类型和 asm；不要仅凭文件名断言实际使用了某条指令。[R8][R9][R11][R12][R13] |
| [ ] | **Q24：累加结果如何经过 epilogue、融合、类型转换和写回 D？** | 分开画 accumulator 到 D 的路径及 C 输入路径；检查 beta=0、C 可选、边界 tile、输出向量化。若 kernel 为异步 launch，区分提交成功与执行完成。[R17][R18] |

主线核查顺序：

```text
【编译期】
Example 配置
 → epilogue CollectiveBuilder → CollectiveEpilogue
 → mainloop CollectiveBuilder → CollectiveMainloop
 → GemmUniversal<ProblemShape, Mainloop, Epilogue, TileScheduler>
 → GemmUniversalAdapter<GemmKernel>

【Host 运行期】
Arguments
 → can_implement / get_workspace_size / workspace 分配
 → initialize → Kernel::to_underlying_arguments → Params
 → run → CUDA launch → device_kernel wrapper

【Device 运行期；具体步骤可能并行、交错】
Kernel::operator()
 → scheduler / 线程角色 / tile 坐标
 → Collective load 与 MMA → epilogue → D

【编译后硬件证据】
实际 CuTe operation 类型 → inline PTX / intrinsic
 → 生成的 PTX 与目标机器码 → 反汇编和 profiler 证据
```

**阶段产物：** 一份精确到文件、符号和行号的调用链，加上一份“同一块 A/B/C/D 数据”的生命周期图。最后重新从 Host 到硬件，再从硬件需求倒推至 Builder，把两条线核对一次。

## 4. GPU 硬件映射：从指令语义向上验证（Q25–Q32）

阅读入口：[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)、[Hopper atom/traits][R12][R13]、[Blackwell UMMA wrapper][R19]、[SM100 官方说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)。

| 完成 | 题号与可直接发起的问题 | 源码调研动作与完成标志 |
|---|---|---|
| [ ] | **Q25：软件五级结构分别涉及 grid、cluster、CTA、warpgroup、warp、lane 中的哪些范围？** | 为标本标注参与线程、同步范围、工作分配范围；说明 Collective 不应被定义成固定一个 warp，也不能给每个软件层硬配一个硬件层。 |
| [ ] | **Q26：SM80 的 mma.sync、SM90 的 wgmma.mma_async、SM100 的 tcgen05.mma 有何关键差异？** | 比较参与方式、输入来源、累加器位置、完成通知。以选定指令变体为单位查 PTX；SM80 常见 warp MMA、SM90 warpgroup MMA 和 SM100 CTA-group 语义不能互换。[R13][R19][R22] |
| [ ] | **Q27：cp.async 与 TMA 的搬运模型和同步协议分别是什么？** | 比较逐线程发起拷贝与 descriptor/bulk tensor 路径，定位地址生成、越界处理、multicast、barrier 关联；分别画 global→SMEM 路径。[R23][R14] |
| [ ] | **Q28：为什么部分指令需要 SMEM descriptor，而非普通寄存器数组？** | 解读一个实际 descriptor 的布局/地址约束；指出 A/B 输入来源随 SS、RS 或其他变体变化，不能一概而论。[R12][R13][R19] |
| [ ] | **Q29：mbarrier、phase、transaction bytes 和异步 MMA wait/commit 各保证什么？** | 区分数据搬运完成、MMA 完成、槽位可覆盖三件事；检查 barrier 初始化、参与数和 phase 翻转，说明少一次等待可能破坏哪项依赖。[R14][R16] |
| [ ] | **Q30：SM100 的 TMEM 如何改变 accumulator 和 epilogue 路径？** | 用 Blackwell CuTe 教程追踪 TMEM 分配、MMA、读取和释放，解释 1SM/2SM 示例的协作差异；不要将其直接套到所有 Blackwell 型号。[R19][R24] |
| [ ] | **Q31：cluster multicast、共享内存复用和 tile 调度如何共同影响数据流？** | 从 ClusterShape 推出 A/B 哪些 tile 可复用、哪些 CTA 接收数据；把减少搬运与新增同步/调度约束分别列出。[R6][R14][R24] |
| [ ] | **Q32：如何验证源码意图确实变成目标 GPU 指令？** | 保存编译目标、编译参数、PTX、SASS；在工具链支持时用 cuobjdump 或 nvdisasm 对照。区分源文件内 asm、PTX 虚拟 ISA、SASS 机器指令，分别记录证据。 |

架构对照的起点，而非所有 kernel 的统一执行公式：

| 对照对象 | 常见计算路径 | 重点核对 |
|---|---|---|
| SM80 Tensor Core 教程 | SMEM→寄存器 fragment→warp MMA | lane/value layout、ldmatrix、cp.async、register accumulator |
| SM90 TMA/GMMA SS 标本 | TMA→SMEM→warpgroup MMA | descriptor、producer/consumer、异步完成、register accumulator |
| SM100 tcgen05 标本 | TMA/其他供数路径→选定 MMA 变体→TMEM accumulator | TMEM 生命周期、CTA group、完成通知、TMEM→epilogue |

**阶段产物：** 一张包含存储位置、指令、线程参与范围、同步对象的硬件数据流图。没有目标 GPU 时，可以完成静态推导；执行行为和性能结论保持“未实测”。

## 5. 性能优化：把每次调参变成可验证假设（Q33–Q40）

阅读入口：[Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)、[CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)、[profiler 源码目录][R25]。旧式 CTA/warp 分块讲解可帮助理解复用，但 SM90/SM100 的异步路径需回到对应实现核实。

| 完成 | 题号与可直接发起的问题 | 源码调研动作与完成标志 |
|---|---|---|
| [ ] | **Q33：如何建立可信的正确性和性能基线？** | 记录 GPU/工具链/提交、布局/精度/误差阈值、warmup、重复次数和统计量；区分 kernel 时间、端到端时间与首次编译/初始化。对照 reference 并覆盖非整齐尺寸。 |
| [ ] | **Q34：当前 GEMM 受算力、带宽、并行度还是同步限制？** | 先计算 dense GEMM 约 2MNK FLOP 和明确假设下的最低数据流量，再与实测流量、计算单元利用率、stall、时间比较；不把理论最小字节数当成真实 DRAM 流量。 |
| [ ] | **Q35：TileShape 的 M/N/K 各改变哪些复用和资源开销？** | 一次只改一个维度，记录寄存器、SMEM、tile 数、边界浪费和时间；用小矩阵与大矩阵对照，避免仅优化一个整齐尺寸。 |
| [ ] | **Q36：pipeline stage 数为何不是越多越好？** | 根据 A/B tile 大小估算每 stage 存储，再加 epilogue carveout、barrier 等开销；对照 Builder 的 StageCount 选择。说明延迟隐藏与占用资源的交换。[R1][R4][R16] |
| [ ] | **Q37：warp specialization、pingpong、cooperative、persistent 分别解决什么问题？** | 每次比较受支持的一对 schedule，检查线程角色、寄存器预算、工作复用和尾部气泡；不把不同命名简单排成性能等级。[R1][R6] |
| [ ] | **Q38：何时调查 Split-K、Stream-K、grouped GEMM 或不同 tile scheduler？** | 从小 M/N 大 K、问题规模不均、tile 不足等现象出发；记录可用实现、额外归约、workspace、同步和数值顺序变化。[R5][R6] |
| [ ] | **Q39：epilogue 融合、swizzle、对齐和向量化各能消除哪类瓶颈？** | 为每个优化写明减少的流量/指令及新增资源；融合前后核对数值、广播语义和计时口径，避免把少一次 launch 与主循环提速混淆。[R17][R18] |
| [ ] | **Q40：如何证明优化有效且可复现？** | 固定其余条件并进行 A/B 对照，保存正确性、时间分布、编译资源报告和 profiler 证据；用多个 shape 验证收益边界。负结果也记录并解释。 |

每次优化填写：`现象 → 瓶颈假设 → 源码位置 → 单一改动 → 正确性 → 时间/资源/计数器 → 结论与适用范围`。理论吞吐上限、实际吞吐和利用率采用同一精度、稀疏性及运算计数口径。

**阶段产物：** 一份可复现的基线和至少一组有原因解释的优化对照。若没有 GPU，这一阶段先形成实验设计，不填写虚构结果。

## 6. 后端移植：先明确可保留的语义，再重建执行契约（Q41–Q48）

这里的“后端”要先拆开：同一 NVIDIA 平台新增指令/架构；跨厂商硬件；CuTe DSL 编译后端。它们复用的层次与工作量不同。下面是基于现有分层推导的工程路线，**不表示官方提供跨厂商后端插件接口**。

| 完成 | 题号与可直接发起的问题 | 源码调研动作与完成标志 |
|---|---|---|
| [ ] | **Q41：目标是扩展 NVIDIA 架构、跨厂商 GPU，还是迁移 DSL 编译链？** | 写清目标硬件/编译器、必须兼容的上层 API、精度和性能目标，以及允许重写的层。先定义一个最小 dense GEMM，不从全部类型和融合功能开始。 |
| [ ] | **Q42：目标硬件与 NVIDIA 的执行和内存语义差在哪里？** | 建 capability 表：subgroup 大小、矩阵指令形状/布局、地址空间、异步 copy、同步范围/内存序、cluster 类能力、寄存器和本地存储限制。明确不支持项的替代方案。 |
| [ ] | **Q43：新增 MMA/Copy operation 和 traits 应满足哪些契约？** | 先实现并验证最小矩阵指令与拷贝原语，再验证 fragment 的线程/值分布、类型、对齐和参与要求；逐项测试 atom，不能只替换 PTX 字符串。[R8][R9][R12][R13] |
| [ ] | **Q44：CuTe 的哪些索引代数可以保留，哪些依赖编译器和设备语义？** | 盘点纯 Shape/Stride/Layout 运算与 CUDA 修饰、地址空间、intrinsic、warp 特性；分别做 host 索引测试和 device 编译试验，不预设 include/cute 全部可直接迁移。[R7] |
| [ ] | **Q45：没有 TMA、WGMMA、TMEM 或 cluster 时，mainloop 和 pipeline 如何重建？** | 以同步、正确的加载/计算/写回为起点，再引入目标平台可用的异步路径；按真实完成通知重写 acquire/wait/release，重新设计 layout、stage 和累加器位置。[R14][R16][R19] |
| [ ] | **Q46：Kernel、scheduler、epilogue 和 Host launch 分别要改什么？** | 标出 CUDA runtime、launch、共享内存配置、硬件信息、workspace、线程角色和归约路径；为目标平台构造完整 Host→Kernel→D 最小闭环。[R2][R6][R15][R17] |
| [ ] | **Q47：怎样把新实现接入策略选择与工程工具？** | 同平台扩展可调查 ArchTag、DispatchPolicy、Collective/Builder 特化及编译开关；跨平台需同步评估 runtime、工具链、测试、实例生成与 profiler，不能只增加 ArchTag。[R4][R5][R25] |
| [ ] | **Q48：若迁移 CuTe DSL，哪些编译环节必须核实？如何验收？** | 从 Python API/operation 到 IR、lowering、目标代码生成、runtime 和打包逐环检查公开源码与外部依赖边界。若某环节不在公开仓库中，记录缺口；不把概念相似视为已有可替换后端。[R20][R21] |

推荐里程碑：

| 里程碑 | 必须交付的证据 |
|---|---|
| P0：语义最小闭环 | 一个固定 shape 的 dense GEMM，通过独立 reference；明确数据类型与布局。 |
| P1：原语契约 | MMA/Copy atom、layout/fragment、同步和内存序的小范围验证。 |
| P2：通用正确性 | 边界尺寸、K 尾部、stride、alpha/beta、C/D 组合及承诺支持的模式；反复运行检查异步缺陷。 |
| P3：性能路径 | 逐步恢复 tiled copy/MMA、pipeline 和调度；每步均有基线和资源证据。 |
| P4：工程接入 | 构建、测试、实例选择、错误报告和 profiler；列清不支持的能力与原因。 |

**阶段产物：** 一份按层拆解的移植差异表、一条最小可运行路径、带验收条件的里程碑。跨厂商移植应把“保留上层表达”与“重用现有 CUDA 实现”分开估计。

## 7. 实际推进顺序与延伸分支

问答主顺序始终按 Q01→Q48。执行时允许做短的验证回环：Q23 遇到指令布局，就先借 Q26/Q28 核查，再回到 Q24 完成执行链；Q32 完成后，重新从 Q17 到 Q24 连走一次。

建议三个贯穿标本：

1. **主线标本：Hopper Example 49。** 显式选定一个 schedule，学习编译期组合、Host API、Collective 与 pipeline。[R1]
2. **基础对照：CuTe SM80 教程。** 用较直接的线程 fragment 映射理解 copy/MMA atom，减少同时理解多种异步硬件的负担。[R10][R22]
3. **进阶对照：Blackwell Example 71 与 CuTe Blackwell 教程。** 引入 tcgen05、TMEM、1SM/2SM；回看哪些层保持接口、哪些层重写实现。[R24][R26]

若实际目标只有 SM100，可直接把主标本换成 Example 71，但保留 SM80/SM90 差异核查。不要把不同标本的类型参数拼成一条不存在的调用链。当前 Example 49 包含多种 runner、自动和显式策略；只读默认 Auto 不能保证完整理解自己最终选中的特化。

本目录已有 `SM100_f16_GEMM_64x128x64_1SM_调用链指南.md`，其声明基线是 `ffa119a1255d`，与本指南提交不同。可作为 Q17–Q32 的专项延伸；其中行号、分派条件和类型需要回到它自己的版本核查。

完成基础链后再展开：低精度/块缩放、稀疏 MMA、grouped/batched GEMM、复杂 epilogue、卷积或 attention、CuTe DSL。每个分支都复用“配置 → 调用链 → 数据流 → 指令 → 正确性/性能”的记录模板。

逐题交互时可直接使用：

> 从 Qxx 开始，按“概念结论 → 固定提交源码入口 → 类型与参数展开 → 小算例或时序图 → 验证方法 → 完成标志”讲解；标明编译期与运行期，以及哪些结论只适用于当前架构和 schedule。

## 8. 已核实的官方源码索引

以下链接均固定到同一提交。目录或文件的存在已按官方 Git tree 核对；Adapter、Builder 总入口、Example 49/71、Hopper cooperative kernel、Hopper TMA/GMMA collective、GMMA traits 和 SM100 UMMA wrapper 另做了内容检查。其余索引是后续逐题阅读入口，不表示本次已经逐行审计。

| 索引 | 文件或目录 | 用途 |
|---|---|---|
| R0 | [README][R0] | 仓库组成、C++ 与 Python 路线 |
| R1 | [49_collective_builder.cu][R1] | Hopper 完整组装与多种 schedule 实例 |
| R2 | [gemm_universal_adapter.h][R2] | can_implement、workspace、initialize、run |
| R3 | [kernel/gemm_universal.hpp][R3] | kernel 总入口及特化包含关系 |
| R4 | [collective/collective_builder.hpp][R4] | 各架构 builder 接入；已有 SM100/SM120 等路径 |
| R5 | [gemm/dispatch_policy.hpp][R5] | mainloop 分派及 schedule 标签 |
| R6 | [sm90_gemm_tma_warpspecialized_cooperative.hpp][R6] | Hopper cooperative kernel 实现 |
| R7 | [include/cute][R7] | Layout、Tensor、布局代数 |
| R8 | [mma_atom.hpp][R8] | MMA_Atom、TiledMMA |
| R9 | [copy_atom.hpp][R9] | Copy_Atom、TiledCopy |
| R10 | [examples/cute/tutorial][R10] | layout/copy/GEMM 教学入口 |
| R11 | [algorithm/gemm.hpp][R11] | CuTe GEMM 分解与分派 |
| R12 | [mma_traits_sm90_gmma.hpp][R12] | Hopper MMA 的布局、类型等元信息 |
| R13 | [mma_sm90_gmma.hpp][R13] | WGMMA PTX wrapper |
| R14 | [sm90_mma_tma_gmma_ss_warpspecialized.hpp][R14] | TMA、GMMA 与 K tile 流水线 |
| R15 | [device_kernel.h][R15] | device kernel wrapper |
| R16 | [include/cutlass/pipeline][R16] | 架构相关 pipeline 协议 |
| R17 | [epilogue/collective][R17] | 输出 collective 和 builder |
| R18 | [epilogue/fusion][R18] | 融合表达与实现 |
| R19 | [mma_sm100_umma.hpp][R19] | tcgen05.mma wrapper，含不同操作数变体 |
| R20 | [python/CuTeDSL/cutlass][R20] | DSL API、编译相关 Python 代码 |
| R21 | [examples/python/CuTeDSL][R21] | Python DSL 标本 |
| R22 | [sgemm_sm80.cu][R22] | SM80 CuTe GEMM 教程；结合指令文件分析 |
| R23 | [copy_sm80.hpp][R23] | SM80 架构 copy 操作 |
| R24 | [tutorial/blackwell][R24] | MMA、TMA、multicast、2SM、epilogue 教程 |
| R25 | [tools/profiler][R25] | profiler 工程入口 |
| R26 | [71_blackwell_gemm_with_collective_builder.cu][R26] | SM100 Builder 组装示例 |

早期文档中“3.0 Builder 仅支持 SM90”描述其当时状态；本基线 `collective_builder.hpp` 明确包含 `sm100_umma_builder.inl` 等实现，不能沿用那句话判断当前支持范围。支持某架构也不等于支持所有类型、layout、schedule 的任意组合，应继续检查实际 builder 特化和约束。[R4]

[R0]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/README.md
[R1]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu
[R2]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/gemm/device/gemm_universal_adapter.h
[R3]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/gemm/kernel/gemm_universal.hpp
[R4]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/gemm/collective/collective_builder.hpp
[R5]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/gemm/dispatch_policy.hpp
[R6]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
[R7]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute
[R8]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/atom/mma_atom.hpp
[R9]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/atom/copy_atom.hpp
[R10]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/examples/cute/tutorial
[R11]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/algorithm/gemm.hpp
[R12]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/atom/mma_traits_sm90_gmma.hpp
[R13]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/arch/mma_sm90_gmma.hpp
[R14]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
[R15]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/device_kernel.h
[R16]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/pipeline
[R17]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/epilogue/collective
[R18]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cutlass/epilogue/fusion
[R19]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/arch/mma_sm100_umma.hpp
[R20]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/python/CuTeDSL/cutlass
[R21]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/examples/python/CuTeDSL
[R22]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/examples/cute/tutorial/sgemm_sm80.cu
[R23]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/include/cute/arch/copy_sm80.hpp
[R24]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/examples/cute/tutorial/blackwell
[R25]: https://github.com/NVIDIA/cutlass/tree/78e46228a1bfc5c972e2f9a634223af784cd6949/tools/profiler
[R26]: https://github.com/NVIDIA/cutlass/blob/78e46228a1bfc5c972e2f9a634223af784cd6949/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu
