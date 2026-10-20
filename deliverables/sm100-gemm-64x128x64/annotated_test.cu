// 本文件是原测试函数的完整中文注释版，供替换原函数或阅读使用。
// 它不是独立编译单元：依赖原 f16_f16_void_f32.cu 的头文件、
// using namespace cute、GoogleTest 环境和目标架构编译配置。
// 本轮仅核对静态源码，未重新编译或在 GPU 上执行。

// TEST 是 GoogleTest 的测试注册宏：两个参数分别是测试套件名和测试名。
// 完整过滤名：
// SM100Only_Device_Gemm_f16t_f16n_void_f32t_tensor_op_f32.64x128x64_1x4x1_1sm
// 名字只是标识；真正决定内核行为的是下面的模板参数。
TEST(SM100Only_Device_Gemm_f16t_f16n_void_f32t_tensor_op_f32, 64x128x64_1x4x1_1sm) {
  // A：数学形状 [M,K]，输入元素为 IEEE 风格 FP16。
  // using 创建类型别名；这里没有分配矩阵，也没有启动 GPU。
  using ElementA = cutlass::half_t;
  // 对齐单位是“元素”：8 个 FP16 = 16 字节 = 128 bit。
  // 这是 Builder 选择搬运实现及运行时参数必须满足的对齐契约。
  constexpr int AlignA = 8;
  // 行主序：A(m,k) 的元素偏移为 m*lda+k，K 方向连续。
  using GmemLayoutA = cutlass::layout::RowMajor;

  // B：数学形状 [K,N]，对齐同样是 8 个 FP16，即 16 字节。
  constexpr int AlignB = 8;
  using ElementB = cutlass::half_t;
  // 列主序：B(k,n) 的元素偏移为 k+n*ldb，K 方向连续。
  // CuTe 内部把 B 按 [N,K,L] 表示；这不要求真的转置一份 B。
  using GmemLayoutB = cutlass::layout::ColumnMajor;

  // C 是 epilogue 的可选源矩阵；void 表示禁用 C 数据输入。
  // 此处的 C 与 MMA 内部名为 C 的“累加操作数”不是同一层概念。
  using ElementC = void;
  // 接口仍要求给出占位的 C 对齐参数；不是 sizeof(void)，也不会申请 C。
  constexpr int AlignC = 4;
  // 同理，保留 C 布局标签以满足 Builder 接口；没有实际 C 读取。
  using GmemLayoutC = cutlass::layout::RowMajor;

  // D：真正的输出矩阵 [M,N]，每个元素 32 bit。
  using ElementD = float;
  // 4 个 FP32 = 16 字节；注意这里与 A/B 的元素数不同，但字节数相同。
  constexpr int AlignD = 4;
  // D(m,n) 的元素偏移为 m*ldd+n，N 方向连续。
  using GmemLayoutD = cutlass::layout::RowMajor;

  // Tensor Core 用 FP16 输入乘法并以 FP32 累加；SM100 此路径的累加器驻留 TMEM。
  using ElementAccumulator = float;
  // epilogue 中 alpha 缩放等运算的精度；独立于输入类型、累加类型和输出类型。
  using ElementCompute = float;

  // Shape 是 CuTe 的形状类型；_64 等是编译期整数类型，不是运行时 int 值。
  // M/N 为本路径的 MMA 分块尺寸，K=64 为 mainloop 每个 K tile 的跨度。
  // 本配置为 1SM，因此一个 CTA 对应输出 64x128 的 tile。
  // FP16 MMA Atom 的 K=256/16=16；一个完整 K tile 沿 K 包含四个 Atom。
  // 整个 GEMM 的实际 M、N、K 由 TestAll 在运行时提供，可大于或小于此 tile。
  using MmaTileShape_MNK = Shape<_64,_128,_64>;
  // 一个 cluster 含 1*4*1=4 个 CTA，沿输出 N 方向排列。
  // 四个 CTA 复用同一块 A，分别读取各自的 B，适合向四个 CTA multicast A。
  // K=1 表示 cluster 不沿 K 扩展；这里没有请求 split-K。
  // 1SM 描述 MMA 的 CTA 协作粒度，并不表示整个 cluster 只用一个 SM。
  using ClusterShape_MNK = Shape<_1,_4,_1>;

  // 先在编译期构造 epilogue 类型，主循环稍后要用它的 SharedStorage 大小。
  // Builder 是模板元编程工厂；typename 表明 CollectiveOp 是其嵌套类型。
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,                    // 架构标签：选择 SM100 实现族。
      cutlass::arch::OpClassTensorOp,          // 运算类别：Tensor Core。
      MmaTileShape_MNK,                        // 与主循环约定同一 MMA tile。
      ClusterShape_MNK,                        // 与主循环约定同一 CTA cluster。
      cutlass::epilogue::collective::EpilogueTileAuto,
                                               // 自动选择输出子分块，不是 GEMM 全局尺寸。
      ElementAccumulator,                     // 从 TMEM 读取的累加值是 float。
      ElementCompute,                         // epilogue 运算精度为 float。
      ElementC, GmemLayoutC, AlignC,           // C 输入描述；void 禁用源矩阵。
      ElementD, GmemLayoutD, AlignD,           // D 输出描述：FP32、行主序、4 元素对齐。
      cutlass::epilogue::collective::EpilogueScheduleAuto
                                               // 编译期选调度；此配置落到 TmaWarpSpecialized1Sm。
      // 省略的 FusionOp 默认是 LinearCombination。
      // 一般语义 D=alpha*Acc+beta*C；C=void 时退化为 D=alpha*Acc。
    >::CollectiveOp;                            // 取出最终 collective epilogue 类型。

  // 编译期构造 mainloop：决定 A/B TMA、SMEM 布局、UMMA 和流水线策略。
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,                    // 目标架构。
      cutlass::arch::OpClassTensorOp,          // Tensor Core 主循环。
      ElementA, GmemLayoutA, AlignA,           // A：FP16、行主序、8 元素对齐。
      ElementB, GmemLayoutB, AlignB,           // B：FP16、列主序、8 元素对齐。
      ElementAccumulator,                     // FP32 累加。
      MmaTileShape_MNK,                        // 64x128x64 的 mainloop 分块。
      ClusterShape_MNK,                        // 1x4x1 的 cluster。
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          // sizeof 得到 epilogue 共享存储所需字节数；typename 标明嵌套类型。
          // static_cast<int> 转换为该策略所需的整型模板实参。
          // 先预留这些字节，再计算 A/B 能安排多少级 SMEM 流水线缓冲。
          // Builder 还会扣除 kernel 层 barrier/scheduler 等存储开销。
          // 这是编译期容量预算，不是在这里运行 cudaMalloc 或申请 SMEM。
      cutlass::gemm::collective::KernelScheduleAuto
          // 编译期选择主循环调度。ClusterM=1、TileM=64，使这里采用 1SM MMA。
    >::CollectiveOp;                            // 取出最终 collective mainloop 类型。

  // 组合设备端内核类型。定义 using 仍不执行 GEMM。
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,                   // 运行时问题形状 (M,N,K,L)，L 是 batch 数。
                                               // 四个 int 不代表固定的四个具体维度值。
      CollectiveMainloop,                     // A/B 搬运、K 循环和 MMA。
      CollectiveEpilogue                      // 累加结果变换及 D 写回。
      // 省略的 TileScheduler 模板参数默认为 void；SM100 选择持久化调度器。
    >;

  // 设备内核的主机端适配器，提供 Arguments、can_implement、
  // get_workspace_size、initialize、run 等接口。
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // 真正进入测试执行：分配并初始化张量、组织多组问题尺寸和调度参数，
  // 经适配器启动内核，同步后对比参考 GEMM，并汇总 bool 结果。
  // 默认 alpha=1；由于 ElementC=void，默认 beta=0；默认采用相对误差检查。
  // TestAll 不是只运行一次 64x128x64，也不是基准性能测量。
  auto pass = test::gemm::device::TestAll<Gemm>();

  // GoogleTest 非致命断言：若 pass=false，标记本测试失败。
  // 注意测试框架的“不支持参数”分支可能返回 true；pass 不能单独证明每组都执行。
  EXPECT_TRUE(pass);
}
