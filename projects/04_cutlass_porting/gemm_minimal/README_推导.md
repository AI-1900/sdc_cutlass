# 128×256×64 GEMM：从分块到线程地址

本文与 `gemm_128x256x64.cu` 一一对应。把 128×256×64 解释为完整问题的 `(M,N,K)`，计算 `C=A×B`，alpha=1、beta=0。A/B 为 FP16，累加与 C 为 FP32。这里选择一种便于手算、可在本机 RTX 4060 上运行的 SM80 MMA 分块；GEMM 尺寸本身并不唯一决定分块。

实际验证环境：CUTLASS 提交 `ffa119a1255d78998536107466cc7097ecefa393`、CUDA 13.4、MSVC 14.44、RTX 4060，编译目标 sm_89。`SM80_...` 是采用的指令 atom 名称，并不表示程序只能运行在 A100 上。

## 1. 先固定矩阵与存储约定

数学表达式：

```text
C(m,n) = sum[k=0..63] A(m,k) * B(k,n)
A: 128×64，行主序
B: 64×256，列主序
C: 128×256，行主序
```

CuTe 的 GEMM 接口把两路输入表示为 `(M,K)` 和 `(N,K)`。所以 B 的 Tensor 写成 `B_view(n,k)=B_math(k,n)`；这只是坐标命名与视图的变化，没有执行数据转置。

| Tensor | shape | stride（元素单位） | 线性元素偏移 |
|---|---|---|---|
| A | (128,64) | (64,1) | 64m+k |
| B_view | (256,64) | (64,1) | 64n+k |
| C | (128,256) | (256,1) | 256m+n |

Layout 是“坐标 → 元素偏移”的函数；Tensor 将它与一个存储引擎组合。在本例指针视图中，元素地址为 `base + layout(coord)`。算字节地址还要乘以 sizeof：A/B 乘 2，C 乘 4。`make_tensor` 创建视图不复制矩阵。

对应代码：

```cpp
auto mA = make_tensor(make_gmem_ptr(A), make_shape(Int<128>{}, Int<64>{}),
                      make_stride(Int<64>{}, _1{}));
auto mB = make_tensor(make_gmem_ptr(B), make_shape(Int<256>{}, Int<64>{}),
                      make_stride(Int<64>{}, _1{}));
auto mC = make_tensor(make_gmem_ptr(C), make_shape(Int<128>{}, Int<256>{}),
                      make_stride(Int<256>{}, _1{}));
```

CuTe 的无显式 stride 的紧凑 Layout 默认第一维连续，不能用这种默认值代替上面显式指定的行主序矩阵 stride。

## 2. Problem → CTA → warp → atom

选择 CTA tile `(BM,BN,BK)=(64,128,32)`：

```text
grid = (128/64, 256/128) = (2,2)  → 4 个 CTA
每个 CTA：128 threads = 4 warps
每个 CTA：C 子块 64×128，共 8192 个输出
每个 CTA：K 主循环 64/32 = 2 次
```

CTA 坐标 `(bm,bn)`：

| (bm,bn) | 全局 M 范围 | 全局 N 范围 |
|---|---|---|
| (0,0) | 0..63 | 0..127 |
| (1,0) | 64..127 | 0..127 |
| (0,1) | 0..63 | 128..255 |
| (1,1) | 64..127 | 128..255 |

每个 CTA 自己完成全部 K 归约，因此不需要 CTA 间归约，也不需要 atomic。

```cpp
auto tile = make_shape(_64{}, _128{}, _32{});
auto coord = make_coord(blockIdx.x, blockIdx.y, _);
auto gA = local_tile(mA, tile, coord, Step<_1, X, _1>{});
auto gB = local_tile(mB, tile, coord, Step<X, _1, _1>{});
auto gC = local_tile(mC, tile, coord, Step<_1, _1, X>{});
```

`Step` 选择 MNK 中的维度，`X` 跳过不属于该操作数的维度，`_` 保留 K tile 索引。因此 gA/gB/gC 的 shape 分别为 `(64,32,2)`、`(128,32,2)`、`(64,128)`。对于 CTA `(bm,bn)`、K tile `kt`：

```text
gA(i,p,kt) → A(64bm+i, 32kt+p)
gB(j,p,kt) → B_math(32kt+p, 128bn+j)
gC(i,j)   → C(64bm+i, 128bn+j)
```

选 Tensor Core atom `SM80_16x8x16_F32F16F16F32_TN`，每个 warp 协作执行一次 `16×8×16` 运算，不能把它理解为每个线程独立计算 16×8 个输出。其寄存器输入按每 lane A=8 个 half、B=4 个 half、C=4 个 float 分布。

```cpp
auto mma = make_tiled_mma(
    SM80_16x8x16_F32F16F16F32_TN{},
    Layout<Shape<_2, _2, _1>>{});
```

这里 `Layout<Shape<_2,_2,_1>>` 描述 atom 实例在 M/N/K 的排列。它第一维连续，因此 warp ID `w=wm+2wn`；基础覆盖为 `(16×2,8×2,16×1)=(32,16,16)`，线程总数 `32×2×2=128`。CTA 进一步重复基础覆盖 `(64/32,128/16,32/16)=(2,8,2)`。

由此得到线程 partition 的 shape（首维可能是嵌套 shape）：

| 分片 | CuTe shape | 每线程元素数 |
|---|---|---|
| partition_A | ((2,2,2),2,2) | 8×2×2=32 half |
| partition_B | ((2,2),8,2) | 4×8×2=64 half |
| partition_C | ((2,2),2,8) | 4×2×8=64 float |

每 CTA 的输出计数 `128×64=8192`。A/B 在不同 warp 间有复用：A 被两个 N warp 使用，B 被两个 M warp 使用，所以不能要求 A/B 的线程分片覆盖次数为 1。

一次 K tile，每 warp 执行 `2×8×2=32` 次 MMA；整个问题执行 `4 CTA×4 warp×2 K tile×32=1024` 次 warp MMA。一次 atom 计 `2×16×8×16=4096 FLOPs`，总计 `4,194,304=2×128×256×64 FLOPs`。这是逻辑指令计数，不是测得的吞吐率。

## 3. 从 atom 的 TV Layout 手推 lane 坐标

先拆线程号：

```text
t ∈ [0,127]
lane = t % 32
w    = t / 32
wm   = w % 2
wn   = w / 2
x    = lane % 4       // 0..3
y    = lane / 4       // 0..7
```

CuTe 源码 `include/cute/atom/mma_traits_sm80.hpp` 定义该 atom 的 TV Layout。以 C 为例：

```text
shape  = ((4,8),(2,2))
stride = ((32,1),(16,8))
```

thread 坐标 `(x,y)`、value 坐标 `(v0,v1)` 产生逻辑线性偏移 `32x+y+16v0+8v1`。这里 atom 的逻辑 `(M,N)` 线性化采用 M 连续，即 `offset=m+16n`，因此：

```text
m_atom = y + 8v1
n_atom = 2x + v0
```

该逻辑线性化用于解释 atom 的坐标分布，与实际 C 的行主序存储是两回事。最后访问物理地址时，仍使用 C 的 stride `(256,1)`。

A 的 TV stride 为 `((32,1),(16,8,128))`，逻辑线性化 `offset=m+16k`；B 的 TV stride 为 `((16,1),(8,64))`，逻辑线性化 `offset=n+8k`。同理可得：

```text
A: m_atom=y+8v1, k_atom=2x+v0+8v2
B: n_atom=y,     k_atom=2x+v0+8v1
```

注意：TV Layout 的 v 表示“一个 lane 持有的 value 坐标”。它不是源矩阵连续内存上的简单索引。

## 4. 加上 warp 与 tile 重复，得到完整地址公式

以下坐标均相对于 CTA 起点，K 相对于当前 K tile。`mi∈[0,1]`、`ni∈[0,7]`、`ki∈[0,1]`。CuTe 的嵌套 value 索引第一维变化最快。

**C 分片**：`v=v0+2v1 ∈ [0,3]`。

```text
i = y + 8v1 + 16wm + 32mi
j = 2x + v0 + 8wn + 16ni
global_m = 64bm+i
global_n = 128bn+j
C element offset = global_m*256 + global_n
```

**A 分片**：`v=v0+2v1+4v2 ∈ [0,7]`。

```text
i = y + 8v1 + 16wm + 32mi
p = 2x + v0 + 8v2 + 16ki
global_m = 64bm+i
global_k = 32kt+p
A element offset = global_m*64 + global_k
```

**B 分片**：`v=v0+2v1 ∈ [0,3]`。

```text
j = y + 8wn + 16ni
p = 2x + v0 + 8v1 + 16ki
global_n = 128bn+j
global_k = 32kt+p
B element offset = global_n*64 + global_k
```

固定 `(wm,wn)` 后，该 warp 的输出 M 集合为 `16wm+{0..15,32..47}`，N 集合为 `8wn+{0..7,16..23,...,112..119}`，共 `32×64` 个输出。它们分布在间隔子块中。若希望每 warp 负责连续矩形，需要另选 permutation/tiling，不能直接套用本公式。

## 5. 一个具体地址从头算到尾

取 CTA `(bm,bn)=(1,1)`、`kt=1`，考察乘加项 `A(67,37)×B_math(37,141)` 对 `C(67,141)` 的贡献。

```text
CTA 内：i=3，j=13，p=5
A offset = 67*64+37  = 4325 half  → 距 A 起点 8650 bytes
B offset = 141*64+37 = 9061 half  → 距 B 起点 18122 bytes
C offset = 67*256+141=17293 float → 距 C 起点 69172 bytes
```

C 由 thread 78 持有：`lane=14,x=2,y=3,wm=0,wn=1`。代入 `v=1→(v0,v1)=(1,0),mi=0,ni=0`，得到 `i=3,j=13`。

A(67,37) 在同一 warp 的 thread 78 中为 `v=1,mi=0,ki=0`；B_math(37,141) 则在该 warp 的 thread 86 中为 `v=1,ni=0,ki=0`，因为 thread 86 的 `y=5`。MMA 是 warp 协作，不能要求某个 C 元素所需的所有 A/B 数据都在持有它的同一个 lane 内。

本例共享内存也让 K 连续：

```text
sA(i,p) offset = i*32+p
sB(j,p) offset = j*32+p
```

所以这两个输入的共享内存偏移分别为 `3*32+5=101 half` 和 `13*32+5=421 half`，即 202 bytes、842 bytes。

合作加载采用 `q=t+128r`：A 由 thread 101 的 r=0 搬入，B 由 thread 37 的 r=3 搬入。**搬运线程分配与 MMA 分片分配是不同映射**，共享内存和同步把两者连接起来。

## 6. 最小 kernel 的数据流

```text
Global Tensor
  → local_tile 选出 CTA 的 A/B/C 视图
  → 128 threads 合作搬 A/B 到 shared memory
  → __syncthreads
  → thr.partition_A / partition_B 确定 MMA 要读取的元素
  → copy 到 make_fragment_A / make_fragment_B 创建的寄存器片段
  → gemm(mma,rA,rB,acc)
  → __syncthreads，才允许下一轮覆盖 shared memory
  → 重复 2 个 K tile
  → copy(acc,thr.partition_C(gC)) 写回
```

Shared memory 使用量：`(64×32+128×32)×2=12288 bytes=12 KiB`。每线程 64 个 FP32 累加元素只是 accumulator 需求，不是编译器分配的总寄存器数。代码显式保留整个 K tile 的 A/B 寄存器片段，优先保证容易阅读。

本例固定整除尺寸，没有边界谓词；不包含 swizzle、cp.async、ldmatrix 专用 TiledCopy、双缓冲或流水线优化。不同尺寸不能只改 M/N/K 后假定仍然正确。没有做性能基准；此实现用于验证坐标与真实计算。

## 7. 编译和运行

本机 Windows：在此目录执行 `build_local.cmd`，随后执行 `cute_minimal.exe`。该脚本固定了本机工具链及 CUTLASS 路径；换机器应修改路径。

Linux 或已初始化 CUDA/C++ 环境：

```sh
nvcc -std=c++17 --expt-relaxed-constexpr -arch=sm_80 \
  -I /path/to/cutlass/include gemm_128x256x64.cu -o cute_minimal
./cute_minimal
./cute_minimal --layout-only
```

编译目标应选择设备支持且支持本 atom 的架构；本机脚本使用 sm_89。`--layout-only` 只运行主机上的 Layout/partition 检查，不启动 GPU；编译仍需要 CUDA 工具链。

提供的 CMake 配置也可用于其他环境：

```sh
cmake -S . -B build -DCUTLASS_DIR=/path/to/cutlass -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

若使用 Windows，需要安装 CMake 的生成器所需 C++/CUDA 集成，或在开发者命令行中选择适当生成器。本次实际验证使用 `build_local.cmd`，未验证 CMake 构建路径。

本机运行结果：

```text
A layout: (_128,_64):(_64,_1)
B layout: (_256,_64):(_64,_1)
C layout: (_128,_256):(_256,_1)
partition_A: ((_2,_2,_2),_2,_2):((_1@1,_8@0,_8@1),_32@0,_16@1)
partition_B: ((_2,_2),_8,_2):((_1@1,_8@1),_16@0,_16@1)
partition_C: ((_2,_2),_2,_8):((_1@1,_8@0),_32@0,_16@1)
PASS: all A/B/C thread-coordinate formulas and coverage
PASS: all 32768 outputs; max_abs_error=0
```

identity Tensor 中的 `@0`、`@1` 分别表示坐标分量 0、1 的步进，不能当作实际内存 stride。程序对所有线程逐项比较手推坐标与 CuTe `partition_*` 的结果，确认 A/B 每元素各覆盖两次、C 每元素覆盖一次。GPU 结果逐元素与 CPU double reference 比较，允许绝对误差 1e-5。当前输入为小的 1/8 倍数，恰好可精确表示，因此本次误差为零；不代表任意 FP16 输入都应得到零误差。

编译通过，CUTLASS 的 `layout.hpp` 出现未使用局部变量 warning #550-D；不影响上述验证。

## 8. 与项目 SM100 示例的关系

项目已有的 SM100 指南选用 CTA tile `64×128×64`。将同一个完整问题 `128×256×64` 按那个 CTA tile 划分，仍是 4 个逻辑输出 tile，但 K 主循环只有一次；persistent scheduler、cluster 调度还需另行分析，不能把逻辑 tile 数直接等同于实际 grid。

本例采用 `64×128×32`、SM80 warp 级 mma.sync，因此能够在本机验证。SM100 UMMA 使用 tcgen05、SMEM descriptor、TMEM 和不同的参与/同步规则，本文的 lane 寄存器坐标公式不适用于它。可迁移的基础是：显式 Layout、Tensor 视图、local_tile 的坐标，以及“先理解 atom，再理解 TiledMma”的推导顺序。

官方对照材料：

- [NVIDIA CuTe GEMM tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md)：CTA 与线程分块的接口说明。
- [NVIDIA SM80 tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)：更完整的 Tensor Core 数据搬运与计算示例。
- [SM80 MMA traits（固定到本次源码提交）](https://github.com/NVIDIA/cutlass/blob/ffa119a1255d78998536107466cc7097ecefa393/include/cute/atom/mma_traits_sm80.hpp)：本文手推 TV Layout 的定义来源。
