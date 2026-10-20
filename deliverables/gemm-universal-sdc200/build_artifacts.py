from pathlib import Path
import csv, hashlib, html, json, math, re, subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = Path('D:/000_SDC/001TASK/02_Research/cutlass')
SDC = Path('D:/lvz/sdc/cutlass')
rows = []
def row(phase, file, anchor, api, reuse, adapt, target, check, status='条件复用 / 必须适配后端'):
    rows.append(dict(阶段=phase, 文件=file, 锚点=anchor, API=api, 可复用部分=reuse, 适配或核验内容=adapt, 建议SDC落点=target, 验收=check, 分类=status))

row('01 构建','CMakeLists.txt','project\(', 'CMake project / CUDA language', '目标分层、源码组织、CTest 思路', 'SDC 编译器识别、device 编译/链接、目标架构参数；不能沿用 sm_100a 标志', '新增 SDC toolchain / 独立构建选项', 'host/device 最小程序编译链接运行；确认实际后端')
row('02 能力宏','include/cute/config.hpp','define CUTE_HOST_DEVICE', 'CUTE_HOST_DEVICE / CUTE_DEVICE / invalid-control-path', '接口注解宏的调用位置', 'SDC host/device 标注、forceinline、RTC、诊断；未实现设备操作必须显式拒绝', '现配置增加明确 SDC 分支', 'host/device 双模式编译；禁用能力触发预期错误')
row('03 指令能力','include/cute/arch/config.hpp','SM100', 'CUTE_ARCH_* feature macros', '按能力开关分发的模式', '将 SDC 能力和 NVIDIA __CUDA_ARCH__ 分离；核对 fallback 是否会静默进入空路径', '新增 SDC feature 宏', '支持/不支持矩阵对应正确特化及诊断')
row('04 架构标签','include/cutlass/arch/arch.h','struct Sm100', 'arch::Sm100 / sm100_smem_capacity_bytes', 'arch tag 抽象和资源查询接口', '新增 SDC tag 与真实 SMEM/TMEM/warp 资源；禁止用假 compute capability 驱动 CUDA 分支', '建议新增 arch::Sdc200', '静态资源计算符合规格，超限配置拒绝')
row('05 架构聚合','include/cutlass/arch/mma_sm100.h','SM100','MMA support declarations / architecture includes', '聚合头和能力暴露方式', '清理 NVIDIA 编译器保护与目标指令依赖，接入 SDC 后端', '建议 mma_sdc200.h 或统一后端入口', 'include 自洽；不意外依赖 NVIDIA PTX')
row('06 问题与模式','include/cutlass/gemm/gemm.h','GemmUniversalMode','GemmUniversalMode / problem semantics', 'GEMM/strided batch 数学语义', 'mode 支持范围明确，指针数组模式不能当 strided batch 处理', '优先复用；能力检查限制模式', 'M/N/K/L、batch stride、mode 对应正确', '优先复用语义 / 审核边界')
row('07 类型元编程','include/cute/int_tuple.hpp','Shape','Shape / Stride / tuple utilities', '编译期整数与运行期 tuple 算法', '编译器模板兼容、整数宽度和溢出边界', '优先原接口复用', '静态 shape 断言与运行时坐标单测', '优先复用')
row('08 Layout','include/cute/layout.hpp','make_layout','Layout / make_layout / compose / partition', '坐标映射和分块代数', '设备注解/编译器兼容；硬件依赖布局仍由 Traits/Builder 重选', '复用通用层；隔离硬件 layout', '索引到地址映射与边界参考一致', '优先复用算法 / 核验编译器')
row('09 Tensor','include/cute/tensor_impl.hpp','make_tensor','Tensor / make_tensor / local_tile', 'tensor 视图与坐标组合', '地址空间、指针标记、TMEM engine 和设备注解', '通用视图复用；新增 SDC memory engine', '同一坐标映射正确地址空间，无错误 cast')
row('10 Swizzle','include/cute/swizzle.hpp','struct Swizzle','Swizzle / composed layout', '通用位变换算法', '合法 bank/swizzle 与 descriptor 编码由 SDC 重新定义', '算法复用 + SDC layout policy', 'golden 地址序列、bank 分布、编码符合规格')
row('11 主循环入口','include/cutlass/gemm/collective/collective_builder.hpp','CollectiveBuilder','gemm::collective::CollectiveBuilder', '调用者模板接口与 CollectiveOp 契约', 'include/register SDC 特化；不支持组合明确报错', '建议 builders/sdc200_umma_builder.inl', '目标元素/布局/alignment/shape 选中正确 Collective')
row('12 Mainloop builder','include/cutlass/gemm/collective/builders/sm100_umma_builder.inl','using CollectiveOp','TiledMma / GmemTiledCopyA/B / PipelineStages / DispatchPolicy', '根据配置组装 Collective 的流程', 'MMA 形状、cluster copy、SMEM layout、资源 carveout、stage、overlap 策略全部核验', '建议 sdc200_umma_builder.inl', '资源不超限；自动策略符合能力而非名称相似')
row('13 主循环 helper','include/cutlass/gemm/collective/builders/sm100_common.inl','sm100_make_trivial_tiled_mma','sm100_make_trivial_tiled_mma / cluster-to-TMA / layout selectors', '类型选择的组织方式', '选择 SDC Operation/Traits 和 copy atom；改变 tile/major 限制时重选布局', '建议 sdc200_common.inl', '类型静态断言 + 代表布局实际数据验证')
row('14 Kernel carveout','include/cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl','Carveout','SMEM carveout helpers', '预留同步元数据后计算可用缓冲的策略', '实际 barrier 大小、CLC/软件队列、对齐、SMEM 容量重新计算', 'SDC carveout helper', 'sizeof 实例总量与 launch SMEM 一致')
row('15 Mainloop policy','include/cutlass/gemm/dispatch_policy.hpp','struct MainloopSm100TmaUmmaWarpSpecialized','MainloopSm100TmaUmmaWarpSpecialized / KernelTmaWarpSpecializedSm100', 'Stages/ClusterShape/Schedule 的类型契约', '新增 SDC policy 与 kernel 匹配条件，分离 load/accumulator/scheduler stage', '建议 MainloopSdc200* / Kernel*Sdc200', '默认与显式 schedule 均分发到预期实现')
row('16 主循环主体','include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp','class CollectiveMma','Arguments / Params / load_init / load / mma / load_tail', '分区、K 循环、producer/consumer 接口', '替换 TMA、TMEM 视图、pipeline、cluster mask 与异步完成；必要时重构分工', '建议 collective/sdc200_mma_warpspecialized.hpp', '多 K tile、phase wrap、尾块、持续多 tile 无误')
row('17 主循环 host 转换','include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp','to_underlying_arguments\(','to_underlying_arguments / make_tma_atom_A/B_sm100', '从 ptr/stride/problem shape 构造 tensor 的流程', 'SDC host tensor descriptor encoder；fallback/cluster 描述符的支持范围', 'SDC Collective Params + descriptor factory', '描述符字节与硬件寻址对照；拒绝不合法 stride')
row('18 MMA 通用算法','include/cute/algorithm/gemm.hpp','gemm\(','cute::gemm', 'tile/Atom 调用展开和接口', '模板限制与 fragment/布局一致性；确认没有进入错误默认路径', '优先复用 + SDC Traits', '编译期形状检查，Atom 次序/累加与参考一致', '优先复用 / 核验实例化')
row('19 MMA Atom','include/cute/atom/mma_atom.hpp','return mma_unpack','MMA_Atom::call / TiledMMA / ADL mma_unpack', 'Operation→Traits→Atom→TiledMMA 框架', 'SDC fragment engine、hidden friend 解包与返回约定', '通用 Atom 复用；新增架构 Traits', '能解析到 SDC fma，且不存在错误 namespace/ADL')
row('20 MMA Traits','include/cute/atom/mma_traits_sm100.hpp','struct MMA_Traits<SM100_MMA_F16BF16_SS','MMA_Traits / Shape_MNK / FrgType / mma_unpack', 'Traits 描述形式和解包流程', 'SDC Atom K/M/N、lane/value layout、TMEM fragment、scale 与 descriptor 编码', '建议 atom/mma_traits_sdc200.hpp', '静态 fragment 映射 + 每元素数值对照')
row('21 TMEM fragment','include/cute/atom/mma_traits_sm100_frag.hpp','tmem_frg','UMMA::tmem_frg_* / TMEM fragment layouts', '表达 accumulator tensor 的抽象', 'TMEM 行列、lane/data-path、地址加法、overlap 范围', 'SDC fragment traits/engine', 'T2R 每 lane 映射和多 stage 互不覆盖')
row('22 MMA 指令','include/cute/arch/mma_sm100_umma.hpp','struct SM100_MMA_F16BF16_SS','SM100_MMA_F16BF16_SS::fma / tcgen05.mma', 'fma 接口形状，受控覆盖/累加语义', 'SDC builtin lowering、FP16/BF16 模式、发射粒度、完成事件、2SM 限制', '已见 mma_sdc200_umma.hpp；需验收', '单 Atom FP16/BF16 + overwrite/accumulate + 真正生成机器指令', '必须适配 / 已有包装未验证')
row('23 MMA descriptor','include/cute/arch/mma_sm100_desc.hpp','union SmemDescriptor','SmemDescriptor / InstrDescriptor / smem_desc / make_instr_desc', '有类型的编码器接口', '地址单位、bitfield ABI、stride/swizzle、idesc 高低位、合法值', '已见 mma_sdc200_desc.hpp；逐字段核验', 'golden bit-pattern、反解与硬件读数一致', '必须确认 ABI / 不可按名称复用')
row('24 TMEM 分配','include/cute/arch/tmem_allocator_sm100.hpp','class Allocator1Sm','TMEM::Allocator1Sm / allocate / free / release_allocation_lock', 'allocator 接口和生命周期管理方式', 'SDC 分配/释放 builtin、粒度/容量、参与 lane、返回位置与许可语义', '建议 tmem_allocator_sdc200.hpp', '分配→写→读→drain→释放；重复运行无泄漏/死锁')
row('25 TMEM helper','include/cutlass/detail/sm100_tmem_helper.hpp','tmem','TMEM partition/accumulator helpers', '将 tile 切成 accumulator stage 的思路', '硬件 TMEM 行列容量与别名、overlap、寻址布局', 'SDC TMEM helper', '所有 stage/subtile 映射合法且不发生意外别名')
row('26 Copy Atom','include/cute/atom/copy_atom.hpp','struct Copy_Atom','Copy_Atom / TiledCopy / copy_unpack', 'copy 分区和调用框架', '新增 SDC copy Traits，核验线程参与/向量宽度约定', '通用 Atom + SDC traits', '每 lane 源/目的覆盖正确，无缺失或重复', '优先复用框架 / 新增后端')
row('27 TMEM copy Traits','include/cute/atom/copy_traits_sm100.hpp','make_tmem_copy','Copy_Traits / make_tmem_copy / T2R', '表达线程到 TMEM 的 copy 接口', 'lane/register layout、load 粒度、合法 shape、完成语义', '建议 copy_traits_sdc200.hpp', '非平凡 TMEM pattern 完整读回')
row('28 TMEM copy ISA','include/cute/arch/copy_sm100.hpp','tcgen05.ld','SM100_TMEM_LOAD* / tcgen05.ld / copy operations', 'copy Operation 的参数边界', 'SDC TMEM load/store builtin、wait、寄存器布局', '建议 copy_sdc200.hpp', '不同 lane/subpartition 读回及 fence 压测', '必须适配')
row('29 TMA Traits SM100','include/cute/atom/copy_traits_sm100_tma.hpp','make_tma_atom_A_sm100','make_tma_atom_A/B_sm100 / multicast partition', '按 A/B tensor 和 cluster 分区的算法意图', 'SDC descriptor、multicast mask、CTA group、tensor shape 限制', '建议 copy_traits_sdc200_tma.hpp', 'A 广播/B 分片/边界 tensor 的数据一致')
row('30 TMA host 编码','include/cute/atom/copy_traits_sm90_tma.hpp','cuTensorMapEncodeTiled','make_tma_copy / cuTensorMapEncodeTiled', 'tensor→copy 参数的接口意图', 'NVIDIA Driver TensorMap ABI 和 driver wrapper 需替换或有证据兼容', 'SDC tensor-map encoder / runtime shim', '跨 shape/stride/swizzle 编码和实际传输正确', '必须适配或证明兼容')
row('31 TMA descriptor 类型','include/cute/arch/copy_sm90_desc.hpp','TmaDescriptor','TmaDescriptor / swizzle enums / descriptor helpers', '描述符抽象与工厂接口', '结构大小、对齐、enum 值、host/device 可见性', '建议 SDC TMA descriptor', 'sizeof/alignof + 编码 golden + kernel 参数 ABI')
row('32 TMA device load/store','include/cute/arch/copy_sm90_tma.hpp','cp.async.bulk','SM90_TMA_LOAD/STORE/MULTICAST / cp.async.bulk', '全局↔共享 tensor copy 意图', 'SDC DMA/bulk copy、OOB 行为、barrier transaction bytes、commit/wait', 'SDC TMA Operations；无 TMA 时换协作 copy', '异步完成、越界 fill/store 边界与事务计数', '必须适配')
row('33 TMA SM100 扩展','include/cute/arch/copy_sm100_tma.hpp','SM100_TMA','SM100 TMA 1SM/2SM Operations', '按能力选择 Operation 的形式', 'CTA-group 和 multicast 新模式；只适配实际支持分支', 'SDC copy Operation 能力表', '支持模式运行；不支持模式编译/运行明确拒绝')
row('34 pipeline','include/cutlass/pipeline/sm100_pipeline.hpp','class PipelineUmmaAsync','PipelineTmaUmmaAsync / PipelineUmmaAsync / PipelineCLCFetchAsync', 'acquire/commit/wait/release、index/phase 契约', 'UMMA completion、TMA expected-bytes、cluster scope、barrier 存储与 arrival counts', '建议 sdc200_pipeline.hpp', 'load/MMA/epilogue 高并发和 phase wrap 压测', '必须适配同步语义')
row('35 通用 pipeline','include/cutlass/pipeline/sm90_pipeline.hpp','class PipelineAsync','PipelineAsync / PipelineTmaAsync / PipelineTmaStore', '环形 state 与空满管理', '内部仍依赖 barrier/TMA 原语；核对每种 commit/tail', '优先接口复用；替换底层同步', '生产/消费多阶段和 tail 不早退')
row('36 barrier/fence','include/cutlass/arch/barrier.h','void umma_arrive','ClusterBarrier / ClusterTransactionBarrier / umma_arrive / fence_view_*', '同步接口与角色表达', 'mbarrier phase、字节计数、UMMA arrival、TMEM load fence、SMEM async proxy 可见性', 'SDC barrier/fence 后端', '跨操作可见性 litmus、参与者计数、scope 与进展保证', '必须适配')
row('37 named/order barrier','include/cutlass/barrier.h','OrderedSequenceBarrier','OrderedSequenceBarrier / barrier helpers', '顺序化 load 与角色协议', '内部 named barrier 与线程集合/warp 大小', 'SDC barrier traits', '可选 C load、load order、退出路径无死锁')
row('38 cluster/election','include/cute/arch/cluster_sm90.hpp','elect_one_sync','cluster_arrive/wait / block_rank_in_cluster / mapa / elect_one_sync', '统一 election/cluster API', 'warp 大小、active mask、cluster shared 寻址和内存屏障；替换内联 PTX', '已见 cluster_sdc200.hpp，但仍含 NVIDIA PTX', '部分 active mask、cluster rank、跨 CTA 可见性；include 不重定义', '必须核验 / 现文件非完成证明')
row('39 cluster SM100','include/cute/arch/cluster_sm100.hpp','cluster','SM100 cluster helpers', '能力隔离入口', '核对相关扩展指令与能力宏是否由 SDC 编译器支持', 'SDC cluster extension 或显式禁用', '仅启用已验收的 cluster 操作')
row('40 register 调度','include/cutlass/arch/reg_reconfig.h','warpgroup_reg','warpgroup_reg_alloc / warpgroup_reg_dealloc', '按角色调整资源的策略意图', 'SDC 是否存在等价动态寄存器重分配；无则重算静态资源/占用率', 'SDC resource policy / 合法无操作分支', '编译资源报告、角色资源足够、性能基线')
row('41 epilogue Builder','include/cutlass/epilogue/collective/collective_builder.hpp','CollectiveBuilder','epilogue::collective::CollectiveBuilder', '输出/fusion/布局接口及 CollectiveOp 契约', '接入 SDC 特化，确保 accumulator 和 Mainloop 匹配', '建议 builders/sdc200_builder.inl', 'C=void、有 C、各 D layout 选择正确实现')
row('42 epilogue 策略生成','include/cutlass/epilogue/collective/builders/sm100_builder.inl','sm100_get_tmem_load_op','EpilogueTileAuto / TmaWarpSpecialized1Sm / sm100_get_tmem_load_op', '按类型/tile 选择输出流程的结构', 'TMEM T2R、subtile、SMEM swizzle、vector width、reuse、delay store 和资源', '建议 sdc200_builder.inl', 'subtile 覆盖、C/D alias 和数据转换正确')
row('43 epilogue dispatch','include/cutlass/epilogue/dispatch_policy.hpp','struct Sm100TmaWarpSpecialized','Sm100TmaWarpSpecialized / TmaWarpSpecialized1Sm', 'policy 提供 stage 与 schedule 常量', '新增 SDC load/store stage 与访问策略', '建议 Sdc200* policy', 'dispatch 与 kernel/Collective 相容')
row('44 epilogue 主体','include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp','class CollectiveEpilogue','load / store / store_tail / FusionCallbacks / CopyOpT2R', '融合生命周期、分片输出流程', 'TMEM ownership、T2R、C DMA、R2S、TMA store/fence/drain；无 TMEM 时改数据通路', '建议 sdc200_epilogue_*', 'alpha/beta、C=void、残块、batch、输出类型与 reference 一致')
row('45 fusion 架构回调','include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp','FusionCallbacks','FusionCallbacks / sm100 visitor adapters', 'visit 生命周期和融合树组合', '架构专用 broadcast/load/store、TMEM/SMEM 访问、归约和同步', 'SDC callbacks；纯算术节点尽量复用', 'bias/activation/辅助输出独立覆盖后组合')
row('46 fusion 通用算术','include/cutlass/epilogue/fusion/operations.hpp','LinearCombination','FusionOperation / LinearCombination', 'alpha*acc+beta*C 等数学定义', '数值精度、舍入、NaN/Inf、设备内建算术兼容', '优先复用数学层', '与明确参考精度/容差一致', '优先复用 / 核验数值')
row('47 数值转换','include/cutlass/numeric_conversion.h','NumericArrayConverter','NumericConverter / NumericArrayConverter', '转换 API 和纯 C++ fallback', '内联 PTX、半精度/低位宽 pack、round/saturate 的目标实现', 'SDC numeric builtin specialization', '边界值、舍入、饱和和向量 pack 位模式')
row('48 Kernel 声明','include/cutlass/gemm/kernel/gemm_universal_decl.h','class GemmUniversal','GemmUniversal<ProblemShape,Mainloop,Epilogue,Scheduler>', '用户模板签名与 composition 概念', '声明本身少改；通过 SDC dispatch 接入特化', '保留入口，新增 SDC kernel 特化', '同样三参数组合可实例化 SDC kernel', '优先复用公共接口')
row('49 Kernel 聚合','include/cutlass/gemm/kernel/gemm_universal.hpp','sm100_gemm_tma_warpspecialized','architecture kernel includes', '架构实现聚合方式', '安全 include SDC 特化，避免 namespace 嵌套和 ODR 冲突', 'include sdc200_gemm_*', '同时包含 NVIDIA/SDC 后端的编译隔离验证')
row('50 Kernel 编排','include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp','class GemmUniversal','Arguments / Params / SharedStorage / operator() / TMEM handoff', '参数聚合、tile 协作与所有权交接设计', 'warp/thread/cluster 分工、SMEM/TMEM 容量、pipeline、scheduler、退出/释放重新核对', '建议 sdc200_gemm_*；协议不同时重构', '多 tile、多 stage、tail/异常/重复启动数值与进展', '必须适配或完整证明硬件契约兼容')
row('51 scheduler selector','include/cutlass/gemm/kernel/tile_scheduler.hpp','Default \(void\) for Sm100','TileSchedulerSelector / default void mapping', 'tag→scheduler 的接口', '新增 SDC 默认映射；不让 arch::Sdc200 误走 Sm100 或缺失特化', 'SDC static/software/CLC scheduler selector', '默认/显式 scheduler 都可构建且行为正确')
row('52 persistent scheduler','include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp','issue_clc_query','PersistentTileSchedulerSm100 / fetch_next_work / CLC', 'WorkTileInfo、raster/swizzle 和坐标转换思想', 'clusterlaunchcontrol.try_cancel/query_cancel 的 SDC 等价或软件替代；修复 workspace/参与角色', '建议 SDC scheduler；先静态后动态', 'tile 覆盖无遗漏/重复，batch 与退出正确', '必须适配硬件 CLC / 可换算法')
row('53 device adapter','include/cutlass/gemm/device/gemm_universal_adapter.h','class GemmUniversalAdapter<','can_implement / get_workspace_size / initialize / update / run', '公共方法和有状态 Params handle', '架构数值分支、cudaFuncSetAttribute、occupancy、launch/error/stream；CudaHostAdapter 只是扩展点', '公共接口 + SDC runtime/launcher 后端', 'initialize/update/run 生命周期、错误传播和实际完成')
row('54 全局入口','include/cutlass/device_kernel.h','void device_kernel','device_kernel<Operator> / Params / dynamic shared memory', 'wrapper 调 functor 的结构', 'SDC kernel ABI、launch_bounds、grid_constant、动态共享内存对齐', 'SDC device entry wrapper', '参数 layout 与动态 SMEM 正确传入')
row('55 普通 launcher','include/cutlass/kernel_launch.h','kernel_launch','kernel_launch / runtime launch APIs', '从 grid/block/SMEM/stream 提交 kernel 的接口', 'SDC runtime launch/参数指针/异步错误与 PDL 支持边界', 'SDC launcher 或有证据兼容 runtime shim', '单 CTA 与多 block 启动、属性/参数及错误正确')
row('56 cluster launcher','include/cutlass/cluster_launch.hpp','cudaLaunchKernelExC','ClusterLauncher::launch_with_fallback_cluster / cudaLaunchKernelExC', 'cluster/fallback 选择接口', 'SDC cluster launch ABI、上限、grid 整除、preferred/fallback 和驱动支持', 'SDC cluster launcher；不支持则禁用配置', '1×4×1 可实际启动，fallback/非法配置正确处理', '必须适配或证明 runtime 兼容')
row('57 硬件信息','include/cutlass/kernel_hardware_info.h','struct KernelHardwareInfo','KernelHardwareInfo / query_device_multiprocessor_count', '信息结构和调用位置', '设备 ID、SM/计算单元、cluster 范围与 runtime 查询；单位不盲目对应 SM', 'SDC hardware info provider', '查询值与真实资源及 scheduler 解释一致')
row('58 host/device 内存','tools/util/include/cutlass/util/device_memory.h','cudaMalloc','device_memory::allocation / memcpy / synchronization', 'RAII、元素容量管理', 'SDC malloc/free/copy/stream/error 与地址空间 ABI', 'SDC runtime shim / utility backend', '分配拷贝回读、零长度/错误路径与生命周期')
row('59 HostTensor','tools/util/include/cutlass/util/host_tensor.h','sync_device','HostTensor / sync_host / sync_device', 'host 数据与布局组织、随机输入工具', '底层 device allocation/copy API、格式表示', '保留测试结构，接 SDC 内存工具', 'A/B/D 数据传输与 batch stride 正确')
row('60 Testbed','test/unit/gemm/device/gemm_testbed_3x.hpp','bool  \(','TestAll / Testbed3x::run / TestbedImpl::verify / sufficient', '尺寸枚举、host reference 对照、GoogleTest 断言', '修复当前缺失函数名；替换 CUDA API；将 skip 与通过分开；initialize 失败立即终止', 'SDC testbed backend 或受控公共抽象', '记录候选/skip/launch/数值数，强制配置不得静默跳过')
row('61 具体测试','test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu','64x128x64_1x4x1_1sm','TEST / Builders / GemmUniversal / EXPECT_TRUE', '测试表达形式和计算语义', 'SDC arch/schedule/能力 guard/测试 target；多布局建立多个实例', '建议 sdc200_tensorop_gemm/*', '每个承诺布局、tail、batch、K 长度均实际执行')
row('62 主机参考','tools/util/include/cutlass/util/reference/host/gett.hpp','void Gemm3x','host::Gemm3x / Gett', 'CPU 参考 GEMM 和 tensor 坐标语义', '元素格式与 host 编译兼容；明确累加/舍入容差，防止跟随错误设备布局', '优先复用独立 host reference', '小矩阵手算/独立参考 + SDC 比较', '优先复用 / 保持参考独立')
row('63 数值比较','tools/util/include/cutlass/util/tensor_compare.h','TensorEquals','TensorEquals / relative comparison helpers', '相等/容差比较方法', '调用方明确 tolerance、NaN/Inf、零附近误差及精度，禁止放宽掩盖错误', '优先复用比较工具', '故意错误数据能被检测，容差符合精度目标', '优先复用 / 配置验收')
row('64 构建测试 target','test/unit/gemm/device/sm100_tensorop_gemm/CMakeLists.txt','cutlass_test_unit','SM100 unit-test target registration', '按架构组织目标和 GTest filter 的方式', 'SDC toolchain、链接库、硬件 capability guard、测试运行器', '建议 SDC 单独 target', 'CI/本机运行到真实 SDC kernel，非空测试或全 skip')
row('65 扩展范围','include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp','StreamK','Stream-K / split-K reduction interfaces', '分解工作和归并的数学需求', 'workspace、全局原子/内存序、fixup barrier、精度；本例默认路径不启用', '后续 SDC StreamK backend', '独立验收后才能加入支持清单', '按需扩展 / 非本例必需')

row('66 CUTLASS 能力宏','include/cutlass/arch/config.h','CUTLASS_ARCH_MMA_SM100_SUPPORTED','CUTLASS_ARCH_MMA_*_SUPPORTED / *_ENABLED', '编译器/目标能力分离的结构', 'SDC 编译器识别、目标 feature、host/device 两阶段宏，不让 host supported 误表示设备指令可用', '明确 SDC 宏并与 cute/arch/config.hpp 联动', 'host 预处理、device 预处理和实际机器指令三者一致')
row('67 HostAdapter / Driver 接入','include/cutlass/cuda_host_adapter.hpp','cuTensorMapEncodeTiled','CudaHostAdapter / CUTLASS_CUDA_DRIVER_WRAPPER_CALL', '可注入 launch/query 的接口思想', 'Driver 动态解析、TensorMap 编码、cluster launch、occupancy；不能覆盖 device PTX 的语义', 'SDC host adapter / driver wrapper backend', '所有被调用方法有实现，descriptor 与 kernel runtime ABI 相容')
row('68 编译器 builtin 声明','D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h','void __sdc_uten_mma_bf16\\(','__sdc_uten_mma_bf16 / compiler lowering', '现有 builtin 声明形参契约', '核验前端识别、后端 lowering、FP16/BF16 idesc、inputD 和完成语义；声明存在不证明生成指令正确', '现有 SDC compiler header / 编译器后端', '编译诊断 + 机器码 + 单 Atom overwrite/accumulate 数值', '外部编译器依赖 / 必须验收')

fixes = {
    '05': {'锚点':'struct Mma<', 'API':'arch::Mma SIMD specialization / architecture includes', '阶段':'05 架构 SIMD 配套（按需）'},
    '06': {'文件':'include/cutlass/gemm/gemm_enumerated_types.h'},
    '11': {'锚点':'collective_builder_decl.hpp'},
    '16': {'锚点':'struct CollectiveMma<'},
    '37': {'文件':'include/cutlass/pipeline/sm90_pipeline.hpp'},
    '63': {'文件':'tools/util/include/cutlass/util/reference/host/tensor_compare.h'}
}
for rr in rows:
    rr.update(fixes.get(rr['阶段'][:2],{}))
evidence = {}
anchor_errors = []
for rr in rows:
    p = ROOT / rr['文件']
    if not p.is_file():
        anchor_errors.append(f'Missing source: {p}')
        continue
    data = p.read_bytes()
    text = data.decode('utf-8-sig')
    lines = text.splitlines()
    hits = [i+1 for i, line in enumerate(lines) if re.search(rr['锚点'], line)]
    if not hits:
        anchor_errors.append(f'{rr["文件"]} :: {rr["锚点"]}')
        continue
    rr['行号'] = hits[0]
    evidence[rr['文件']] = {'path':str(p), 'sha256':hashlib.sha256(data).hexdigest(), 'lines':len(lines)}
if anchor_errors:
    raise RuntimeError('Missing anchors: ' + '; '.join(anchor_errors))

def slink(rr):
    p = (ROOT / rr['文件']).as_posix()
    return f'[{rr["文件"]}:{rr["行号"]}]({p}:{rr["行号"]})'

parts = ['# SDC200 完整流程适配参考表', '',
    '基线：dense FP16 A/B → FP32 accumulator/D，1SM，含 Mainloop、Epilogue、Kernel、Adapter 与测试链。以下 68 项为源码入口级清单；SDC 候选文件标为“建议”的均是设计建议。分类含“复用”不意味着整个文件无硬件依赖。', '',
    '相对路径的根目录为 `D:/000_SDC/001TASK/02_Research/cutlass`；第 68 项显式列出外部 SDC compiler header 的绝对路径。链接与行号来自本轮工作树。实际 SDC 包装位于另一棵 `D:/lvz/sdc/cutlass`，两者版本不同，不能混作同一源码快照。', '',
    '优先读 11–17（Mainloop）、41–50（Epilogue/Kernel）、51–57（调度与启动）、60–64（验收）；底层必须补齐 19–40。本文给出架构条件下的适配建议，没有声称 SDC 硬件已支持这些能力。', '',
    '整条链至少有四种独立 ABI：kernel 参数、TMA TensorMap、UMMA SMEM/指令 descriptor、TMEM 地址/fragment 布局。MMA builtin 可编译不能证明其余 ABI 兼容。', '',
    '| 阶段 / 分类 | 实际文件与源码入口 | API | 可复用部分 | 需要适配或核验 | SDC 落点 | 验收条件 |',
    '|---|---|---|---|---|---|---|']
for rr in rows:
    parts.append('| ' + ' | '.join(str(v).replace('|',' / ').replace('\n',' ') for v in [rr['阶段']+'；'+rr['分类'],slink(rr),rr['API'],rr['可复用部分'],rr['适配或核验内容'],rr['建议SDC落点'],rr['验收']])+' |')
parts += ['', '## 现有 SDC200 证据', '',
    '- [MMA 包装](D:/lvz/sdc/cutlass/include/cute/arch/mma_sdc200_umma.hpp:108)：存在 `SDC200_MMA_F16BF16_SS::fma`，调用 `__sdc_uten_mma_bf16`。',
    '- [builtin 声明](D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h:897)：形参是 `dTmem/aDesc/bDesc/iDesc/inputD`。声明对应不构成数值或完成语义验证。',
    '- [SDC descriptor](D:/lvz/sdc/cutlass/include/cute/arch/mma_sdc200_desc.hpp:109)：存在 ABI 敏感 union，仍需位级对照。',
    '- [SDC cluster/election](D:/lvz/sdc/cutlass/include/cute/arch/cluster_sdc200.hpp:180)：仍有 NVIDIA `elect.sync` 与 cluster PTX；本轮无法据此确认原生 SDC 支持。', '',
    '## 能力扩展，按实际需求另立验收项', '',
    '2SM MMA、sparse、FP8/FP6/FP4、block-scaled、mixed-input、pointer-array/grouped GEMM、复杂 fusion、PDL、RTC、graph capture、distributed GEMM 和 profiler dispatch 均不因本例通过而自动获得支持。扩展时从其独立 Builder/Collective/Kernel/Traits/Operation/测试入口继续审计；只实现本例时应明确拒绝不支持能力。', '',
    '## 测试通过口径', '',
    '当前 Testbed 对 `can_implement` 失败返回 true。SDC 验收应单独列出 requested、unsupported/skip、initialized、launched、synchronized、numerically_passed；承诺支持的配置跳过视为未完成。编译、运行、数值、机器码与性能是五种不同证据。本轮只有静态源码和文档图稿验证。', '',
    '[返回完整说明](report.md) · [下载 UTF-8 CSV](sdc200-adaptation.csv)', '']
(OUT/'sdc200-adaptation.md').write_text('\n'.join(parts),encoding='utf-8')
with (OUT/'sdc200-adaptation.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['阶段','分类','文件','行号','API','可复用部分','适配或核验内容','建议SDC落点','验收']
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

# One scene model is rendered to both PNG and native editable mxGraph nodes.
FONT='C:/Windows/Fonts/msyh.ttc'; BOLD='C:/Windows/Fonts/msyhbd.ttc'
COLORS={'blue':('#eaf2ff','#3b72b7'),'green':('#e7f5ef','#26806b'),'orange':('#fff2df','#c58428'),'purple':('#f0eafa','#8060af'),'red':('#fceaea','#b94c50'),'gray':('#edf1f5','#64758b')}
class Scene:
    def __init__(self,name,title,subtitle,w,h):
        self.name=name;self.w=w;self.h=h;self.nodes=[];self.edges=[]
        self.box('title',30,18,w-60,65,title,'gray',30,False)
        self.box('subtitle',30,85,w-60,58,subtitle,'gray',18,False)
    def box(self,id,x,y,w,h,text,color='blue',size=20,border=True):
        self.nodes.append(dict(id=id,x=x,y=y,w=w,h=h,text=text,color=color,size=size,border=border));return id
    def arrow(self,a,b,label='',color='#60748a',points=None,ports=None,labelpos=None):
        self.edges.append(dict(a=a,b=b,label=label,color=color,points=points,ports=ports,labelpos=labelpos))
    def render(self):
        scale=1.5
        im=Image.new('RGB',(int(self.w*scale),int(self.h*scale)),'#f8fafc');d=ImageDraw.Draw(im)
        def xy(p):return (int(p[0]*scale),int(p[1]*scale))
        byid={n['id']:n for n in self.nodes}
        def center(n):return n['x']+n['w']/2,n['y']+n['h']/2
        def endpoints(a,b):
            ac,bc=center(a),center(b);dx=bc[0]-ac[0];dy=bc[1]-ac[1]
            def boundary(n,dx,dy):
                t=min((n['w']/2/abs(dx)) if dx else 1e9,(n['h']/2/abs(dy)) if dy else 1e9)
                c=center(n);return c[0]+dx*t,c[1]+dy*t
            return boundary(a,dx,dy),boundary(b,-dx,-dy)
        for e in self.edges:
            a,b=byid[e['a']],byid[e['b']]
            p,q=endpoints(a,b)
            if e['ports']:
                ax,ay,bx,by=e['ports'];p=(a['x']+a['w']*ax,a['y']+a['h']*ay);q=(b['x']+b['w']*bx,b['y']+b['h']*by)
            pts=[p]+(e['points'] or [])+[q]
            d.line([xy(v) for v in pts],fill=e['color'],width=3)
            prev=pts[-2];angle=math.atan2(q[1]-prev[1],q[0]-prev[0]); tip=xy(q)
            wing=[tip,xy((q[0]-13*math.cos(angle-.45),q[1]-13*math.sin(angle-.45))),xy((q[0]-13*math.cos(angle+.45),q[1]-13*math.sin(angle+.45)))]
            d.polygon(wing,fill=e['color'])
            if e['label']:
                mx,my=e['labelpos'] or ((p[0]+q[0])/2,(p[1]+q[1])/2)
                font=ImageFont.truetype(FONT,24)
                bb=d.textbbox((0,0),e['label'],font=font); tw=bb[2]-bb[0]
                x,y=xy((mx,my));d.rectangle((x-tw/2-7,y-14,x+tw/2+7,y+20),fill='#f8fafc');d.text((x-tw/2,y-13),e['label'],font=font,fill=e['color'])
        for n in self.nodes:
            fill,stroke=COLORS[n['color']]
            x,y,w,h=n['x'],n['y'],n['w'],n['h']
            if n['border']:d.rounded_rectangle((*xy((x,y)),*xy((x+w,y+h))),radius=18,fill=fill,outline=stroke,width=2)
            font=ImageFont.truetype(BOLD if n['id']=='title' else FONT,int(n['size']*scale))
            lines=n['text'].split('\n'); lineh=n['size']*1.55*scale; total=len(lines)*lineh
            for i,line in enumerate(lines):
                bbox=d.textbbox((0,0),line,font=font)
                if bbox[2]-bbox[0] > (w-18)*scale:raise RuntimeError(f'text overflow {self.name}/{n["id"]}: {line}')
                if total > h*scale:raise RuntimeError(f'vertical overflow {self.name}/{n["id"]}')
                d.text((int((x+14)*scale),int((y+h/2)*scale-total/2+i*lineh-3*scale)),line,font=font,fill='#172e47')
        im.save(OUT/(self.name+'.png'))
        mx=ET.Element('mxfile',{'host':'app.diagrams.net','version':'26.0.0'})
        dg=ET.SubElement(mx,'diagram',{'id':self.name,'name':self.name})
        model=ET.SubElement(dg,'mxGraphModel',{'dx':str(self.w),'dy':str(self.h),'grid':'1','gridSize':'10','page':'1','pageWidth':str(self.w),'pageHeight':str(self.h)})
        root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
        for n in self.nodes:
            fill,stroke=COLORS[n['color']]
            style=f'rounded=1;whiteSpace=wrap;html=0;align=left;verticalAlign=middle;spacingLeft=14;fontFamily=Microsoft YaHei;fontSize={n["size"]};fontColor=#172e47;fillColor={fill};strokeColor={stroke};'
            if not n['border']:style+='fillColor=none;strokeColor=none;'
            c=ET.SubElement(root,'mxCell',{'id':n['id'],'value':n['text'],'style':style,'vertex':'1','parent':'1'})
            ET.SubElement(c,'mxGeometry',{'x':str(n['x']),'y':str(n['y']),'width':str(n['w']),'height':str(n['h']),'as':'geometry'})
        for i,e in enumerate(self.edges):
            estyle=f'endArrow=block;html=0;strokeWidth=2;strokeColor={e["color"]};fontFamily=Microsoft YaHei;fontSize=16;fontColor={e["color"]};labelBackgroundColor=#f8fafc;'
            if e['ports']:
                ax,ay,bx,by=e['ports'];estyle+=f'exitX={ax};exitY={ay};entryX={bx};entryY={by};exitPerimeter=0;entryPerimeter=0;'
            c=ET.SubElement(root,'mxCell',{'id':f'edge{i}','value':e['label'],'style':estyle,'edge':'1','parent':'1','source':e['a'],'target':e['b']})
            geo=ET.SubElement(c,'mxGeometry',{'relative':'1','as':'geometry'})
            if e['points']:
                ar=ET.SubElement(geo,'Array',{'as':'points'})
                for x,y in e['points']:ET.SubElement(ar,'mxPoint',{'x':str(x),'y':str(y)})
            if e['labelpos']:
                # Absolute editable label node preserves the same label location as PNG.
                c.attrib['value']=''
                lx,ly=e['labelpos'];label=ET.SubElement(root,'mxCell',{'id':f'label{i}','value':e['label'],'style':f'text;html=0;fontFamily=Microsoft YaHei;fontSize=16;align=center;verticalAlign=middle;fillColor=#f8fafc;strokeColor=none;fontColor={e["color"]};','vertex':'1','parent':'1'})
                ET.SubElement(label,'mxGeometry',{'x':str(lx-90),'y':str(ly-15),'width':'180','height':'30','as':'geometry'})
        ET.indent(mx);ET.ElementTree(mx).write(OUT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)
        return {'name':self.name,'nodes':len(self.nodes),'edges':len(self.edges),'png':list(im.size)}

s=Scene('01-end-to-end','两个 Collective → 一个可启动 GEMM','本例：FP16 A/B → FP32 Acc/D，C=void，tile 64×128×64，cluster 1×4×1，1SM',1740,1420)
for x,title in [(30,'① 编译期：组成类型'),(610,'② 主机：参数与启动'),(1190,'③ 设备：一次 kernel 内')]:s.box('h'+str(x),x,155,520,65,title,'gray',24)
s.box('epi',30,250,520,120,'Epilogue Builder → Epilogue 类型\n输出类型 / fusion / SMEM / subtile','purple')
s.box('main',30,420,520,120,'Mainloop Builder → Mainloop 类型\nMMA / TMA / SMEM / stage / schedule','blue')
s.box('kernel',30,590,520,140,'GemmUniversal<Shape<int,int,int,int>,\nMainloop, Epilogue>\n选中 SM100 kernel 编排特化','green',19)
s.box('adaptertype',30,780,520,110,'GemmUniversalAdapter<GemmKernel>\n得到 host handle 类型 Gemm','green')
s.box('test',610,250,520,120,'TestAll<Gemm> → Testbed::run\n固定布局；枚举尺寸与调度参数','orange')
s.box('args',610,420,520,120,'Arguments + workspace + can_implement\nA/B/D 指针、stride、MNKL、标量','blue',19)
s.box('params',610,590,520,140,'initialize → Kernel 参数转换\nMainloop / Epilogue / Scheduler\n生成 descriptor、保存 Params','blue')
s.box('launch',610,780,520,110,'run → grid / block / cluster / SMEM\nClusterLauncher → CUDA launch','green',19)
s.box('entry',1190,250,520,120,'device_kernel<GemmKernel>(Params)\n→ operator()(Params, smem)','green')
s.box('init',1190,420,520,120,'构造两个 Collective、pipeline、scheduler\n共享存储 / warp 分工 / TMEM 分配','gray',19)
s.box('mma',1190,590,520,140,'MainloopLoad：GMEM → SMEM\nMMA：SMEM → UMMA → TMEM\n发布 accumulator ready','blue')
s.box('store',1190,780,520,110,'Epilogue：TMEM → RMEM → fusion\n→ SMEM D → TMA store → GMEM D','purple',19)
s.box('verify',610,1010,1100,130,'设备完成 → synchronize → host::Gemm3x/Gett → compare_reference\n返回 pass → TestAll 内层断言 → TEST 的 EXPECT_TRUE(pass)','green',23)
s.box('limits',30,1190,1680,160,'边界：run 成功是提交状态；同步与数值比较是后续证据。\n当前 TestAll 定义缺失函数名；can_implement 不支持时 Testbed 可返回 true。\n本图为静态源码说明，本轮未编译、未运行 GPU/SDC，不能视为数值通过。','red',22)
for a,b in [('epi','main'),('main','kernel'),('kernel','adaptertype'),('test','args'),('args','params'),('params','launch'),('entry','init'),('init','mma'),('mma','store'),('store','verify')]:s.arrow(a,b)
s.arrow('adaptertype','test',points=[(580,835),(580,310)],ports=(1,.5,0,.5))
s.arrow('launch','entry',points=[(1160,835),(1160,310)],ports=(1,.5,0,.5))
scenes=[s.render()]

s=Scene('02-collective-handoff','Kernel 如何组织两个 Collective','同一 launch 内的角色协作；箭头表示数据/同步依赖，时间向下，实际运行允许重叠',1900,1660)
s.box('common',30,160,1840,110,'Kernel 的共同上下文：ProblemShape + CTA tile 坐标 + SharedStorage + TMEM base\nMainloopPipeline 管 A/B；AccumulatorPipeline 管累加器所有权；EpiStorePipeline 管输出缓冲','gray',23)
xs=[30,410,790,1170,1550]; heads=['Sched / warp 1','Load / warp 2','MMA / warp 0','EpiLoad / warp 3','Epilogue / warp ≥4']
for i,(x,t) in enumerate(zip(xs,heads)):s.box('head'+str(i),x,300,320,70,t,'gray',21)
s.box('sched',30,415,320,145,'获取下一工作 tile\nCLC 或调度实现\n输出统一 MNKL 坐标','orange',21)
s.box('load',410,415,320,145,'producer_acquire\nA/B TMA → SMEM stage\n绑定异步完成 barrier','blue',20)
s.box('mma',790,415,320,145,'consumer_wait(A/B)\n读取 SMEM descriptor\ncute::gemm → UMMA','blue',20)
s.box('cload',1170,415,320,145,'按需加载 C / 辅助输入\n独立 load pipeline\n本例 C=void：无 C load','purple',19)
s.box('cwait',1550,415,320,145,'获得同一 tile 坐标\n以及 accumulator 视图\n按需等待 C / 辅助输入','purple',20)
s.box('releaseab',410,650,320,130,'A/B stage 所有权归还\n经 UMMA 同步语义\n才允许下一次覆盖','blue',20)
s.box('acc',790,650,320,130,'TMEM accumulator\n首个 Atom：覆盖\n后续 Atom：累加','green',21)
s.box('waitacc',1550,650,320,130,'acc.consumer_wait\n等异步 MMA 可读\nTMEM → 寄存器','green',20)
s.box('commit',790,880,320,145,'mma 返回到 kernel\nacc.producer_commit\n发布对应 stage 完成','green',20)
s.box('racc',1550,880,320,145,'读完所需 TMEM 数据\nfence + consumer_release\n累加器 stage 可复用','green',18)
s.box('fusion',1550,1120,320,145,'visit：缩放 / 融合\nRMEM → SMEM D\nfence → TMA store','purple',21)
s.box('next',30,1120,1080,145,'持续处理后续输出 tile\nA/B stage、accumulator stage、scheduler stage 各自推进\n不同角色可以重叠；不把局部顺序当全局时间线','orange',21)
s.box('tail',30,1380,1840,170,'退出：load_tail / accumulator drain / epilogue store_tail / TMEM deallocation 同步 → free\n核心交接契约 = 相同工作 tile + 相同 TMEM 视图 + ready/empty 协议 + 异步可见性。\nSDC200 必须为每一条完成/复用依赖提供等价语义；仅替换 fma builtin 不足以完成移植。','red',23)
for a,b in [('sched','load'),('load','mma'),('mma','acc'),('mma','releaseab'),('acc','commit'),('cload','cwait'),('cwait','waitacc'),('waitacc','racc'),('racc','fusion'),('fusion','tail'),('next','tail')]:s.arrow(a,b)
s.arrow('commit','waitacc','ready',points=[(1135,952.5),(1135,715)],ports=(1,.5,0,.5),labelpos=(1320,715))
s.arrow('racc','acc','empty / 可复用',points=[(1505,952.5),(1505,815),(950,815)],ports=(0,.5,.5,1),labelpos=(1280,815))
scenes.append(s.render())

s=Scene('03-concept-cards','九张卡片理解完整流程','蓝色：类型与输入计算   绿色：组织与交接   紫色：输出   橙色：测试与移植',1800,1390)
cards=[
('01 运行期问题','Shape<int,int,int,int>\n数值为 (M,N,K,L)\nL 是 batch 数\n与 tile / cluster 分开','blue'),
('02 Builder 产出类型','Epilogue 先确定输出侧资源\nMainloop 再选 MMA / copy / stage\nCollectiveOp 是类型\n此时没有启动计算','blue'),
('03 Mainloop','load 角色：A/B → SMEM\nMMA 角色：K 循环 → TMEM\n首项覆盖、后续累加\nC=void 不会取消内部累加','blue'),
('04 Kernel 编排器','共享存储 + 参数聚合\nwarp 角色 + scheduler\n分配 TMEM + 构造两 Collective\n共同维护 tile 与退出协议','green'),
('05 Collective 交接','同一输出 tile 坐标\n同一 TMEM accumulator 视图\nproducer_commit → consumer_wait\n读完 release → 允许复用','green'),
('06 Epilogue','TMEM → 寄存器 → fusion\n输出转换 → SMEM → GMEM D\n本例 D = alpha × Acc\nstore_tail 确保输出缓冲排空','purple'),
('07 Adapter / launch','Args → Params 转换入口\nworkspace / SMEM / grid / cluster\n真正入口 device_kernel<Kernel>\nrun 成功后仍需同步与比较','green'),
('08 TestAll / EXPECT_TRUE','固定 Gemm 的布局类型\n枚举尺寸、raster、swizzle 等\n多布局需多个 Gemm 实例\n不支持可能返回 true：须计 skip','orange'),
('09 SDC200 适配','优先保留上层抽象契约\n重做 ISA / ABI / 异步完成\n核验 TMEM / TMA / cluster / CLC\nAtom → 单 CTA → 流水线 → 集成','orange')]
for i,(title,body,color) in enumerate(cards):
    x=30+(i%3)*590;y=165+(i//3)*385
    s.box('card'+str(i),x,y,560,335,title+'\n\n'+body,color,21)
s.box('foot',30,1320,1740,45,'源码核对与图稿验证已做；编译、设备运行、数值、机器码、性能未在本轮验证。','gray',19,False)
scenes.append(s.render())

s=Scene('04-sdc200-porting','SDC200 适配地图与验收顺序','“对标 SM100”不等于 descriptor、TMEM、cluster、CLC 或 CUDA launch ABI 兼容',1800,1440)
s.box('reuse',30,170,1740,150,'上层优先复用：GEMM 数学语义、ProblemShape、CuTe Layout/Tensor、Atom 框架、公共 Builder/Kernel/Adapter 签名\n测试组织、CPU reference 与比较工具尽量保留；host/device 注解、数据类型与 runtime 依赖仍需审查。','green',23)
s.box('compose',30,390,540,205,'组合层：新增 SDC 策略与实现\nMainloop / Epilogue Builder\nDispatchPolicy / Kernel 特化\nSMEM/TMEM 资源和 warp 分工','blue',23)
s.box('sync',630,390,540,205,'协议层：必须确认等价语义\nPipeline acquire/commit/wait/release\nTMA/UMMA completion 与 fence\nTMEM 生命周期 / cluster scope','purple',22)
s.box('runtime',1230,390,540,205,'运行时：替换或证明兼容\n参数 ABI / descriptor encoder\nlaunch / cluster / memory / stream\n硬件查询 / 错误 / 同步','orange',22)
s.box('isa',30,680,1740,145,'底层必须逐项闭环：MMA Operation + Traits + SMEM/instruction descriptor + TMEM allocator / T2R\nTMA copy / TensorMap + barrier / fence + election / cluster + scheduler CLC 或软件替代','red',23)
s.box('existing',30,895,1740,130,'已见：SDC200_MMA_F16BF16_SS::fma → __sdc_uten_mma_bf16；SDC descriptor union。\n仍待验收：FP16/BF16 模式、编码、完成事件；cluster_sdc200.hpp 仍含 NVIDIA election/cluster PTX。','gray',23)
for i,t in enumerate(['P0 契约 / 能力表','P1 单 Atom 闭环','P2 单 CTA / 静态调度','P3 多级异步流水线','P4 cluster / CLC / 融合']):
    s.box('gate'+str(i),30+i*354,1110,324,100,t,'green',19)
    if i:s.arrow('gate'+str(i-1),'gate'+str(i))
s.box('accept',30,1280,1740,100,'P5 集成验收：候选 / skip / initialize / launch / synchronize / 数值通过分别计数；承诺支持的配置不能静默跳过。\n后续再验证机器码与性能。完整文件/API 清单见 68 项适配参考表；不支持能力应显式拒绝。','orange',21)
s.arrow('reuse','compose');s.arrow('reuse','sync');s.arrow('reuse','runtime')
s.arrow('compose','isa');s.arrow('sync','isa');s.arrow('runtime','isa');s.arrow('isa','existing')
scenes.append(s.render())

extra=[(SDC/'include/cute/arch/mma_sdc200_umma.hpp',[(72,121)]),(SDC/'include/cute/arch/mma_sdc200_desc.hpp',[(109,160)]),(SDC/'include/cute/arch/cluster_sdc200.hpp',[(175,207)]),(Path('D:/lvz/sdc/sccnext-main/sccnext-main/clang/lib/Headers/sdc/sdc_builtins.h'),[(892,900)])]
excerpts=[]
for rr in rows:
    p=ROOT/rr['文件'];ls=p.read_text(encoding='utf-8-sig').splitlines();a=max(1,rr['行号']-3);b=min(len(ls),rr['行号']+16)
    excerpts.append(f'\n### {rr["阶段"]} | {p}:{rr["行号"]}\n'+'\n'.join(f'{i}: {ls[i-1]}' for i in range(a,b+1)))
for p,ranges in extra:
    raw=p.read_bytes();ls=raw.decode('utf-8-sig').splitlines();evidence[str(p)]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'lines':len(ls)}
    for a,b in ranges:excerpts.append(f'\n### {p}:{a}\n'+'\n'.join(f'{i}: {ls[i-1]}' for i in range(a,b+1)))
(OUT/'source-excerpts.txt').write_text('\n'.join(excerpts),encoding='utf-8')
(OUT/'source-evidence.json').write_text(json.dumps({'generated':datetime.now(timezone(timedelta(hours=8))).isoformat(),'sm100_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'sdc_head':subprocess.check_output(['git','-C',str(SDC),'rev-parse','HEAD'],text=True).strip(),'sources':evidence},ensure_ascii=False,indent=2),encoding='utf-8')

# Offline reading page, primary report and long table remain separate for navigation.
try:
    import markdown
    def md(t):return markdown.markdown(t,extensions=['tables','fenced_code'])
except ImportError:
    def md(t):
        node='C:/Users/lvzhe/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        script="const fs=require('fs');import('file:///C:/Users/lvzhe/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js').then(({marked})=>process.stdout.write(marked.parse(fs.readFileSync(0,'utf8'))))"
        return subprocess.run([node,'-e',script],input=t,text=True,encoding='utf-8',capture_output=True,check=True).stdout
style='''body{font-family:"Microsoft YaHei",sans-serif;background:#f5f8fc;color:#19324e;margin:0;line-height:1.8}main{max-width:1200px;margin:auto;padding:30px;background:white}h1,h2,h3{line-height:1.4;color:#173e67}h2{margin-top:55px;border-bottom:2px solid #e1e9f2;padding-bottom:12px}a{color:#2369b1}img{max-width:100%;height:auto;border:1px solid #d8e2ee}table{display:block;overflow:auto;border-collapse:collapse;font-size:14px}th,td{border:1px solid #dce4ed;padding:10px;vertical-align:top;min-width:115px}th{background:#eaf2ff}pre{background:#edf3f9;padding:20px;overflow:auto}code{font-family:Consolas,monospace}nav{position:sticky;top:0;background:#173e67;color:white;padding:12px;z-index:2}nav a{color:white;margin-right:20px}.note{background:#fff2df;padding:15px;border-radius:8px}'''
for source,target in [('report.md','index.html'),('sdc200-adaptation.md','sdc200-adaptation.html')]:
    content=(OUT/source).read_text(encoding='utf-8').replace('(sdc200-adaptation.md)','(sdc200-adaptation.html)').replace('(report.md)','(index.html)')
    body=md(content)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>GEMM 完整流程与 SDC200 适配</title><style>'+style+'</style><nav><a href="index.html">完整说明与四张图</a><a href="sdc200-adaptation.html">68 项适配表</a><a href="sdc200-adaptation.csv">CSV</a></nav><main>'+body+'</main></html>'
    (OUT/target).write_text(page,encoding='utf-8')

validation={'table_rows':len(rows),'source_files':len(evidence),'all_source_files_and_regex_anchors_exist':True,'diagrams':scenes,'xml_parse':True,'png_visual_review':'pending','utf8':True,'compiled':False,'device_run':False,'numerical_validation':False,'sass_or_sdc_machine_code':False,'performance':False,'source_modified':False,'cross_computer_sync':False}
for scene in scenes:
    tree=ET.parse(OUT/(scene['name']+'.drawio'));root=tree.getroot()
    ids={v.attrib['id'] for v in root.iter('mxCell')}
    for e in root.iter('mxCell'):
        if e.attrib.get('edge')=='1':assert e.attrib['source'] in ids and e.attrib['target'] in ids
for p in OUT.glob('*.md'):
    content=p.read_text(encoding='utf-8');assert '\ufffd' not in content
    assert not any(line.endswith(' ') for line in content.splitlines()),p
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'sources':len(evidence),'diagrams':scenes,'output':str(OUT)},ensure_ascii=False))
