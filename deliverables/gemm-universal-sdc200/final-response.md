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