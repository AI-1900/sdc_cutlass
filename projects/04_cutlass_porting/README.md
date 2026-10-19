# 04_cutlass_porting

CUTLASS 源码分析、CuTe 映射、GEMM 执行链与后端移植学习资料归档。

## 阅读入口

- [项目问答全文与索引](conversations/README.md)
- [CUTLASS 架构学习与源码调研指南](CUTLASS_架构学习与源码调研指南.md)
- [源码与架构完整指南](deliverables/cutlass-guide/CUTLASS源码与架构指南.md)
- [离线 HTML 指南](deliverables/cutlass-guide/CUTLASS源码与架构指南.html)
- [SM100 GEMM 调用链指南](SM100_f16_GEMM_64x128x64_1SM_调用链指南.md)
- [SM100 流程图、架构图与 MMA Atom](SM100-GEMM-visual/index.html)
- [128×256×64 最小 GEMM 示例与推导](gemm_minimal/README_推导.md)
- [补充交互页面与配图](visualizations/)
- [归档文件清单与 SHA-256](archive-manifest.json)

HTML 交互页面请下载后用浏览器打开，GitHub 文件预览不会执行页面脚本。

## 归档范围

本分支保存当前项目目录的内容，以及能从本机记录中确认属于该项目的用户问答和可视化产物。问答保留可见的用户提问、助手回复及进度消息，不包含内部推理、系统指令、环境注入和工具原始输出。当前同步会话以导出时刻为截止点。仅在云端、未同步到本机的内容不在本次可验证范围。

`sources/` 为只读同步参考资料；本次归档不修改这些文件。教程中的源码路径、硬件条件、历史验证结论保留原记录，归档操作不代表重新运行 CUDA 示例或重新验证性能。

`tools/export_project.py` 可在原本机项目目录重新导出问答、收集会话对应的可视化目录并更新文件校验清单。

后续手动同步：先运行 `python tools/export_project.py`，再运行 `python tools/sync_project.py --push`。同步使用 `.sync/` 中独立的 Git 存储，仅更新远端项目分支及其中的归档目录；该本地同步缓存不进入归档。此操作没有创建定时同步任务。
