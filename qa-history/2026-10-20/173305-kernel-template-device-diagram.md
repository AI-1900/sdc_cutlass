# 模板组合、编译与 kernel 内部协作图

时间：2026-10-20T17:33:05.279193+08:00

## 用户问题

生成图片来直观描述 测例通过模板组合出 kernel 类型，编译器再生成具体的入口与设备代码。两个 Collective 是这个 kernel 内部的协作组件的过程

## 实际交付的完整回答

这张图分成三个阶段，并把两个 Collective 放在**同一个 kernel 的编排主体内部**：

![从模板组合到 kernel 生成与执行](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/kernel-template-to-device/kernel-template-to-device.png)

[PNG 原图](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/kernel-template-to-device/kernel-template-to-device.png) · [可编辑 Draw.io](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/kernel-template-to-device/kernel-template-to-device.drawio) · [说明与源码位置](C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/kernel-template-to-device/explanation.md)

两份图稿的结构、文字和连线一致；字体与抗锯齿可能略有差异。这是静态源码示意，本轮未编译或运行设备。

<oai-mem-citation>
<citation_entries>
MEMORY.md:226-227|note=[PNG and editable diagram delivery and opening preferences]
skills/cxx-cute-documentation-delivery/SKILL.md:17-33|note=[source grounded diagram and validation boundary]
</citation_entries>
<rollout_ids>
</rollout_ids>
</oai-mem-citation>

## 完整说明

# 测例模板如何生成并启动 kernel

图分为三个阶段：

1. 测例提供元素类型、布局、tile、cluster 等编译期配置，两个 Builder 选择 Mainloop/Epilogue 类型，GemmUniversal 将它们组合成 GemmKernel；Adapter 再包装成主机 handle 类型。
2. 编译器处理测例与其 include 的模板实现，实例化 host handle、device_kernel<GemmKernel> 及其编排逻辑，并进行设备编译和优化。源码类/函数/文件边界不保证作为独立机器码函数边界保留。
3. 主机测试通过 Adapter 转换参数并提交 launch。具体 global 入口构造 GemmKernel，调用 operator()；该主体组织两个 Collective，Mainloop 将累加器写入 TMEM，Epilogue 通过相同 tile、TMEM 视图和 pipeline 协议消费结果。

图中的 Collective 是逻辑组件，不是两次独立 kernel launch。四个 int 表示运行时 M/N/K/L 的类型，编译期 tile 与运行时尺寸必须区分。

## 本轮核对的源码

- D:/000_SDC/001TASK/02_Research/cutlass/test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32.cu:234：GemmKernel 类型组合；:244：Adapter 类型。
- D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/device_kernel.h:118：实际 global 入口；:122–123：构造 Operator 并调用 operator()。
- D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:458：启动目标 device_kernel<GemmKernel>。
- D:/000_SDC/001TASK/02_Research/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:435–436：kernel 中构造两个 Collective。

## 图稿与验证边界

PNG 和 Draw.io 使用同一节点、文字、层级、连线与坐标数据。Draw.io 中各个图形、文字和箭头均可编辑；GPU 框包含入口，入口调用的编排主体内包含两个 Collective。字体字形、行距、抗锯齿和箭头尖端在不同渲染器中可能略有差异。

已检查 XML、文本边界与图片尺寸；未在本轮编译、运行 GPU 或验证机器码。现工作树的 TestAll 入口仍有此前记录的函数名缺失，本图描述源码的预期结构，不表示当前测例可直接编译通过。


## 验证

{
  "png_size": [
    3060,
    3240
  ],
  "editable_vertices": 52,
  "editable_edges": 13,
  "drawio_xml_and_parent_edge_references": "passed",
  "text_bounds": "passed in generation",
  "png_visual_review": "passed",
  "same_scene_model": true,
  "compiled": false,
  "device_run": false,
  "machine_code_checked": false,
  "cross_computer_sync": false
}

成果保存在 C:/Users/lvzhe/.codex/.chatgpt-projects/g-p-6a7c4e38fe348191a166ee0e1d330af6/deliverables/kernel-template-to-device。本轮问答本地保存完成；跨电脑同步未完成。
