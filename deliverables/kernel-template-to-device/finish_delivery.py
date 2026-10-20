from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib
import xml.etree.ElementTree as ET
from PIL import Image

out=Path(__file__).resolve().parent
base=out.as_posix()
png=out/'kernel-template-to-device.png'
drawio=out/'kernel-template-to-device.drawio'
root=ET.parse(drawio)
cells=list(root.iter('mxCell'));ids={c.attrib['id'] for c in cells}
for c in cells:
    if c.attrib.get('edge')=='1':assert c.attrib['source'] in ids and c.attrib['target'] in ids
    if 'parent' in c.attrib:assert c.attrib['parent'] in ids
im=Image.open(png);assert im.size==(3060,3240)
validation={'png_size':list(im.size),'editable_vertices':sum(c.attrib.get('vertex')=='1' for c in cells),'editable_edges':sum(c.attrib.get('edge')=='1' for c in cells),'drawio_xml_and_parent_edge_references':'passed','text_bounds':'passed in generation','png_visual_review':'passed','same_scene_model':True,'compiled':False,'device_run':False,'machine_code_checked':False,'cross_computer_sync':False}
(out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
final=f'''这张图分成三个阶段，并把两个 Collective 放在**同一个 kernel 的编排主体内部**：

![从模板组合到 kernel 生成与执行]({base}/kernel-template-to-device.png)

[PNG 原图]({base}/kernel-template-to-device.png) · [可编辑 Draw.io]({base}/kernel-template-to-device.drawio) · [说明与源码位置]({base}/explanation.md)

两份图稿的结构、文字和连线一致；字体与抗锯齿可能略有差异。这是静态源码示意，本轮未编译或运行设备。

<oai-mem-citation>
<citation_entries>
MEMORY.md:226-227|note=[PNG and editable diagram delivery and opening preferences]
skills/cxx-cute-documentation-delivery/SKILL.md:17-33|note=[source grounded diagram and validation boundary]
</citation_entries>
<rollout_ids>
</rollout_ids>
</oai-mem-citation>'''
(out/'final-response.md').write_text(final,encoding='utf-8')
now=datetime.now(timezone(timedelta(hours=8)))
qa=out.parent.parent/'qa-history'/'2026-10-20'/(now.strftime('%H%M%S')+'-kernel-template-device-diagram.md')
question='生成图片来直观描述 测例通过模板组合出 kernel 类型，编译器再生成具体的入口与设备代码。两个 Collective 是这个 kernel 内部的协作组件的过程'
qa.write_text(f'# 模板组合、编译与 kernel 内部协作图\n\n时间：{now.isoformat()}\n\n## 用户问题\n\n{question}\n\n## 实际交付的完整回答\n\n{final}\n\n## 完整说明\n\n'+(out/'explanation.md').read_text(encoding='utf-8')+'\n\n## 验证\n\n'+json.dumps(validation,ensure_ascii=False,indent=2)+'\n\n成果保存在 '+base+'。本轮问答本地保存完成；跨电脑同步未完成。\n',encoding='utf-8')
print(json.dumps({'qa':str(qa),'validation':validation},ensure_ascii=False))
