from pathlib import Path
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
import csv, hashlib, json, re, zipfile
import xml.etree.ElementTree as ET

out=Path(__file__).resolve().parent
project=out.parent.parent
report=out/'report.md'
text=report.read_text(encoding='utf-8')
text=text.replace('见 65 项','见 68 项')
report.write_text(text,encoding='utf-8')

# Final verification after all generation, including exact graph content equality.
ev=json.loads((out/'source-evidence.json').read_text(encoding='utf-8'))
for v in ev['sources'].values():
    assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'],v['path']
class AuditHTML(HTMLParser):
    def __init__(self):super().__init__();self.tags={};self.imgs=[]
    def handle_starttag(self,tag,attrs):
        self.tags[tag]=self.tags.get(tag,0)+1
        if tag=='img':self.imgs.append(dict(attrs)['src'])
ha=AuditHTML();ha.feed((out/'index.html').read_text(encoding='utf-8'))
assert ha.tags.get('h1')==1 and len(ha.imgs)==4 and ha.tags.get('table',0)>5
for img in ha.imgs:assert (out/img).is_file()
hb=AuditHTML();hb.feed((out/'sdc200-adaptation.html').read_text(encoding='utf-8'))
assert hb.tags.get('tr')==69,hb.tags
with (out/'sdc200-adaptation.csv').open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==68
validation=json.loads((out/'validation.json').read_text(encoding='utf-8'))
for scene in validation['diagrams']:
    tree=ET.parse(out/(scene['name']+'.drawio'))
    nodes=[v for v in tree.iter('mxCell') if v.attrib.get('vertex')=='1']
    edges=[v for v in tree.iter('mxCell') if v.attrib.get('edge')=='1']
    assert len(nodes)>=scene['nodes'] and len(edges)==scene['edges']
validation.update(png_visual_review='reviewed all four PNGs; revised launch routing and ready/empty label routing',html_structure_and_local_image_paths=True,csv_rows=68,all_source_hashes_unchanged_at_final_verification=True,drawio_nodes_are_editable=True,visual_differences='Font glyphs, antialiasing, arrow tip geometry may differ between Pillow and diagrams.net.')
(out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')

base=out.as_posix()
def link(name,label=None):return f'[{label or name}]({base}/{name})'
final=f'''已完成源码核对、详细说明、四组图稿和 **68 项 SDC200 适配参考表**。

建议从{link('index.html','完整阅读页')}开始；另有{link('report.md','详细 Markdown')}、{link('sdc200-adaptation.md','完整适配表')}和{link('sdc200-adaptation.csv','可筛选 CSV')}。

核心过程是：Builder 生成两个 Collective 类型 → `GemmUniversal` 组织参数、共享存储、warp 分工、调度与同步 → Adapter 转换参数并启动 `device_kernel` → Mainloop 将结果累加到 TMEM → Epilogue 等待结果、融合并写 D → Testbed 同步、计算主机参考并比较 → `EXPECT_TRUE(pass)` 登记结果。

两个 Collective 的交接核心是 **相同 tile 坐标、同一 TMEM 累加器视图，以及 producer/consumer 流水线协议**。文档逐项展开了初始化、A/B 搬运、MMA、累加器交接、输出与退出过程，并附实际源码位置。

| 图稿 | 图片 | 可编辑源文件 |
|---|---|---|
| 完整调用流程 | {link('01-end-to-end.png','PNG')} | {link('01-end-to-end.drawio','Draw.io')} |
| 两个 Collective 的协作 | {link('02-collective-handoff.png','PNG')} | {link('02-collective-handoff.drawio','Draw.io')} |
| 九张核心卡片 | {link('03-concept-cards.png','PNG')} | {link('03-concept-cards.drawio','Draw.io')} |
| SDC200 适配地图 | {link('04-sdc200-porting.png','PNG')} | {link('04-sdc200-porting.drawio','Draw.io')} |

适配表逐项列出文件、API、可复用部分、需要适配的实现、建议落点与验收条件，覆盖 Builder、两个 Collective、Kernel、Adapter、调度、TMA、TMEM、MMA、同步、运行时、编译器和测试。

源码核对还确认了三点：

- 单次 `TestAll<Gemm>()` 固定矩阵布局类型；多布局由不同 Gemm 实例覆盖。
- 当前工作树的 `TestAll` 定义缺失函数名；而 Testbed 对某些不支持输入会返回 `true`，所以需要单独统计实际执行与跳过。
- 已有 SDC200 MMA builtin 包装，但 cluster 文件仍含 NVIDIA PTX，整条路径的适配尚不能据此认定完成。

{link('gemm-universal-sdc200.zip','下载完整资料包')}。本轮完成静态源码与文档图稿检查，未进行 CUDA/SDC 编译、设备运行、数值或性能验证。图与源文件主要结构一致，字体和抗锯齿可能因渲染器略有差异。问答已本地保存，跨电脑同步未完成。

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
</oai-mem-citation>'''
(out/'final-response.md').write_text(final,encoding='utf-8')
question='''详细介绍 GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue> 把两个 Collective 组织成可启动 kernel（int×4 是运行期 M/N/K/ 批次数）；GemmUniversalAdapter 提供 device 级参数打包与 launch；最后 TestAll<Gemm>() 跑多组问题规模 × 布局组合的标准测试，EXPECT_TRUE(pass) 断言 这个完整的过程，以及这俩collective 是如何组织详细具体过程，生成对应的流程图片和卡片直观说明。同时给出若需适配一个对标SM100 的私有架构SDC200，整个流程中有哪些文件和API需要适配，给出一张完备详细的适配参考表格，包含整个流程中可以复用的文件和API，以及需要适配的API和文件'''
now=datetime.now(timezone(timedelta(hours=8)))
qadir=project/'qa-history'/'2026-10-20';qadir.mkdir(parents=True,exist_ok=True)
state=out/'qa-record-path.txt'
qa=Path(state.read_text(encoding='utf-8')) if state.exists() else qadir/(now.strftime('%H%M%S')+'-gemm-universal-collectives-sdc200.md')
state.write_text(str(qa),encoding='utf-8')
artifacts='\n'.join(f'- [{p.name}]({p.as_posix()})' for p in sorted(out.iterdir()) if p.is_file() and p.suffix not in ['.py','.zip'])
qa.write_text(f'# GEMM Collective 编排与 SDC200 适配\n\n记录时间：{now.isoformat()}\n\n## 用户问题\n\n{question}\n\n## 实际交付的完整回答\n\n{final}\n\n## 详细成果\n\n{artifacts}\n\n## 完整技术说明\n\n'+report.read_text(encoding='utf-8')+'\n\n## 完整适配表\n\n'+(out/'sdc200-adaptation.md').read_text(encoding='utf-8')+'\n\n## 验证与未完成项\n\n'+json.dumps(validation,ensure_ascii=False,indent=2)+'\n\n未执行编译、设备运行、数值、机器码、性能验证；未更改 CUTLASS/SDC 源码；没有已验证授权的跨电脑同步渠道，故同步未完成。\n',encoding='utf-8')
with zipfile.ZipFile(out/'gemm-universal-sdc200.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file() and p.suffix not in ['.zip','.py'] and p.name!='qa-record-path.txt':z.write(p,p.name)
    z.write(qa,'qa-history/'+qa.name)
print(json.dumps({'qa':str(qa),'zip':str(out/'gemm-universal-sdc200.zip'),'files':len(list(out.iterdir())),'validation':validation},ensure_ascii=False))
