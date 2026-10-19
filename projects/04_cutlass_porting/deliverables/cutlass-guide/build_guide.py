from pathlib import Path
import html, re, json, subprocess, zipfile

OUT = Path(__file__).resolve().parent
REPO = Path('E:/000wsp/01repo/08sutlass/cutlass')
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)

# Each entry is checked against the actual analyzed checkout, not a remote branch.
sources = [
('S01','版本声明','include/cutlass/version.h',r'#define CUTLASS_MAJOR'),
('S02','3.x 架构设计','media/docs/cpp/cutlass_3x_design.md',r'# CUTLASS 3.0 Design'),
('S03','3.x GEMM API 与层级','media/docs/cpp/gemm_api_3x.md',r'## CUTLASS GEMM Components'),
('S04','Hopper Builder 与 EVT 组装','examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu',r'using CustomEVT'),
('S05','SM90 Builder 生成具体 Collective','include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl',r'using CollectiveOp = CollectiveMma'),
('S06','Hopper cooperative Kernel 特化','include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',r'class GemmUniversal<'),
('S07','应用 Arguments 与调用顺序','examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu',r'bool run\(const Options&'),
('S08','Adapter 初始化与 Params','include/cutlass/gemm/device/gemm_universal_adapter.h',r'^  initialize\('),
('S09','Adapter launch 分支','include/cutlass/gemm/device/gemm_universal_adapter.h',r'^  run\(Params& params'),
('S10','Kernel Arguments 转换','include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',r'to_underlying_arguments\('),
('S11','CUDA device kernel 入口','include/cutlass/device_kernel.h',r'void device_kernel\('),
('S12','Kernel mainloop load 调用','include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp',r'collective_mainloop.load\('),
('S13','Hopper Collective 的 load 与 mma','include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',r'load_init\('),
('S14','MMA Atom 与 TiledMMA','include/cute/atom/mma_atom.hpp',r'struct MMA_Atom<MMA_Traits'),
('S15','CuTe GEMM 调用 Atom','include/cute/algorithm/gemm.hpp',r'mma.call\(D, A, B, C\)'),
('S16','GMMA 专用 unpack','include/cute/atom/mma_traits_sm90_gmma.hpp',r'mma_unpack\(MMA_Traits'),
('S17','代表 WGMMA 指令封装','include/cute/arch/mma_sm90_gmma.hpp',r'struct MMA_64x8x16_F32F16F16_SS'),
('S18','TMA pipeline acquire/commit 协议','include/cutlass/pipeline/sm90_pipeline.hpp',r'void producer_acquire\(uint32_t stage, uint32_t phase\)'),
('S19','TMA PTX 搬运封装','include/cute/arch/copy_sm90_tma.hpp',r'struct SM90_TMA_LOAD_2D'),
('S20','Tile scheduler 标签与选择','include/cutlass/gemm/kernel/tile_scheduler.hpp',r'struct PersistentScheduler'),
('S21','CuTe Layout 教程','media/docs/cpp/cute/01_layout.md',r'^#'),
('S22','CuTe Tensor 教程','media/docs/cpp/cute/03_tensor.md',r'^#'),
('S23','布局代数','media/docs/cpp/cute/02_layout_algebra.md',r'^#'),
('S24','Hopper TMA epilogue','include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp',r'class CollectiveEpilogue|class Epilogue|struct CollectiveEpilogue|struct Epilogue'),
('S25','融合 operation 定义','include/cutlass/epilogue/fusion/operations.hpp',r'struct LinearCombination'),
('S26','SM100 Builder 示例','examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu',r'using CollectiveEpilogue'),
('S27','SM100 Kernel 与 TMEM 管理','include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp',r'using TmemAllocator'),
('S28','SM100 Collective MMA','include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp',r'cute::gemm\(tiled_mma'),
('S29','SM100 UMMA PTX 封装','include/cute/arch/mma_sm100_umma.hpp',r'struct SM100_MMA_F16BF16_SS'),
('S30','SM100 TMEM load 指令','include/cute/arch/copy_sm100.hpp',r'tcgen05.ld'),
('S31','手写 SM100 教程','examples/cute/tutorial/blackwell/01_mma_sm100.cu',r'gemm_device\('),
('S32','Python DSL 概览','media/docs/pythonDSL/overview.rst',r'CUTLASS Python is'),
('S33','CuTe IR 编译器栈说明','cutlass_compiler/README.md',r'^# CuTe IR'),
('S34','示例 49 构建目标','examples/49_hopper_gemm_with_collective_builder/CMakeLists.txt',r'^cutlass_example_add_executable'),
('S35','示例 71 架构条件','examples/71_blackwell_gemm_with_collective_builder/CMakeLists.txt',r'^if\(CUTLASS_NVCC_ARCHS'),
('S36','性能工具参数与用法','media/docs/cpp/profiler.md',r'--warmup-iterations'),
('S37','SM100 Pipeline 的完成通知','include/cutlass/pipeline/sm100_pipeline.hpp',r'class PipelineTmaUmmaAsync'),
('S38','UMMA arrive 的 PTX commit','include/cutlass/arch/barrier.h',r'void umma_arrive\(uint64_t const\* smem_ptr\)'),
]
records = []
for sid, title, rel, pattern in sources:
    p = REPO / rel
    lines = p.read_text(encoding='utf-8').splitlines()
    found = [i+1 for i,s in enumerate(lines) if re.search(pattern,s)]
    if not found:
        raise RuntimeError(f'No source anchor: {sid} {rel} {pattern}')
    records.append(dict(id=sid,title=title,path=rel,line=found[0],absolute=p.as_posix()))
(OUT/'source-index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

INK='#16324f'; MUTED='#53687d'; BLUE='#236caa'; TEAL='#137e79'; ORANGE='#a76025'
def esc(s): return html.escape(str(s))
class SVG:
    def __init__(self,w,h,title,subtitle=''):
        self.w,self.h=w,h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9" fill="{BLUE}"/></marker></defs><rect width="{w}" height="{h}" rx="18" fill="#f7fafc"/>']
        self.text(30,39,title,24,INK,'bold')
        if subtitle:self.text(30,68,subtitle,15,MUTED)
    def text(self,x,y,s,size=17,color=INK,weight='normal',anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')
    def box(self,x,y,w,h,title,lines=(),color=BLUE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="{color}" stroke-width="1.6"/>')
        self.text(x+15,y+28,title,18,color,'bold')
        for i,s in enumerate(lines):self.text(x+15,y+53+i*23,s,15)
    def arrow(self,x1,y1,x2,y2,label='',dash=False):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{BLUE}" stroke-width="2" fill="none" marker-end="url(#arrow)"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
        if label:self.text((x1+x2)/2+8,(y1+y2)/2-7,label,14,MUTED)
    def save(self,name):
        self.parts.append('</svg>')
        marker='arrow-'+name.removesuffix('.svg')
        svg=''.join(self.parts).replace('id="arrow"',f'id="{marker}"').replace('url(#arrow)',f'url(#{marker})')
        (ASSETS/name).write_text(svg,encoding='utf-8')

s=SVG(1100,700,'01  架构：编译期组合，运行期协作','主线为当前仓库保留的 C++ 3.x 风格 API；箭头表示依赖/组合，不是逐层虚函数调用。')
layers=[('Device / Host Adapter','Arguments、workspace、Params、CUDA launch'),('Kernel / GemmUniversal','线程角色、工作 tile、shared storage、资源生命周期'),('Collective / Mainloop + Epilogue','K 循环、搬运/计算流水线、输出融合'),('CuTe / Tiled MMA + Copy','将指令铺到 tile；划分线程和数据'),('Atom + Traits','指令形状、fragment、线程/值布局、unpack'),('Architecture operations','PTX、descriptor、barrier、TMEM 操作')]
for i,(t,d) in enumerate(layers):
    y=95+i*88;s.box(35,y,680,69,t,[d],TEAL if i in (2,3) else BLUE)
    if i<5:s.arrow(375,y+69,375,y+84)
s.box(770,120,295,126,'编译期配置',['Arch / 类型 / Layout','Tile / Cluster / Stage','Schedule / Fusion'],ORANGE)
s.box(770,300,295,126,'运行时输入',['M,N,K,L / stride','A,B,C,D 指针','alpha,beta / workspace'],TEAL)
s.box(770,480,295,126,'GPU 执行资源',['线程与矩阵单元','GMEM / SMEM / 寄存器','TMEM / 异步完成机制'])
s.arrow(770,183,720,210);s.arrow(770,363,720,130);s.arrow(715,580,765,543)
s.text(35,668,'边界：CuTe 描述空间映射；Collective 与 Kernel 组织时间和协作。',17,MUTED)
s.save('01-architecture.svg')

s=SVG(1100,545,'02  CuTe：坐标、线程、存储与操作','视图构造 / partition 不会自动搬运数据；copy 和 MMA 才对应实际硬件工作。')
s.box(30,100,310,120,'逻辑矩阵与坐标',['A[M,K] / B[N,K] / D[M,N]','shape + stride → 线性偏移','动态尺寸与编译期布局可混合'])
s.box(395,100,310,120,'Tensor = Engine + Layout',['Engine：访问/存储语境','Layout：坐标映射','GMEM / SMEM / RMEM / TMEM'])
s.box(760,100,310,120,'Tiled 操作 / 线程切片',['TiledMMA / TiledCopy','get_slice(thread_idx)','partition_A/B/C 或 S/D'])
s.arrow(340,160,390,160);s.arrow(705,160,755,160)
s.box(30,300,310,130,'地址例：shape=(2,3)',['stride=(3,1)，偏移=3m+n','第 0 行：0  1  2','第 1 行：3  4  5'],TEAL)
s.box(395,300,310,130,'线程和值的映射',['哪些 lane 持有哪些 fragment','哪些线程发射 / 参与指令','逻辑布局 ≠ 物理内存连续性'],TEAL)
s.box(760,300,310,130,'Atom / Traits → 指令',['copy：触发实际搬运','gemm：展开到 MMA 操作','descriptor：编码硬件访问'],ORANGE)
s.arrow(915,220,915,294);s.arrow(705,365,755,365)
s.text(30,490,'调试时分别记录：逻辑 shape/stride、线程角色、存储位置、数据生命周期。',18,MUTED)
s.save('02-cute-mapping.svg')

s=SVG(1100,835,'03  一次 GEMM：Host → Kernel → Collective → 指令','代表路径：Hopper TMA + GMMA cooperative；其他 schedule 的分工和启动分支可能不同。')
items=[('应用','Arguments：M/N/K/L、A/B/C/D、stride、alpha/beta'),('Host Adapter','can_implement → workspace → initialize → Params'),('启动入口','run：grid / block / SMEM / cluster → device_kernel'),('Kernel','operator()：角色、pipeline、scheduler、工作 tile'),('Collective mainloop','load：TMA → SMEM；mma：等待输入 → CuTe GEMM'),('CuTe / Atom','gemm → MMA_Atom::call → traits::mma_unpack → fma'),('架构操作','内联 PTX：wgmma.mma_async...；异步完成协议'),('Epilogue / 收尾','Acc + C → 融合/转换 → D；tail → GPU 完成与检查')]
for i,(t,d) in enumerate(items):
    y=94+i*87;s.box(35,y,800,68,t,[d],TEAL if i in (3,4,7) else BLUE)
    if i<7:s.arrow(430,y+68,430,y+83)
s.box(870,120,195,110,'Host 阶段',['参数检查与准备','launch 异步提交'],ORANGE)
s.box(870,382,195,150,'GPU 阶段',['生产者搬输入','消费者发 MMA','输出与资源回收'],TEAL)
s.text(35,816,'数据链：GMEM A/B → SMEM stage → descriptor/MMA → Acc → epilogue → GMEM D',16,MUTED)
s.save('03-gemm-execution.svg')

s=SVG(1100,650,'04  流水线：缓冲状态与工作重叠','时间长度和 stage 数仅用于说明；真实时序需要在目标 GPU 上测量。')
for x,t,ls in [(30,'可写 / Empty',['旧消费者已释放']), (300,'写入中 / Loading',['TMA 搬运并计入事务']), (570,'可读 / Ready',['事务完成且满足可见性']), (840,'使用中 / Reading',['MMA 仍可能读取 SMEM'])]:
    s.box(x,105,230,94,t,ls,TEAL if x in (30,570) else BLUE)
for a,b in [(260,295),(530,565),(800,835)]:s.arrow(a,150,b,150)
s.parts.append(f'<path d="M955 200 V238 H145 V205" stroke="{BLUE}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
s.text(370,230,'MMA 完成相关读取 → consumer_release → 允许复用',16,MUTED)
s.text(30,300,'示意时间线',19,INK,'bold')
for i,l in enumerate(['Producer / TMA','Consumer / MMA','Epilogue']):s.text(30,354+i*75,l,17)
for row,blocks in enumerate([[(0,'Load K0'),(1,'Load K1'),(2,'Load K2'),(3,'Load K3')],[(1,'MMA K0'),(2,'MMA K1'),(3,'MMA K2'),(4,'MMA K3')],[(5,'Store D')]]):
    for col,label in blocks:
        x=230+col*130;y=325+row*75
        s.box(x,y,119,54,label,[],TEAL if row==1 else BLUE)
s.arrow(230,565,1055,565,'时间推进')
s.text(30,610,'关键：index 区分槽位，phase 区分复用轮次；发出异步指令不代表访问已经结束。',17,MUTED)
s.save('04-pipeline.svg')

s=SVG(1100,620,'05  从软件表示到机器指令','C++ 模板路径；Python DSL / CuTe IR 是另行进入编译栈的入口。')
for x,t,ls in [(30,'CUTLASS / CuTe',['模板实例、layout、atom','CUDA C++ + 内联 PTX']), (395,'CUDA 编译工具链',['编译 / PTX / ptxas','fatbinary 可含 cubin/PTX']), (760,'GPU 可执行机器码',['目标 SASS','或驱动按需 JIT PTX'])]:s.box(x,105,310,115,t,ls)
s.arrow(340,163,390,163);s.arrow(705,163,755,163)
s.box(30,330,310,145,'搬运与同步',['TMA / cp.async','GMEM → SMEM','mbarrier / cluster 协作'],TEAL)
s.box(395,330,310,145,'矩阵乘加',['SM80：mma.sync','SM90：WGMMA / 寄存器 Acc','SM100：UMMA / TMEM Acc'],TEAL)
s.box(760,330,310,145,'输出与资源',['读取 accumulator','融合、转换、写回 GMEM','完成异步访问，再释放资源'],ORANGE)
s.arrow(915,220,915,320);s.arrow(760,287,190,320,dash=True);s.arrow(880,265,550,320,dash=True)
s.text(30,550,'核对三个层次：源码是否封装了指令；编译产物是否包含目标指令；运行时是否走该路径。',17,MUTED)
s.save('05-lowering-hardware.svg')

s=SVG(1100,590,'06  Hopper 与 SM100：高层组合相似，内部通路不同','代表数据通路；省略 cache、具体 store 路径及部分异步协议。')
for y,name in [(110,'Hopper / SM90a'),(325,'Blackwell / SM100')]:s.text(30,y,name,21,INK,'bold')
for y,acc,mma in [(140,'寄存器 Acc','WGMMA'),(355,'TMEM Acc','tcgen05 / UMMA')]:
    s.box(30,y,185,83,'GMEM A/B',['全局矩阵'])
    s.box(285,y,185,83,'SMEM tile',['TMA / 协作缓冲'],TEAL)
    s.box(540,y,230,83,acc,[mma],ORANGE)
    s.box(840,y,230,83,'Epilogue → D',['融合、转换、写回'])
    s.arrow(215,y+42,280,y+42,'搬运');s.arrow(470,y+42,535,y+42);s.arrow(770,y+42,835,y+42)
s.text(285,264,'输入 stage 流水线 + GMMA 完成与 stage 释放',17,MUTED)
s.text(285,479,'输入 pipeline + accumulator pipeline + TMEM 分配/交接/回收',17,MUTED)
s.text(30,548,'SM120 必须查其独立实现，不能仅凭 Blackwell 名称套用 SM100 的 TMEM/UMMA 路径。',17,MUTED)
s.save('06-generations.svg')

s=SVG(1100,625,'07  调度：工作 tile 与 K 工作量怎样分配','图示为概念对比，省略具体架构的 scheduler 与归约实现。')
s.box(30,105,320,370,'常规 data parallel',['CTA0 → tile0 的全部 K','CTA1 → tile1 的全部 K','CTA2 → tile2 的全部 K','','输出 tile 多时易于铺满设备','最后一波可能只剩少量工作'])
s.box(390,105,320,370,'Persistent',['驻留 CTA0：tile0 → tile3','驻留 CTA1：tile1 → tile4','驻留 CTA2：tile2 → tile5','','CTA 持续取得后续工作','收益取决于任务数与协作成本'],TEAL)
s.box(750,105,320,370,'Split-K / Stream-K',['CTA0 → tile0 的一段 K','CTA1 → tile0 的另一段 K','部分和 → fixup / reduction','','增加可分配工作量','付出归约、同步和 workspace'],ORANGE)
s.text(30,528,'Tile scheduler 管“下一个工作”；kernel schedule 管“角色如何协作”；pipeline 管“何时复用”。',17,MUTED)
s.text(30,567,'Persistent 不会凭空增加输出 tile；K 拆分后必须在正确的最终归约点执行 epilogue。',17,MUTED)
s.save('07-schedulers.svg')

s=SVG(1100,775,'08  新硬件后端：先正确，再组合，最后优化','这是基于源码耦合点提出的工程路线，不是现有的跨厂商插件 API。')
steps=[('M0 硬件契约','线程、指令、存储、同步、工具链、runtime'),('M1 Atom / Traits','独立 copy/MMA；逐元素映射与精度验证'),('M2 单 tile GEMM','单缓冲、边界与参考结果；最小纵切片'),('M3 Pipeline','多 stage、多轮复用；证明异步完成和可见性'),('M4 Kernel / 调度','全网格、batch、stride、输出覆盖、资源回收'),('M5 Host / Runtime','参数检查、workspace、launch、stream/event'),('M6 性能 / M7 高级能力','代表形状调优；再加入融合、分 K、grouped')]
for i,(t,d) in enumerate(steps):
    y=96+i*88;s.box(35,y,765,68,t,[d],TEAL if i in (1,2,3) else BLUE)
    if i<6:s.arrow(410,y+68,410,y+84)
s.box(840,115,225,150,'可复用的思想',['分块算法与布局代数','组件分层与测试规范','不等于直接复用 PTX'],TEAL)
s.box(840,380,225,174,'必须重新核对',['subgroup 参与语义','地址空间与内存序','异步完成与 barrier','资源限额与 launch ABI'],ORANGE)
s.text(35,751,'验收层级分开记录：能编译 → 功能正确 → 使用矩阵单元 → 具备稳定性能。',17,MUTED)
s.save('08-porting-roadmap.svg')

mdpath=OUT/'CUTLASS源码与架构指南.md'
md=mdpath.read_text(encoding='utf-8')
rows=['| 编号 | 证据与源码入口 |','|---|---|']
for r in records:
    rows.append(f'| <span id="{r["id"]}">{r["id"]}</span> | {r["title"]}：[{r["path"]}:{r["line"]}]({r["absolute"]}:{r["line"]}) |')
if '{{SOURCE_INDEX}}' in md:
    md=md.replace('{{SOURCE_INDEX}}','\n'.join(rows))
else:
    md=re.sub(r'\| 编号 \| 证据与源码入口 \|.*?(?=\n### 在线一手资料)',lambda _: '\n'.join(rows)+'\n',md,flags=re.S)
md=re.sub(r'(?<!!)\[(S\d{2})\](?!\()',r'[\1](#\1)',md)
mdpath.write_text(md,encoding='utf-8')

node=Path('C:/Users/Luke/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
marked=Path('C:/Users/Luke/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js')
render_js=OUT/'render.mjs'
render_js.write_text('import {marked} from '+json.dumps(marked.as_uri())+';\nimport fs from "node:fs";\nconst md=fs.readFileSync(process.argv[2],"utf8");\nprocess.stdout.write(marked.parse(md));',encoding='utf-8')
body=subprocess.check_output([str(node),str(render_js),str(mdpath)],text=True,encoding='utf-8')
def heading(m):
    level,title=m.group(1),m.group(2)
    plain=re.sub('<[^>]+>','',html.unescape(title))
    slug=re.sub(r'[^\w\-\u4e00-\u9fff]','',plain.lower().replace(' ','-'))
    return f'<h{level} id="{slug}">{title}</h{level}>'
body=re.sub(r'<h([1-6])>(.*?)</h\1>',heading,body)
for p in ASSETS.glob('*.svg'):
    body=re.sub(r'<img src="assets/'+re.escape(p.name)+r'" alt="([^"]*)">',lambda m:'<figure>'+p.read_text(encoding='utf-8')+'<figcaption>'+m.group(1)+'</figcaption></figure>',body)
# Standard browser file links retain a visible line number; line anchors are editor-dependent.
body=re.sub(r'href="(E:/[^\"]+):(\d+)"',lambda m:'href="file:///'+m.group(1)+'#L'+m.group(2)+'"',body)
toc=[]
for anchor,title in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',body):toc.append(f'<a href="#{anchor}">{title}</a>')
css='''
:root{--ink:#18344c;--muted:#53687d;--accent:#146f77;--line:#d7e2ea}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:26px}body{margin:0;background:#f4f7fa;color:var(--ink);font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-size:16px;line-height:1.85}aside{position:fixed;inset:0 auto 0 0;width:260px;background:#15354e;color:white;padding:32px 23px;overflow:auto}aside strong{display:block;font-size:24px;letter-spacing:1px;margin-bottom:8px}aside small{display:block;color:#b4d5df;margin-bottom:28px}aside a{display:block;color:#d6e8ed;text-decoration:none;padding:9px 0;line-height:1.5;font-size:14px;border-bottom:1px solid #305066}main{max-width:1270px;margin-left:260px;padding:48px 54px 100px;background:white;min-height:100vh}h1{font-size:38px;line-height:1.4;margin:0 0 30px}h2{font-size:28px;margin:58px 0 22px;border-bottom:2px solid var(--line);padding-bottom:12px}h3{font-size:21px;margin:33px 0 14px;color:#126e76}a{color:#166baa;text-decoration:none;overflow-wrap:anywhere}a:hover{text-decoration:underline}p{margin:14px 0}blockquote{background:#edf5f7;border-left:5px solid #127e82;margin:24px 0;padding:14px 22px;color:#34566a}table{width:100%;border-collapse:collapse;margin:22px 0;font-size:14px;line-height:1.7}th{background:#eaf2f6;color:#21495f;text-align:left}td,th{padding:11px 13px;border:1px solid var(--line);vertical-align:top}tbody tr:nth-child(even){background:#f8fafc}code{font-family:Consolas,monospace;font-size:.91em;background:#f0f4f7;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{background:#142f45;color:#e2edf5;padding:22px;border-radius:9px;overflow:auto;line-height:1.65;font-size:14px}pre code{background:none;padding:0;overflow-wrap:normal}figure{margin:30px 0;break-inside:avoid}figure svg{width:100%;height:auto;display:block}figcaption{font-size:13px;color:var(--muted);text-align:center;margin-top:8px}li{margin:7px 0}#源码证据索引~table{font-size:13px}button{border:1px solid #638394;background:transparent;color:white;padding:8px 12px;border-radius:6px;margin-top:24px;cursor:pointer}@media(max-width:1000px){aside{position:static;width:100%;padding:24px}aside a{display:inline-block;margin-right:20px;border:0}main{margin:0;padding:30px 22px}h1{font-size:30px}table{display:block;overflow-x:auto}}@media print{aside{display:none}main{margin:0;padding:0;max-width:none}body{background:white;font-size:10pt}h1{font-size:25pt}h2{font-size:18pt;break-before:page;margin-top:12pt}h3{break-after:avoid}table{font-size:8pt}pre{white-space:pre-wrap;font-size:8pt}a{color:inherit}figure{break-inside:avoid}blockquote{font-size:9pt}@page{size:A4;margin:16mm}}
'''
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CUTLASS 源码与架构指南</title><style>'+css+'</style><aside><strong>CUTLASS</strong><small>源码 · 架构 · 执行 · 移植<br>4.8.0 dev / ffa119a1</small>'+''.join(toc)+'<button onclick="window.print()">打印 / 保存为 PDF</button></aside><main>'+body+'</main></html>'
(OUT/'CUTLASS源码与架构指南.html').write_text(page,encoding='utf-8')
with zipfile.ZipFile(OUT/'CUTLASS指南与配图.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in [mdpath,OUT/'CUTLASS源码与架构指南.html',OUT/'source-index.json',*sorted(ASSETS.glob('*.svg'))]:z.write(p,p.relative_to(OUT))
print(json.dumps({'sources':len(records),'figures':len(list(ASSETS.glob('*.svg'))),'markdown_chars':len(md),'html_bytes':len(page.encode()),'output':str(OUT)},ensure_ascii=False))
