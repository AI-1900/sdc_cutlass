from pathlib import Path
from html import escape
import json

OUT=Path(__file__).parent
OUT.mkdir(parents=True,exist_ok=True)
INK='#183544'; SUB='#526977'; BG='#f5f8fa'; BORDER='#c6d3da'
COLORS={'calc':('#e5f1ff','#215b94'),'data':('#e0f4ed','#17664f'),'impl':('#fff0df','#89531e')}
class Figure:
 def __init__(self,name,title,subtitle,h=800):
  self.name=name;self.h=h;self.p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{h}" viewBox="0 0 1120 {h}" role="img" aria-label="{escape(title)}"><defs><marker id="arr-{name}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="{SUB}"/></marker></defs><rect width="1120" height="{h}" fill="{BG}"/>'];self.text(32,43,title,28);self.text(32,77,subtitle,17,SUB)
 def text(self,x,y,s,size=19,color=INK,weight=400):self.p.append(f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')
 def rect(self,x,y,w,h,fill='white',stroke=BORDER,r=10):self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
 def box(self,x,y,w,h,title,lines=(),kind=None):
  fill,col=COLORS[kind] if kind else ('white',INK);self.rect(x,y,w,h,fill);self.text(x+16,y+31,title,21,col,600)
  for i,line in enumerate(lines):self.text(x+16,y+62+i*27,line,17)
 def arrow(self,x,y,xx,yy):self.p.append(f'<path d="M{x} {y}L{xx} {yy}" stroke="{SUB}" stroke-width="1.7" fill="none" marker-end="url(#arr-{self.name})"/>')
 def path(self,d):self.p.append(f'<path d="{d}" stroke="{BORDER}" stroke-width="2" fill="none"/>')
 def footer(self,s):self.text(32,self.h-23,s,15,SUB)
 def save(self):self.p.append('</svg>');(OUT/(self.name+'.svg')).write_text(''.join(self.p),encoding='utf-8')

f=Figure('01-three-lines','一张图：三条线如何汇合成一个专用 GEMM kernel','计算拆分决定任务；CuTe 决定空间分工；类型与策略决定实现。',840)
f.box(32,108,326,112,'计算如何拆分',['工作 tile → K 主循环','搬运 → MMA → 输出处理'],'calc')
f.box(397,108,326,112,'数据如何分工',['矩阵坐标 ↔ 线程 / 值','数据布局 → 操作数位置'],'data')
f.box(762,108,326,112,'实现如何确定',['架构 / 类型 / Tile / 策略','Builder / 模板特化'],'impl')
for x in (195,560,925):f.arrow(x,221,x,260)
f.box(32,265,326,112,'算法部件',['Mainloop + Epilogue','Kernel + Scheduler'],'calc')
f.box(397,265,326,112,'CuTe 空间描述',['Layout + Tensor','TiledCopy / TiledMMA'],'data')
f.box(762,265,326,112,'具体实现约束',['Atom / Traits / Pipeline','对齐、布局与同步约定'],'impl')
f.arrow(195,378,420,433);f.arrow(560,378,560,433);f.arrow(925,378,700,433)
f.box(250,439,620,100,'编译期组合 → 具体 C++ Kernel 类型',['选定 Mainloop / Epilogue / 调度 / 资源组织'])
f.arrow(560,540,560,569)
f.box(250,575,620,95,'CUDA 工具链 → 目标机器码',['特化、内联、索引化简 → PTX → SASS'])
f.arrow(560,671,560,700)
f.box(250,706,620,83,'运行：传入矩阵地址、尺寸和标量',['执行分块搬运与 MMA 流水线，写回 D'])
f.footer('C++ 3.x 风格 API；本地提交 ffa119a1。组合受兼容约束；图中箭头表示依赖或生成关系。');f.save()

f=Figure('02-computation','计算拆分：盯住一个输出 tile，沿 K 一段段累加','示意：M=N=256，K=192；工作 tile 为 128×128，K 分块为 64。',865)
def grid(fig,x,y,rows,cols,cw,ch,hot,label,kind='calc'):
 fill,col=COLORS[kind];fig.text(x,y-15,label,20,col,600)
 for r in range(rows):
  for c in range(cols):
   fig.rect(x+c*cw,y+r*ch,cw-4,ch-4,fill if (r,c) in hot else 'white',BORDER,3)
   fig.text(x+c*cw+10,y+r*ch+ch/2+5,f'{r},{c}',16)
grid(f,48,150,2,3,78,65,{(0,1)},'A：选中 M 块 × K₁')
f.text(305,224,'×',30)
grid(f,350,150,3,2,85,48,{(1,0)},'B：K₁ × 选中 N 块')
f.text(548,224,'→',30)
grid(f,610,150,2,2,85,65,{(0,0)},'输出：固定 tile (0,0)')
f.box(830,150,250,131,'当前 K 分块 1',['A：128×64','B：64×128'],'calc')
f.text(32,345,'Mainloop：保持输出 tile 不变，只推进 K 分块',23,INK,600)
for i,(title,lines) in enumerate([('K₀',['Acc += A₀ × B₀']),('K₁',['Acc += A₁ × B₁']),('K₂',['Acc += A₂ × B₂'])]):
 f.box(32+i*365,370,326,90,title,lines,'calc')
 if i<2:f.arrow(359+i*365,415,390+i*365,415)
f.text(32,510,'每个 K 分块内部：空间分工 + 时间协作',23,INK,600)
for i,(t,l) in enumerate([('① 搬运',['GMEM → SMEM']),('② 就绪',['等待当前数据']),('③ MMA',['基本指令铺满 tile']),('④ 复用',['消费完成后释放'])]):
 f.box(32+i*274,535,246,95,t,l,'calc')
 if i<3:f.arrow(279+i*274,580,300+i*274,580)
f.arrow(560,632,560,669)
f.box(210,675,700,100,'Epilogue：全部 K 分块完成后处理输出',['D_tile = α · Acc + β · C_tile；可选 bias / 激活 / 类型转换'])
f.footer('网格每格代表一个分块。MMA 的数据路径、累加位置与完成协议取决于架构；此图不表示真实耗时。');f.save()

f=Figure('03-cute-mapping','数据分工：把“谁负责”与“数据在哪里”连接起来','4×4 教学 tile，4 个线程，每线程 4 个元素；这不是某条真实 MMA 的线程布局。',855)
f.box(32,110,310,95,'线程 / 值坐标',['(thread t, value v)'],'data');f.arrow(344,157,385,157)
f.box(395,110,310,95,'逻辑矩阵坐标',['(m, n)'],'data');f.arrow(707,157,748,157)
f.box(758,110,330,95,'存储索引 / 偏移',['行主序示例：offset = 4m + n'],'data')
f.text(40,254,'线程所有权：每格一个元素',21,INK,600)
for m in range(4):
 for n in range(4):
  t=m%2+2*(n%2);v=m//2+2*(n//2);x=40+n*94;y=278+m*66
  f.rect(x,y,89,61,COLORS['data'][0] if t==1 else 'white',BORDER,5);f.text(x+13,y+36,f'T{t} / v{v}',17)
f.text(40,570,'高亮 T1：四个元素分布在不同坐标',17,COLORS['data'][1])
f.text(500,254,'同一 tile 的行主序内存偏移',21,INK,600)
for m in range(4):
 for n in range(4):
  x=500+n*94;y=278+m*66;f.rect(x,y,89,61,COLORS['data'][0] if m%2==1 and n%2==0 else 'white',BORDER,5);f.text(x+32,y+36,str(4*m+n),20)
f.box(915,280,173,260,'例：T1 / v2',['m = 1','n = 2','offset = 6','','指针 + 偏移','定位该元素'],'data')
f.box(32,615,510,140,'Layout：坐标到索引的映射',['Shape / Stride / 嵌套布局 / 组合变换','Tensor：数据引擎 + Layout','tile / partition：构造对应数据视图'],'data')
f.box(578,615,510,140,'Tiled 操作：匹配指令参与方式',['get_slice(t)：取得线程切片','partition_A/B/C：取得操作数视图','copy / gemm：执行搬运或计算'],'data')
f.footer('视图变换通常不搬运数据。实际 Copy/MMA 映射须满足 Atom Traits 的参与线程与操作数约定。');f.save()

f=Figure('04-composition-tree','架构树：可组合的部件怎样各司其职','依赖关系和职责分层；不是每层都对应一个固定硬件层级。',890)
f.box(32,111,550,80,'Device · GemmUniversalAdapter',['检查 / workspace / 参数转换 / launch'])
f.box(75,225,507,80,'Kernel · GemmUniversal',['调度工作 tile / 组织线程角色和资源'])
f.path('M52 191V265H75')
f.box(118,340,464,80,'CollectiveMainloop',['K 循环 / 搬运与 MMA 协作'],'calc');f.path('M95 305V380H118')
f.box(165,455,417,80,'TiledCopy / TiledMMA',['空间铺开 / 线程与元素划分'],'data');f.path('M138 420V495H165')
f.box(212,570,370,80,'Atom + Traits',['基本操作 / 线程与操作数约定'],'impl');f.path('M185 535V610H212')
f.box(655,225,433,80,'Tile Scheduler',['决定下一个工作 tile'])
f.box(655,340,433,80,'CollectiveEpilogue',['输出变换 / 融合 / 存储'],'calc')
f.box(655,455,433,80,'Pipeline',['数据就绪与消费完成 / 缓冲复用'])
f.path('M582 265H625V380H655');f.path('M625 265H655');f.path('M582 380H610V495H655')
f.arrow(397,652,397,690)
f.box(212,697,876,100,'Architecture 封装 → 目标硬件操作',['Copy / MMA / descriptor / barrier / 架构特定存储与同步'])
f.footer('Collective 主要组织时间与协作；Tiled 主要表达空间分工；Atom 不是 atomicAdd 的内存原子性。');f.save()

f=Figure('05-specialization','实现确定：类型与策略，怎样变成 GPU 指令','普通 C++ 模板实例化路径；PTX 可由工具链离线编译，或在部署中经驱动 JIT。',900)
f.box(32,110,510,140,'编译期配置',['Arch / 元素类型 / Layout 类型与静态部分','Tile / Cluster / Stage / Schedule','融合操作类型'],'impl')
f.box(578,110,510,140,'运行时输入',['M,N,K,L / 矩阵指针','动态 stride / α,β','workspace / stream'])
f.arrow(287,252,287,285)
f.box(32,290,510,106,'① Builder / 模板特化',['选择支持的 Mainloop、Epilogue','确定 Atom、布局、Pipeline 与策略'],'impl')
f.arrow(287,397,287,431)
f.box(32,436,510,106,'② Kernel 类型组合',['Mainloop + Epilogue + Scheduler','→ GemmUniversal → Device Adapter'],'impl')
f.arrow(287,543,287,577)
f.box(32,582,510,106,'③ CUDA 编译优化',['内联 / 常量传播 / 索引化简','架构操作封装发出相应 PTX'],'impl')
f.arrow(544,635,577,635)
f.box(582,582,506,106,'④ PTX → 目标 SASS',['目标代码生成 / 寄存器分配 / 调度','形成可执行 kernel'],'impl')
f.arrow(835,690,835,733)
f.box(582,738,506,100,'⑤ 启动并执行',['传入运行时参数 → GPU 处理各工作 tile'])
f.path('M1088 180H1103V788H1088');f.text(640,342,'Auto：按 Builder 规则选择配置',20);f.text(640,377,'性能：仍需匹配 GPU 与问题形状',20);f.text(640,412,'并通过实际测量验证',20)
f.footer('模板固定实现策略；动态尺寸不必固定。优化效果与资源占用须检查生成代码并在目标 GPU 上验证。');f.save()

cards=[('01 · Tile','为什么分块？','让输入在片上被多次复用，','同时控制单个任务的资源需求。','计算拆分'),('02 · Mainloop','沿哪个维度推进？','保持输出 tile 不变，','沿 K 分块持续累加。','计算拆分'),('03 · Epilogue','K 循环结束后做什么？','缩放、可选融合、类型转换，','再将结果存储到 D。','计算拆分'),('04 · Layout','坐标怎样找到元素？','Layout 将坐标映射为索引；','Tensor 再结合数据引擎。','数据分工'),('05 · Thread / Value','每个线程负责几个元素？','可负责多个。用 (thread,value)','描述它们对应的逻辑位置。','数据分工'),('06 · Partition','切分视图会搬运数据吗？','通常不会；copy / gemm 等操作','才执行相应搬运或计算。','数据分工'),('07 · Atom / Traits','为什么线程布局不能随意选？','基本操作规定参与线程、','形状和操作数接口约定。','实现确定'),('08 · Template','抽象如何变具体？','类型和策略选择实现；','编译器内联并化简静态映射。','实现确定'),('09 · Performance','Auto 就能保证最快吗？','不能。规则选择后，还要验证','资源占用、正确性与实际性能。','实现确定')]
f=Figure('06-review-cards','复习卡片：用九个问题串起三条线','先看问题，尝试回答，再对照卡片中的关键句。',960)
for i,(label,q,a,b,tag) in enumerate(cards):
 x=32+(i%3)*365;y=110+(i//3)*265;kind=('calc','data','impl')[i//3];fill,col=COLORS[kind]
 f.rect(x,y,326,235,'white');f.rect(x,y,326,46,fill,fill,8);f.text(x+16,y+30,label,20,col,600);f.text(x+16,y+83,q,18,INK,600);f.text(x+16,y+127,a,17);f.text(x+16,y+157,b,17);f.text(x+16,y+209,tag,15,col)
f.footer('记忆主线：任务怎样拆 → 数据由谁处理 → 实现怎样选定 → 编译成代码 → 传参执行。');f.save()

manifest={'source_commit':'ffa119a1','figures':sorted(p.name for p in OUT.glob('*.svg')),'mapping':'m=t%2+2*(v%2); n=t//2+2*(v//2)','scope':'C++ 3.x API conceptual diagrams; no GPU performance measurements'}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
