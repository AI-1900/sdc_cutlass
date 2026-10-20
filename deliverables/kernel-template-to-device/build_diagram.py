from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, math, hashlib, html
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
W,H=2040,2160
S=1.5
FONT='C:/Windows/Fonts/msyh.ttc'
BOLD='C:/Windows/Fonts/msyhbd.ttc'
nodes=[];edges=[]

def box(id,x,y,w,h,fill,stroke,parent=None,radius=20):
    nodes.append(dict(id=id,x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,parent=parent,radius=radius,kind='box'))
def text(id,x,y,w,h,content,size=24,bold=False,color='#19344d',parent=None):
    nodes.append(dict(id=id,x=x,y=y,w=w,h=h,text=content,size=size,bold=bold,color=color,parent=parent,kind='text'))
def card(id,x,y,w,h,title,body,color='blue',size=23,parent=None):
    fill,stroke={'blue':('#e6efff','#487ab8'),'purple':('#efe9fb','#8b69b5'),'green':('#e4f5ec','#3e9772'),'gold':('#fff2d8','#be8a30'),'gray':('#edf2f7','#77899a')}[color]
    box(id,x,y,w,h,fill,stroke,parent)
    text(id+'title',x+20,y+14,w-40,40,title,25,True,parent=id)
    text(id+'body',x+20,y+65,w-40,h-75,body,size,parent=id)
def edge(id,a,b,pts,color='#58728c',dashed=False):edges.append(dict(id=id,a=a,b=b,pts=pts,color=color,dashed=dashed))

text('title',45,25,1950,65,'从测例模板，到一个真正启动的 GPU kernel',42,True)
text('subtitle',45,100,1950,45,'先组合类型，再编译生成代码，最后启动入口；两个 Collective 在同一个 kernel 内协作。',24)

box('phase1',30,165,1980,580,'#f3f7ff','#c6d6ee')
text('p1title',60,185,1900,50,'① 编译期：测例用模板组合出 GemmKernel 类型',30,True,parent='phase1')
card('config',70,285,420,215,'测例中的编译期配置','元素类型 / 矩阵布局 / 对齐\ntile / cluster / schedule\n通过 CollectiveBuilder 选择实现','blue',23,'phase1')
card('mltype',560,255,500,150,'CollectiveMainloop 类型','输入搬运 + MMA + 累加器布局','blue',23,'phase1')
card('eptype',560,455,500,150,'CollectiveEpilogue 类型','输出融合 + 类型转换 + 存储','purple',23,'phase1')
card('kerneltype',1140,285,820,275,'组合得到 GemmKernel 类型','using GemmKernel = GemmUniversal<\n    Shape<int,int,int,int>,\n    CollectiveMainloop,\n    CollectiveEpilogue>;','green',24,'phase1')
box('adaptertype',1140,615,820,95,'#e4f5ec','#3e9772','phase1')
text('adaptertypetext',1160,630,780,65,'GemmUniversalAdapter<GemmKernel>\n包装成主机可使用的 Gemm handle 类型',23,parent='adaptertype')
text('typehint',70,645,1010,65,'这里的 using / Builder 产生类型关系。\n此时没有执行矩阵乘法，也没有把两个 Collective 分别启动。',22,parent='phase1')
edge('config_ml','config','mltype',[(490,335),(525,335),(525,330),(560,330)])
edge('config_ep','config','eptype',[(490,445),(525,445),(525,530),(560,530)])
edge('ml_kernel','mltype','kerneltype',[(1060,330),(1100,330),(1100,375),(1140,375)])
edge('ep_kernel','eptype','kerneltype',[(1060,530),(1100,530),(1100,480),(1140,480)])
edge('kernel_adapter','kerneltype','adaptertype',[(1550,560),(1550,615)])

box('phase2',30,800,1980,490,'#fffbf1','#e6d2a8')
text('p2title',60,820,1600,45,'② 编译器：根据模板实例化生成主机代码与设备代码',30,True,parent='phase2')
card('compiler',380,900,1280,155,'编译输入：测例 .cu + 多个头文件中的模板实现','模板实例化 / 类型检查 → host/device 编译与优化\n实例化入口与编排逻辑；下层方法可被内联，源码文件边界不等于机器码函数边界。','gold',23,'phase2')
card('hostcode',70,1120,760,130,'编译产物：主机代码','TestAll / Testbed / Adapter 参数处理与启动代码','gold',22,'phase2')
card('devicecode',1000,1120,960,130,'编译产物：GPU 设备代码','device_kernel<GemmKernel> 的入口及其计算实现','gold',24,'phase2')
edge('types_compile','adaptertype','compiler',[(1960,665),(1985,665),(1985,977),(1660,977)],'#b58a40')
edge('compile_host','compiler','hostcode',[(750,1055),(750,1085),(450,1085),(450,1120)],'#b58a40')
edge('compile_device','compiler','devicecode',[(1300,1055),(1300,1085),(1480,1085),(1480,1120)],'#b58a40')

box('phase3',30,1350,1980,690,'#f3faf6','#bddbc9')
text('p3title',60,1370,690,50,'③ 运行时：主机启动，GPU 内部协作',30,True,parent='phase3')
card('hostrun',70,1480,650,235,'Host：测试与 Device Adapter','TestAll<Gemm>() → Testbed::run\ninitialize(Arguments) → Params\nGemmUniversalAdapter<GemmKernel>::run\n提交 launch：目标为具体的入口函数','green',23,'phase3')
box('gpu',800,1450,1160,545,'#e7f4ed','#3e9772','phase3')
text('gputitle',825,1467,970,45,'GPU：这次启动的一个 kernel 内部',26,True,parent='gpu')
card('entry',830,1525,1100,130,'真正的 __global__ 入口','device_kernel<GemmKernel>(Params) → GemmKernel op; op(params, smem);','green',23,'gpu')
box('body',830,1690,1100,265,'#f5f8fc','#6f879f','gpu')
text('bodytitle',850,1705,1060,65,'GemmKernel::operator()(Params, smem)\n统一组织：共享存储、warp 分工、scheduler 与同步',23,True,parent='body')
card('mlrun',860,1800,430,125,'CollectiveMainloop','A/B → SMEM → MMA → TMEM','blue',21,'body')
card('eprun',1450,1800,450,125,'CollectiveEpilogue','TMEM → 融合 → 写回 GMEM D','purple',21,'body')
box('handoff',1303,1800,134,90,'#f5f8fc','#f5f8fc','body',0)
text('handofftext',1305,1805,130,80,'同一 tile\nTMEM 视图\npipeline 协议',18,parent='handoff')
edge('loadhost','hostcode','hostrun',[(780,1250),(780,1530),(720,1530)],'#8b9caa',True)
edge('loaddevice','devicecode','entry',[(1930,1250),(1980,1250),(1980,1505),(1880,1505),(1880,1525)],'#8b9caa',True)
edge('launch','hostrun','entry',[(720,1600),(830,1600)],'#298465')
edge('entry_body','entry','body',[(1380,1655),(1380,1690)],'#298465')
edge('collective_handoff','mlrun','eprun',[(1290,1905),(1450,1905)],'#477c9f')
text('gpufoot',850,1960,1060,30,'Mainloop 与 Epilogue 是内部逻辑组件；它们通过数据与同步协议协作。',20,parent='gpu')
text('footer',50,2070,1940,65,'源码分布在多个文件；运行时启动的是具体的函数实例。图表示逻辑组织，编译器可内联与重排实现。\n依据本地 SM100 源码绘制；本轮未进行编译、设备运行或机器码验证。',22)

byid={n['id']:n for n in nodes}
im=Image.new('RGB',(int(W*S),int(H*S)),'#fcfdff');d=ImageDraw.Draw(im)
xy=lambda p:(round(p[0]*S),round(p[1]*S))
for n in nodes:
    if n['kind']=='box':
        x,y,w,h=n['x'],n['y'],n['w'],n['h']
        d.rounded_rectangle((*xy((x,y)),*xy((x+w,y+h))),radius=round(n['radius']*S),fill=n['fill'],outline=n['stroke'],width=2)
for e in edges:
    pts=e['pts'];scaled=[xy(p) for p in pts]
    if e['dashed']:
        for (x1,y1),(x2,y2) in zip(scaled,scaled[1:]):
            length=math.hypot(x2-x1,y2-y1)
            for start in range(0,math.ceil(length),18):
                end=min(start+10,length)
                d.line((x1+(x2-x1)*start/length,y1+(y2-y1)*start/length,x1+(x2-x1)*end/length,y1+(y2-y1)*end/length),fill=e['color'],width=3)
    else:d.line(scaled,fill=e['color'],width=4)
    p,q=pts[-2:];ang=math.atan2(q[1]-p[1],q[0]-p[0]);d.polygon([xy(q),xy((q[0]-13*math.cos(ang-.43),q[1]-13*math.sin(ang-.43))),xy((q[0]-13*math.cos(ang+.43),q[1]-13*math.sin(ang+.43)))],fill=e['color'])
for n in nodes:
    if n['kind']!='text':continue
    font=ImageFont.truetype(BOLD if n['bold'] else FONT,round(n['size']*S))
    lines=n['text'].split('\n');lh=n['size']*1.45
    assert len(lines)*lh <= n['h']+3,('height',n['id'])
    for i,line in enumerate(lines):
        assert d.textlength(line,font=font) <= n['w']*S,('width',n['id'],line)
        d.text(xy((n['x'],n['y']+i*lh)),line,font=font,fill=n['color'])
im.save(OUT/'kernel-template-to-device.png')

mx=ET.Element('mxfile',{'host':'app.diagrams.net'})
dg=ET.SubElement(mx,'diagram',{'id':'kernel-template-to-device','name':'模板组合与 kernel 生成'})
model=ET.SubElement(dg,'mxGraphModel',{'grid':'1','page':'1','pageWidth':str(W),'pageHeight':str(H)})
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
for n in nodes:
    parent=n['parent'] or '1';x,y=n['x'],n['y']
    if parent!='1':x-=byid[parent]['x'];y-=byid[parent]['y']
    if n['kind']=='box':style=f'rounded=1;arcSize=8;whiteSpace=wrap;html=0;fillColor={n["fill"]};strokeColor={n["stroke"]};strokeWidth=1;'
    else:style=f'text;html=0;whiteSpace=wrap;align=left;verticalAlign=top;spacing=0;fontFamily=Microsoft YaHei;fontSize={n["size"]};fontColor={n["color"]};fontStyle={1 if n["bold"] else 0};fillColor=none;strokeColor=none;'
    c=ET.SubElement(root,'mxCell',{'id':n['id'],'value':n.get('text',''),'style':style,'vertex':'1','parent':parent})
    ET.SubElement(c,'mxGeometry',{'x':str(x),'y':str(y),'width':str(n['w']),'height':str(n['h']),'as':'geometry'})
for e in edges:
    a,b=byid[e['a']],byid[e['b']];p,q=e['pts'][0],e['pts'][-1]
    ax=(p[0]-a['x'])/a['w'];ay=(p[1]-a['y'])/a['h'];bx=(q[0]-b['x'])/b['w'];by=(q[1]-b['y'])/b['h']
    style=f'html=0;endArrow=block;strokeWidth=2;strokeColor={e["color"]};dashed={1 if e["dashed"] else 0};exitX={ax};exitY={ay};entryX={bx};entryY={by};exitPerimeter=0;entryPerimeter=0;'
    c=ET.SubElement(root,'mxCell',{'id':e['id'],'style':style,'edge':'1','parent':'1','source':e['a'],'target':e['b']})
    g=ET.SubElement(c,'mxGeometry',{'relative':'1','as':'geometry'});ar=ET.SubElement(g,'Array',{'as':'points'})
    for p in e['pts'][1:-1]:ET.SubElement(ar,'mxPoint',{'x':str(p[0]),'y':str(p[1])})
ET.indent(mx);ET.ElementTree(mx).write(OUT/'kernel-template-to-device.drawio',encoding='utf-8',xml_declaration=True)
ET.parse(OUT/'kernel-template-to-device.drawio')
(OUT/'diagram-data.json').write_text(json.dumps({'nodes':nodes,'edges':edges},ensure_ascii=False,indent=2),encoding='utf-8')
notes='''# 测例模板如何生成并启动 kernel

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
'''
(OUT/'explanation.md').write_text(notes,encoding='utf-8')
print(json.dumps({'png':str(OUT/'kernel-template-to-device.png'),'drawio':str(OUT/'kernel-template-to-device.drawio'),'size':im.size,'nodes':len(nodes),'edges':len(edges)},ensure_ascii=False))
