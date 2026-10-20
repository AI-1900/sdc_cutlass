$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$outDir = $PSScriptRoot
$nodes = @(
  @{id='epi';x=55;y=115;w=440;h=115;text="Epilogue Builder`nFP32 线性组合 → CollectiveEpilogue`nSharedStorage 用于主循环 SMEM 预算";color='#E8F0FE'},
  @{id='main';x=580;y=115;w=440;h=115;text="Mainloop Builder`nFP16 A/B + FP32 Acc + Auto`n64×128×64，cluster 1×4×1";color='#E8F0FE'},
  @{id='kernel';x=1105;y=115;w=440;h=115;text="GemmUniversal → Adapter`n问题形状 (M,N,K,L)`nTestAll 提供运行时尺寸与参数";color='#E8F0FE'},
  @{id='gmem';x=55;y=390;w=300;h=105;text="A / B 全局内存`nFP16，均为 K 连续";color='#E6F4EA'},
  @{id='smem';x=450;y=390;w=300;h=105;text="TMA → SMEM stage`nA 可向四个 CTA multicast";color='#E6F4EA'},
  @{id='mma';x=845;y=390;w=300;h=105;text="UMMA / tcgen05.mma`nAtom：64×128×16`n每个完整 TileK 包含 4 次";color='#E6F4EA'},
  @{id='tmem';x=1240;y=390;w=305;h=105;text="TMEM 累加器`nFP32，cta_group::1";color='#FFF3D6'},
  @{id='epirun';x=1240;y=590;w=305;h=105;text="TMEM load → 寄存器`nalpha × Acc`n→ epilogue SMEM";color='#FFF3D6'},
  @{id='out';x=845;y=590;w=300;h=105;text="TMA store → D`nFP32 行主序输出";color='#FFF3D6'},
  @{id='check';x=450;y=590;w=300;h=105;text="同步、参考 GEMM、比较`nTestAll → bool`nEXPECT_TRUE(pass)";color='#F0E8FA'},
  @{id='void';x=55;y=590;w=300;h=105;text="C = void：无 C 输入`n默认 alpha=1、beta=0`nD = A × B";color='#F0E8FA'},
  @{id='sharedA';x=535;y=885;w=530;h=95;text="同一 A tile [64,64]`n沿 N 方向 multicast 给 4 个 CTA";color='#E8F0FE'},
  @{id='c0';x=55;y=1080;w=335;h=115;text="CTA 0：1SM MMA`nB：N 偏移 [0,128)`nD：64×128";color='#E6F4EA'},
  @{id='c1';x=440;y=1080;w=335;h=115;text="CTA 1：1SM MMA`nB：N 偏移 [128,256)`nD：64×128";color='#E6F4EA'},
  @{id='c2';x=825;y=1080;w=335;h=115;text="CTA 2：1SM MMA`nB：N 偏移 [256,384)`nD：64×128";color='#E6F4EA'},
  @{id='c3';x=1210;y=1080;w=335;h=115;text="CTA 3：1SM MMA`nB：N 偏移 [384,512)`nD：64×128";color='#E6F4EA'}
)
$labels = @(
  @{id='title';x=55;y=20;w=1490;h=60;text='SM100 GEMM：64×128×64 / cluster 1×4×1 / 1SM';size=30},
  @{id='l1';x=55;y=78;w=1490;h=30;text='① 编译期组合：Builder 生成类型，using 不启动 GPU';size=22},
  @{id='l2';x=55;y=315;w=1490;h=45;text='② 运行时数据链：搬运、MMA 与输出由流水线重叠组织';size=25},
  @{id='l3';x=55;y=790;w=1490;h=45;text='③ 一个逻辑 cluster：4 个 CTA 沿 N 排列，完整输出覆盖 64×512';size=25},
  @{id='note';x=55;y=1260;w=1490;h=60;text='静态源码示意；未重新编译或运行 GPU。1SM 是 MMA 协作粒度，cluster 仍有 4 个 CTA。';size=20}
)
$edges = @(
  @{s='epi';t='main';sx=1;sy=0.5;tx=0;ty=0.5},
  @{s='main';t='kernel';sx=1;sy=0.5;tx=0;ty=0.5},
  @{s='gmem';t='smem';sx=1;sy=0.5;tx=0;ty=0.5},
  @{s='smem';t='mma';sx=1;sy=0.5;tx=0;ty=0.5},
  @{s='mma';t='tmem';sx=1;sy=0.5;tx=0;ty=0.5},
  @{s='tmem';t='epirun';sx=0.5;sy=1;tx=0.5;ty=0},
  @{s='epirun';t='out';sx=0;sy=0.5;tx=1;ty=0.5},
  @{s='out';t='check';sx=0;sy=0.5;tx=1;ty=0.5},
  @{s='sharedA';t='c0';sx=0.15;sy=1;tx=0.5;ty=0},
  @{s='sharedA';t='c1';sx=0.38;sy=1;tx=0.5;ty=0},
  @{s='sharedA';t='c2';sx=0.62;sy=1;tx=0.5;ty=0},
  @{s='sharedA';t='c3';sx=0.85;sy=1;tx=0.5;ty=0}
)
$doc = New-Object System.Xml.XmlDocument
$mxfile = $doc.CreateElement('mxfile'); $mxfile.SetAttribute('host','app.diagrams.net'); [void]$doc.AppendChild($mxfile)
$diagram = $doc.CreateElement('diagram'); $diagram.SetAttribute('id','sm100-flow'); $diagram.SetAttribute('name','SM100 GEMM'); [void]$mxfile.AppendChild($diagram)
$model = $doc.CreateElement('mxGraphModel'); $model.SetAttribute('pageWidth','1600'); $model.SetAttribute('pageHeight','1350'); $model.SetAttribute('page','1'); [void]$diagram.AppendChild($model)
$root = $doc.CreateElement('root'); [void]$model.AppendChild($root)
foreach($v in @(@{id='0'},@{id='1';parent='0'})) { $cell=$doc.CreateElement('mxCell'); foreach($k in $v.Keys){$cell.SetAttribute($k,[string]$v[$k])}; [void]$root.AppendChild($cell) }
$bmp = New-Object System.Drawing.Bitmap 1600,1350
$g = [System.Drawing.Graphics]::FromImage($bmp); $g.Clear([System.Drawing.Color]::White)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$pen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml('#52647A')),2
$pen.CustomEndCap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap 5,6
$lookup = @{}; foreach($n in $nodes){$lookup[$n.id]=$n}
$ei=0
foreach($e in $edges){
  $s=$lookup[$e.s];$t=$lookup[$e.t]
  $g.DrawLine($pen,[single]($s.x+$s.w*$e.sx),[single]($s.y+$s.h*$e.sy),[single]($t.x+$t.w*$e.tx),[single]($t.y+$t.h*$e.ty))
  $c=$doc.CreateElement('mxCell');$c.SetAttribute('id',"edge$ei");$ei++
  $c.SetAttribute('edge','1');$c.SetAttribute('parent','1');$c.SetAttribute('source',$e.s);$c.SetAttribute('target',$e.t)
  $c.SetAttribute('style',"edgeStyle=none;endArrow=block;strokeColor=#52647A;strokeWidth=2;exitX=$($e.sx);exitY=$($e.sy);entryX=$($e.tx);entryY=$($e.ty);")
  $geo=$doc.CreateElement('mxGeometry');$geo.SetAttribute('relative','1');$geo.SetAttribute('as','geometry');[void]$c.AppendChild($geo);[void]$root.AppendChild($c)
}
$border=New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml('#9AABBE')),1.5
$ink=New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml('#17283D'))
$sf=New-Object System.Drawing.StringFormat; $sf.Alignment='Center';$sf.LineAlignment='Center'
foreach($n in @($nodes)+@($labels)){
  $isBox=$n.ContainsKey('color'); $size=if($isBox){20}else{$n.size}
  $font=New-Object System.Drawing.Font 'Microsoft YaHei',$size,([System.Drawing.FontStyle]::Regular),([System.Drawing.GraphicsUnit]::Pixel)
  $rect=New-Object System.Drawing.RectangleF ([single]$n.x),([single]$n.y),([single]$n.w),([single]$n.h)
  if($isBox){$brush=New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml($n.color));$g.FillRectangle($brush,$rect);$g.DrawRectangle($border,[single]$n.x,[single]$n.y,[single]$n.w,[single]$n.h);$brush.Dispose()}
  $g.DrawString($n.text,$font,$ink,$rect,$sf);$font.Dispose()
  $c=$doc.CreateElement('mxCell');$c.SetAttribute('id',$n.id);$c.SetAttribute('parent','1');$c.SetAttribute('vertex','1');$c.SetAttribute('value',$n.text)
  $style="whiteSpace=wrap;html=0;fontFamily=Microsoft YaHei;fontSize=$size;fontColor=#17283D;align=center;verticalAlign=middle;"
  if($isBox){$style+="rounded=0;fillColor=$($n.color);strokeColor=#9AABBE;"}else{$style+='text;strokeColor=none;fillColor=none;'}
  $c.SetAttribute('style',$style);$geo=$doc.CreateElement('mxGeometry');$geo.SetAttribute('as','geometry')
  foreach($k in @('x','y','w','h')){$key=if($k -eq 'w'){'width'}elseif($k -eq 'h'){'height'}else{$k};$geo.SetAttribute($key,[string]$n[$k])}
  [void]$c.AppendChild($geo);[void]$root.AppendChild($c)
}
$bmp.Save((Join-Path $outDir 'flow.png'),[System.Drawing.Imaging.ImageFormat]::Png)
$settings=New-Object System.Xml.XmlWriterSettings;$settings.Indent=$true;$settings.Encoding=New-Object System.Text.UTF8Encoding $false
$writer=[System.Xml.XmlWriter]::Create((Join-Path $outDir 'flow.drawio'),$settings);$doc.Save($writer);$writer.Dispose()
$g.Dispose();$bmp.Dispose();$pen.Dispose();$border.Dispose();$ink.Dispose();$sf.Dispose()
Write-Output 'Created flow.png and editable flow.drawio from the same node/edge definitions.'
