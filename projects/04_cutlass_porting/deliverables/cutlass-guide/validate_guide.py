from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
import re, json, xml.etree.ElementTree as ET, zipfile

base=Path(__file__).resolve().parent
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        if tag=='a':self.links.append(attrs.get('href',''))
        if tag=='img':self.images.append(attrs.get('src',''))
p=Parser();page=(base/'CUTLASS源码与架构指南.html').read_text(encoding='utf-8');p.feed(page)
assert len(p.ids)==len(set(p.ids)), 'Duplicate IDs'
broken=[x for x in p.links if x.startswith('#') and unquote(x[1:]) not in p.ids]
assert not broken,broken
assert not p.images, 'All HTML figures should be embedded SVG'
assert page.count('<svg ')==8
md=(base/'CUTLASS源码与架构指南.md').read_text(encoding='utf-8')
assert '{{' not in md
for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md):assert (base/image).is_file(),image
records=json.loads((base/'source-index.json').read_text(encoding='utf-8'))
for r in records:
    src=Path(r['absolute']);assert src.is_file()
    assert 1<=r['line']<=len(src.read_text(encoding='utf-8').splitlines())
for svg in (base/'assets').glob('*.svg'):ET.parse(svg)
with zipfile.ZipFile(base/'CUTLASS指南与配图.zip') as z:assert z.testzip() is None
result={'source_anchors':len(records),'svg_figures':8,'internal_links':sum(x.startswith('#') for x in p.links),'broken_links':len(broken),'embedded_html_figures':8,'xml_parse':'pass','zip_integrity':'pass','browser_layout_check':'not performed; file URL blocked','gpu_execution':'not performed for SM90/SM100'}
(base/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
