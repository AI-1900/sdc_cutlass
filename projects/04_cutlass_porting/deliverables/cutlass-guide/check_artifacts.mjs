import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire('C:/Users/Luke/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json');
const sharp=require('sharp');
const base=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const qa=path.join(base,'qa');fs.mkdirSync(qa,{recursive:true});
const files=fs.readdirSync(path.join(base,'assets')).filter(x=>x.endsWith('.svg')).sort();
const tiles=[];
for(let i=0;i<files.length;i++){
 const inp=path.join(base,'assets',files[i]);
 const png=path.join(qa,files[i].replace('.svg','.png'));
 await sharp(inp).png().toFile(png);
 const thumb=await sharp(png).resize({width:550,height:430,fit:'contain',background:'#ffffff'}).png().toBuffer();
 tiles.push({input:thumb,left:(i%2)*550,top:Math.floor(i/2)*430});
}
await sharp({create:{width:1100,height:Math.ceil(files.length/2)*430,channels:3,background:'#ffffff'}}).composite(tiles).png().toFile(path.join(qa,'diagrams-contact-sheet.png'));
console.log(JSON.stringify({offlineRenderedFigures:files.length,contactSheet:path.join(qa,'diagrams-contact-sheet.png')}));
