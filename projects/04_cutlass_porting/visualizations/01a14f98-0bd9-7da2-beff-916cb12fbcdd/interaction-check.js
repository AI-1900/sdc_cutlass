
(()=>{
const root=document.getElementById('cc-flow');
const q=id=>root.querySelector('#'+id);
let state={thread:0,value:0,k:0,tab:0};
const map=(t,v)=>({m:t%2+2*(v%2),n:Math.floor(t/2)+2*Math.floor(v/2)});
const restore=s=>{const p=s?.privateContent;if(!p)return;for(const key of ['thread','value','k','tab'])if(Number.isInteger(p[key])&&p[key]>=0&&p[key]<=3)state[key]=p[key];};
restore(window.openai?.widgetState);
function render(){
 q('cc-thread').value=state.thread;q('cc-value').value=state.value;
 const selected=map(state.thread,state.value);const cells=Array.from({length:16},()=>null);
 for(let t=0;t<4;t++)for(let v=0;v<4;v++){const c=map(t,v);cells[c.m*4+c.n]={t,v,...c};}
 q('cc-logical').innerHTML=cells.map(c=>'<div class="cell'+(c.t===state.thread?' chosen':'')+(c.t===state.thread&&c.v===state.value?' active':'')+'" aria-label="坐标 '+c.m+','+c.n+'，线程 '+c.t+'，值 '+c.v+'">T'+c.t+' · v'+c.v+'</div>').join('');
 q('cc-memory').innerHTML=cells.map((c,i)=>'<div class="cell'+(c.t===state.thread?' chosen':'')+(c.t===state.thread&&c.v===state.value?' active':'')+'">'+i+'</div>').join('');
 q('cc-detail').textContent='T'+state.thread+' / v'+state.value+' → ('+selected.m+', '+selected.n+') → 偏移 '+(selected.m*4+selected.n);
 q('cc-klabel').textContent=state.k<3?'K 分块 '+state.k+' / 2':'Epilogue';
 q('cc-a').textContent=state.k<3?'A[:, K'+state.k+']':'α × Acc';q('cc-b').textContent=state.k<3?'B[K'+state.k+', :]':'β × C';q('cc-acc').textContent=state.k<3?'Acc += A块 × B块':'写回 D';
 q('cc-prev').disabled=state.k===0;q('cc-next').disabled=state.k===3;
 root.querySelectorAll('[role="tab"]').forEach((b,i)=>{const on=i===state.tab;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on));q(b.getAttribute('aria-controls')).hidden=!on;});
}
function save(){window.openai?.setWidgetState?.({modelContent:{thread:state.thread,value:state.value,k:state.k,tab:state.tab},privateContent:state})?.catch(()=>{});}
q('cc-thread').addEventListener('change',e=>{state.thread=Number(e.target.value);render();save();});
q('cc-value').addEventListener('change',e=>{state.value=Number(e.target.value);render();save();});
q('cc-prev').addEventListener('click',()=>{state.k=Math.max(0,state.k-1);render();save();});
q('cc-next').addEventListener('click',()=>{state.k=Math.min(3,state.k+1);render();save();});
root.querySelectorAll('[role="tab"]').forEach((b,i)=>b.addEventListener('click',()=>{state.tab=i;render();save();}));
window.addEventListener('openai:set_globals',e=>{restore(e.detail?.globals?.widgetState);render();});render();
})();
