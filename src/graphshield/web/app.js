'use strict';
const $ = id => document.getElementById(id);
const state = {catalog:null, cases:[], selected:null, epoch:0, detailEpoch:0, busy:false};
const statuses = {new:'New', in_review:'In review', escalated:'Escalated', closed:'Closed'};
const reasons = {repeated_strong_component:'Repeated connected transfer group',repeated_fan_in:'Repeated fan-in transfers',repeated_fan_out:'Repeated fan-out transfers'};
const el = (tag,text,cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const fmt = n => Number(n).toLocaleString(undefined,{maximumFractionDigits:2});
const title = s => s.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
function error(message){$('error').textContent=message;$('error').hidden=!message;}
async function api(path,body){
  const response=await fetch('/api/v1'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Request could not be accepted. Check the supplied fields.');
  return data;
}
function scope(){return {world_id:$('world').value,model:$('model').value,policy:$('policy').value};}
function note(){
  const chosen=state.catalog.models[$('model').value];
  $('selection-note').textContent=(chosen.selected_on_tuning?'Selected during Phase 5 tuning. ':'Comparison model; not selected during tuning. ')+($('policy').value==='tuned_threshold'?'Review threshold '+chosen.threshold.toFixed(3)+'.':'No score floor.')+' Up to 3 new cases per snapshot.';
}
async function loadQueue(){
  const epoch=++state.epoch;note();error('');
  const result=await api('/cases?'+new URLSearchParams({...scope(),limit:'200'}));
  if(epoch!==state.epoch)return;
  state.cases=result.items;
  $('metric-cases').textContent=fmt(result.total);
  $('metric-new').textContent=fmt(state.cases.filter(c=>c.status==='new').length);
  $('metric-closed').textContent=fmt(state.cases.filter(c=>c.status==='closed').length);
  $('metric-accounts').textContent=fmt(new Set(state.cases.flatMap(c=>c.members)).size);
  drawQueue();
  const visible=filteredCases();
  const selected=visible.find(c=>c.case_id===state.selected?.case_id)||visible[0];
  if(selected)await openCase(selected.case_id);
  else{++state.detailEpoch;state.selected=null;$('detail').hidden=true;$('detail-empty').hidden=false;}
}
function filteredCases(){const status=$('status-filter').value;return state.cases.filter(c=>status==='all'||c.status===status);}
function drawQueue(){
  const items=filteredCases();$('queue-count').textContent=items.length;$('queue').replaceChildren();
  if(!items.length){$('queue').append(el('div',state.cases.length?'No cases match this status.':'No saved cases yet. Run batch replay for this network and model.','empty'));return;}
  for(const c of items){
    const button=el('button',undefined,'case-row'+(state.selected?.case_id===c.case_id?' selected':''));button.type='button';button.setAttribute('aria-label','Open case '+c.case_id.slice(-6));
    const header=el('div',undefined,'row-head');header.append(el('span','Case '+c.case_id.slice(-6).toUpperCase(),'case-name'),el('span',c.score.toFixed(3),'score-pill'));
    const meta=el('div',undefined,'row-meta');meta.append(el('span',c.members.length+' accounts · Step '+c.cutoff_step),el('span',statuses[c.status],'status '+c.status));
    button.append(header,el('div',c.reasons.map(r=>reasons[r]||title(r)).join(' · '),'row-description'),meta);button.addEventListener('click',()=>openCase(c.case_id).catch(e=>error(e.message)));$('queue').append(button);
  }
}
function setTab(name){
  document.querySelectorAll('[data-tab]').forEach(button=>{const active=button.dataset.tab===name;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;$('view-'+button.dataset.tab).hidden=!active;});
}
function factList(target,entries){$(target).replaceChildren();for(const [label,value]of entries){$(target).append(el('dt',label),el('dd',String(value)));}}
async function openCase(id){
  const epoch=++state.detailEpoch;
  const c=await api('/cases/'+encodeURIComponent(id));if(epoch!==state.detailEpoch)return;
  state.selected=c;$('detail-empty').hidden=true;$('detail').hidden=false;
  $('case-title').textContent='Case '+c.case_id.slice(-6).toUpperCase();
  $('case-score').textContent=c.score.toFixed(3);
  $('case-subline').textContent=statuses[c.status]+' · Snapshot step '+c.cutoff_step+' · '+c.members.length+' accounts';
  $('case-footer').textContent='Evidence frozen at '+c.snapshot_id+' · '+c.provenance;
  $('reasons').replaceChildren(...c.reasons.map(r=>el('span',reasons[r]||title(r),'reason')));
  const transfers=c.evidence.transactions, devices=c.evidence.device_observations;
  $('evidence-counts').replaceChildren();
  for(const [number,label]of [[c.members.length,'accounts'],[transfers.length,'transfers'],[new Set(devices.map(r=>r.device_id)).size,'shared devices']]){const box=el('div');box.append(el('strong',fmt(number)),document.createTextNode(label));$('evidence-counts').append(box);}
  drawGraph(c);drawTimeline(c);
  const labels=new Map(c.members.map((a,i)=>[a,'A'+String(i+1).padStart(2,'0')]));
  $('transfer-rows').replaceChildren();
  for(const t of [...transfers].sort((a,b)=>a.step-b.step||a.transaction_id.localeCompare(b.transaction_id))){const row=el('tr');row.title=t.transaction_id;for(const value of [t.step,labels.get(t.sender_id),labels.get(t.receiver_id),fmt(t.amount),t.available_step])row.append(el('td',String(value)));$('transfer-rows').append(row);}
  $('transfer-count').textContent=transfers.length+' observed transfers. Hover a row for its source transaction ID.';
  $('account-identifiers').replaceChildren(...c.members.map(a=>el('div',labels.get(a)+' · '+a)));
  $('device-evidence').replaceChildren(...devices.map(o=>el('div',labels.get(o.account_id)+' · '+o.device_id+' · event '+o.step+' / available '+o.available_step)));
  if(!devices.length)$('device-evidence').append(el('div','No shared-device evidence in this case.'));
  factList('model-facts', [['Model',state.catalog.models[c.model].label],['Training run',state.catalog.learned_run],['Threshold',c.threshold.toFixed(6)],['Score meaning',c.score_semantics],['Embedding mode',title(c.embedding_mode)],['History revision',c.history_revision],['Bundle version',c.bundle_version],['Candidate ID',c.candidate_id]]);
  factList('features',Object.entries(c.features).map(([k,v])=>[title(k),Number.isInteger(v)?fmt(v):Number(v).toFixed(4)]));
  $('contributions').replaceChildren();
  if(c.contributions.available){
    $('contribution-note').textContent='Contributions explain this model’s raw margin, not guilt or fraud probability. Positive values raise its score; negative values lower it. Showing the six largest absolute contributions.';
    for(const [name,value]of Object.entries(c.contributions.features).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,6)){
      const row=el('div',undefined,'contribution');row.append(el('span',title(name)),el('strong',(value>=0?'+':'')+value.toFixed(3),value>=0?'positive':'negative'));$('contributions').append(row);
    }
  }else $('contribution-note').textContent=c.contributions.reason+'. Use the observed evidence and case features alongside this score.';
  $('review-status').value=c.status;$('review-note').value='';$('review-message').textContent='';
  $('review-history').replaceChildren();
  for(const r of [...c.reviews].reverse()){const row=el('div',undefined,'history-row');row.append(el('strong',statuses[r.status]),el('small','Review '+r.version+' · '+new Date(r.created_at).toLocaleString()),el('p',r.note));$('review-history').append(row);}
  if(!c.reviews.length)$('review-history').append(el('p','No reviewer decision has been recorded.','muted'));
  drawQueue();
}
function svgNode(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
function drawGraph(c){
  const svg=$('network');svg.replaceChildren();
  const defs=svgNode('defs'),marker=svgNode('marker',{id:'arrow',viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});marker.append(svgNode('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#a4bbc7'}));defs.append(marker);svg.append(defs);
  const positions=new Map(c.members.map((a,i)=>{const angle=(i/c.members.length)*Math.PI*2-Math.PI/2;return [a,{x:325+230*Math.cos(angle),y:155+112*Math.sin(angle),label:'A'+String(i+1).padStart(2,'0')}];}));
  const deviceIds=[...new Set(c.evidence.device_observations.map(r=>r.device_id))];
  const visibleDevices=deviceIds.slice(0,5);
  const devicePositions=new Map(visibleDevices.map((d,i)=>[d,{x:325+(i-(visibleDevices.length-1)/2)*46,y:155,label:'D'+(i+1)}]));
  const shared=new Set();
  for(const o of c.evidence.device_observations){const from=positions.get(o.account_id),to=devicePositions.get(o.device_id),key=o.account_id+'|'+o.device_id;if(!to||shared.has(key))continue;shared.add(key);svg.append(svgNode('line',{x1:from.x,y1:from.y,x2:to.x,y2:to.y,class:'device-link'}));}
  const edges=new Set();
  for(const t of c.evidence.transactions){const key=t.sender_id+'|'+t.receiver_id;if(edges.has(key))continue;edges.add(key);const a=positions.get(t.sender_id),b=positions.get(t.receiver_id),dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy)||1;svg.append(svgNode('line',{x1:a.x+dx*19/length,y1:a.y+dy*19/length,x2:b.x-dx*22/length,y2:b.y-dy*22/length,class:'transfer','marker-end':'url(#arrow)'}));}
  for(const [id,p]of positions){const group=svgNode('g');group.append(svgNode('title',{},id),svgNode('circle',{cx:p.x,cy:p.y,r:18,class:'account'}),svgNode('text',{x:p.x,y:p.y+4,class:'node-label'},p.label));svg.append(group);}
  for(const [id,p]of devicePositions){const group=svgNode('g');group.append(svgNode('title',{},id),svgNode('rect',{x:p.x-15,y:p.y-15,width:30,height:30,rx:5,class:'device'}),svgNode('text',{x:p.x,y:p.y+4},p.label));svg.append(group);}
  $('graph-caption').textContent='Arrows show observed transfer directions. Dashed links show shared-device observations. Full IDs are in Transfers.'+(deviceIds.length>5?' Showing 5 of '+deviceIds.length+' devices.':'');
}
function drawTimeline(c){
  const svg=$('timeline');svg.replaceChildren();const buckets=Array(24).fill(0),span=c.cutoff_step-c.window_start;
  for(const t of c.evidence.transactions){const bin=Math.min(23,Math.floor((t.step-c.window_start)/span*24));buckets[bin]++;}
  const max=Math.max(1,...buckets);svg.append(svgNode('line',{x1:12,y1:57,x2:638,y2:57}));
  buckets.forEach((count,i)=>{const h=count/max*38;const bar=svgNode('rect',{x:15+i*26,y:57-h,width:16,height:h,rx:2});bar.append(svgNode('title',{},count+' transfers'));svg.append(bar);});
  svg.append(svgNode('text',{x:12,y:76},'Step '+c.window_start),svgNode('text',{x:570,y:76},'Step '+(c.cutoff_step-1)));
  $('window-note').textContent='Only events and arrivals before step '+c.cutoff_step+' are included. Evidence window: steps '+c.window_start+'–'+(c.cutoff_step-1)+'.';
}
function busy(value){state.busy=value;for(const id of ['replay','world','model','policy'])$(id).disabled=value;$('replay').textContent=value?'Replay in progress…':'Run batch replay ↗';}
async function runReplay(){
  if(state.busy)return;error('');busy(true);
  try{
    const job=await api('/replays',scope());
    let current=job;
    while(!['complete','failed'].includes(current.status)){
      $('run-status').textContent=current.completed_snapshots+' / '+current.total_snapshots+' snapshots · '+current.alerts+' cases';
      await new Promise(resolve=>setTimeout(resolve,1200));current=await api('/replays/'+job.replay_id);
    }
    $('run-status').textContent=current.completed_snapshots+' / '+current.total_snapshots+' snapshots complete · '+current.alerts+' cases';
    if(current.status==='failed')throw new Error(current.error||'Replay failed. Retry to resume completed checkpoints.');
    await loadQueue();
  }catch(e){error(e.message);$('run-status').textContent='Replay needs attention. Retry to resume.';}finally{busy(false);}
}
async function init(){
  try{
    state.catalog=await api('/catalog');const c=state.catalog;
    for(const w of c.worlds){const option=el('option',w.label+(w.split==='train'?' · '+w.world_id.slice(-6):''));option.value=w.world_id;$('world').append(option);}
    for(const [name,m]of Object.entries(c.models)){const option=el('option',m.label+(m.selected_on_tuning?' · selected':''));option.value=name;$('model').append(option);}
    $('world').value=c.default_world;$('model').value=c.default_model;$('connection').textContent='Local app ready';$('version').textContent=c.learned_run+' · '+c.bundle_version.slice(0,10);$('replay').disabled=false;
    await loadQueue();
  }catch(e){$('connection').textContent='Connection unavailable';error(e.message+' Start the local app and refresh this page.');}
}
for(const id of ['world','model','policy'])$(id).addEventListener('change',()=>{state.selected=null;$('run-status').textContent='';loadQueue().catch(e=>error(e.message));});
$('status-filter').addEventListener('change',()=>loadQueue().catch(e=>error(e.message)));
$('replay').addEventListener('click',runReplay);
document.querySelectorAll('[data-tab]').forEach((button,index,buttons)=>{
  button.addEventListener('click',()=>setTab(button.dataset.tab));
  button.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();const next=(index+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;setTab(buttons[next].dataset.tab);buttons[next].focus();}});
});
$('review-form').addEventListener('submit',async event=>{
  event.preventDefault();if(!state.selected)return;
  const selected=state.selected;const note=$('review-note').value.trim();if(!note)return;
  $('save-review').disabled=true;error('');
  try{
    await api('/cases/'+selected.case_id+'/reviews',{request_id:crypto.randomUUID(),expected_version:selected.review_version,status:$('review-status').value,note});
    await loadQueue();$('review-message').textContent='Review saved. Model evidence is unchanged.';
  }catch(e){error(e.message);}finally{$('save-review').disabled=false;}
});
$('refresh-health').addEventListener('click',async()=>{
  $('refresh-health').disabled=true;
  try{
    const data=await api('/monitoring'),w=data.workflow,r=data.requests.routes;
    $('health-status').textContent=w.status==='ok'?'No replay failures or inactive jobs need attention.':'A replay needs attention.';
    factList('health-facts',[
      ['Requests since startup',r.reduce((sum,row)=>sum+row.requests,0)],
      ['Rejected requests (4xx)',r.reduce((sum,row)=>sum+row.errors_4xx,0)],
      ['Server errors (5xx)',r.reduce((sum,row)=>sum+row.errors_5xx,0)],
      ['Active replays',w.active_replays],
      ['Completed replays',w.replay_status_counts.complete||0],
      ['Cases awaiting review',w.case_status_counts.new||0],
      ['Local database',fmt(w.database_bytes/1048576)+' MB']]);
    $('health-issues').replaceChildren(...w.issues.map(i=>el('li',title(i.condition)+': '+i.action)));
  }catch(e){$('health-status').textContent='Unable to read system health: '+e.message;}
  finally{$('refresh-health').disabled=false;}
});
setTab('network');init();
