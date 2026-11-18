export const el = (selector, root=document) => root.querySelector(selector);
export const html = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;",
})[char]);
export const attr = html;
export const n = (value, digits=2) => Number.isFinite(value)
  ? value.toLocaleString("en-US",{maximumFractionDigits:digits}) : "—";
export const percent = (value, digits=1) => Number.isFinite(value) ? n(value*100,digits)+"%" : "—";
export const shortDate = value => value ? new Date(value).toLocaleDateString(undefined,{month:"short",day:"numeric"}) : "";
export const search = data => new URLSearchParams(Object.entries(data).filter(([,v])=>v!==null&&v!==undefined)).toString();

export function control(text, action, {id, selected=false, type="quiet", disabled=false, title=""}={}) {
  return `<button type="button" class="${attr(type)} ${selected?"active":""}" data-action="${attr(action)}"${id===undefined?"":` data-id="${attr(id)}"`}${selected?' aria-pressed="true"':""}${disabled?" disabled":""}${title?` title="${attr(title)}"`:""}>${html(text)}</button>`;
}

export function link(text,url,{small=false}={}) {
  return `<a class="button ${small?"quiet":""}" href="${attr(url)}" download>${html(text)}</a>`;
}

export function field(label,name,value="",{type="text",step,min,max,placeholder="",required=false}={}) {
  return `<label class="field">${html(label)}<input type="${attr(type)}" name="${attr(name)}"${type==="file"?"":` value="${attr(value)}"`}${step===undefined?"":` step="${attr(step)}"`}${min===undefined?"":` min="${attr(min)}"`}${max===undefined?"":` max="${attr(max)}"`}${placeholder?` placeholder="${attr(placeholder)}"`:""}${required?" required":""}></label>`;
}

export function select(label,name,options,selected) {
  return `<label class="field">${html(label)}<select name="${attr(name)}">${options.map(([value,label])=>`<option value="${attr(value)}"${value===selected?" selected":""}>${html(label)}</option>`).join("")}</select></label>`;
}

export function textArea(label,name,value="",placeholder="") {
  return `<label class="field">${html(label)}<textarea name="${attr(name)}" placeholder="${attr(placeholder)}">${html(value)}</textarea></label>`;
}

export function check(label,name,value=true) {
  return `<label class="check"><input type="checkbox" name="${attr(name)}"${value?" checked":""}>${html(label)}</label>`;
}

export const box = (title,contents) => `<section class="box"><h3>${html(title)}</h3>${contents}</section>`;
export const warning = text => `<div class="warning">${html(text)}</div>`;

export function stat(label,value) {
  return `<div class="stat-line"><span>${html(label)}</span><strong class="mono">${html(value)}</strong></div>`;
}

export function metrics(values) {
  return `<div class="metrics">${values.map(([label,value])=>`<div class="metric"><span>${html(label)}</span><strong>${html(value)}</strong></div>`).join("")}</div>`;
}

export function table(head,rows) {
  return `<div class="table-scroll"><table><thead><tr>${head.map(label=>`<th>${html(label)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(value=>`<td>${html(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

export const chart = (label,id="chart") => `<canvas class="chart" id="${attr(id)}" role="img" aria-label="${attr(label)}"></canvas>`;

export function drawChart(canvas,config) {
  if (!canvas||!config||!config.series?.length) return;
  const dpi = Math.min(devicePixelRatio||1,2);
  const w = Math.max(canvas.clientWidth,190), h = Math.max(canvas.clientHeight,100);
  canvas.width = Math.round(w*dpi);
  canvas.height = Math.round(h*dpi);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpi,dpi);
  const left=40, right=w-11, top=10, bottom=h-23;
  const all = config.series.flatMap(series=>series.points).filter(([x,y])=>Number.isFinite(x)&&Number.isFinite(y));
  if (!all.length) return;
  const xs = all.map(p=>p[0]), ys = all.map(p=>p[1]);
  const xmin=config.xmin??Math.min(...xs), xmax=config.xmax??Math.max(...xs);
  const ymin=config.ymin??Math.min(...ys), ymax=config.ymax??Math.max(...ys);
  const toX=x => left + (right-left)*(x-xmin)/Math.max(xmax-xmin,1e-12);
  const toY=y => bottom - (bottom-top)*(y-ymin)/Math.max(ymax-ymin,1e-12);
  ctx.font = "10px monospace";
  ctx.strokeStyle="#323d46";
  ctx.fillStyle="#a5b0b8";
  for (let i=0;i<=3;i++) {
    const y=top+(bottom-top)*i/3;
    ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
    ctx.fillText(n(ymax+(ymin-ymax)*i/3,3),4,y+3,33);
  }
  ctx.fillText(n(xmin,2),left,bottom+15,40);
  ctx.textAlign="right";
  ctx.fillText(n(xmax,2)+(config.unit?" "+config.unit:""),right,bottom+15);
  ctx.textAlign="left";
  if (Number.isFinite(config.guide)) {
    ctx.strokeStyle="#bd9e69";ctx.setLineDash([4,4]);ctx.beginPath();
    ctx.moveTo(left,toY(config.guide));ctx.lineTo(right,toY(config.guide));ctx.stroke();ctx.setLineDash([]);
  }
  for (const entry of config.series) {
    ctx.fillStyle=entry.color??"#ecbb72";ctx.strokeStyle=ctx.fillStyle;ctx.lineWidth=1.7;
    if (entry.bars) {
      const width=(right-left)/Math.max(entry.points.length,1);
      for (const [x,y] of entry.points) ctx.fillRect(toX(x)-width*0.45,toY(y),width*0.87,Math.max(0,bottom-toY(y)));
      continue;
    }
    ctx.beginPath();
    let continuous=false;
    for (const [x,y] of entry.points) {
      if (!Number.isFinite(y)) { continuous=false; continue; }
      if (continuous) ctx.lineTo(toX(x),toY(y)); else ctx.moveTo(toX(x),toY(y));
      continuous=true;
    }
    ctx.stroke();
    if (entry.points.length<=40) {
      for (const [x,y] of entry.points) if (Number.isFinite(y)) {
        ctx.beginPath();ctx.arc(toX(x),toY(y),2.3,0,Math.PI*2);ctx.fill();
      }
    }
  }
}
