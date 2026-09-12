"""JSON, manifest, command-list and static HTML exporters."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .media import render_selected_video
from .models import Candidate, HighlightConfig, VideoInfo, format_timestamp


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_checkpoint(
    output_dir: Path,
    video_path: str,
    video: VideoInfo,
    config: HighlightConfig,
    candidates: Iterable[Candidate],
) -> None:
    """Persist resumable generation state after each completed candidate."""
    source = Path(video_path)
    stat = source.stat()
    candidate_list = list(candidates)
    _json(output_dir / ".highlight-progress.json", {
        "version": 1,
        "input": {
            "path": str(source.absolute()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "video": video.to_dict(),
        "config": config.to_dict(),
        "completed_ids": [candidate.id for candidate in candidate_list if _candidate_complete(candidate, config)],
        "candidates": [candidate.to_dict() for candidate in candidate_list],
    })


def load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / ".highlight-progress.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _candidate_complete(candidate: Candidate, config: HighlightConfig) -> bool:
    return bool(candidate.thumbnail and (config.no_previews or candidate.preview))


def export_clips_txt(path: Path, candidates: Iterable[Candidate], video_path: str) -> None:
    lines = []
    for index, candidate in enumerate(candidates, 1):
        duration = max(0, candidate.end_ms - candidate.start_ms) / 1000
        argv = [
            "ffmpeg", "-y", "-ss", f"{candidate.start_ms / 1000:.3f}", "-i", str(video_path),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-c:a", "aac",
            f"clip_{index:04d}.mp4",
        ]
        lines.append(shlex.join(argv))
    _atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def export_html(
    path: Path,
    candidates: list[Candidate],
    video_name: str,
    video_path: str | None = None,
    render_endpoint: str | None = None,
) -> None:
    data = json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = escape(f"Buzz 高光候选 - {video_name}")
    video_name_json = json.dumps(video_path or video_name, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    render_endpoint_json = json.dumps(render_endpoint, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    render_button = '<button id="render">生成最终视频</button>' if render_endpoint else ''
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root {{ color-scheme: light dark; }} body {{ font: 14px system-ui,sans-serif; margin: 0; background: #f4f5f7; color:#202124; }}
header {{ position:sticky;top:0;z-index:2;padding:14px 18px;background:#fff;border-bottom:1px solid #ddd; }}
.controls {{ display:flex;gap:8px;flex-wrap:wrap;align-items:center; }} button,select,input {{ padding:7px 9px;border:1px solid #bbb;border-radius:6px;background:#fff;color:inherit; }}
#summary {{ margin-top:8px;color:#666; }} main {{ display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;padding:16px; }}
.card {{ background:#fff;border:2px solid #ddd;border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:8px; }}
.card.keep {{ border-color:#20a464; }} .card.ignore {{ opacity:.58;border-color:#9b4dca; }} img {{ width:100%;aspect-ratio:16/9;object-fit:cover;background:#ddd;border-radius:6px; }}
.meta {{ display:grid;grid-template-columns:auto 1fr;gap:3px 9px; }} .transcript {{ white-space:pre-wrap;max-height:6em;overflow:auto;line-height:1.4; }}
video {{ width:100%;max-height:250px;background:#000;border-radius:6px; }} .actions {{ display:flex;gap:6px; }}
@media (prefers-color-scheme: dark) {{ body {{ background:#202124;color:#eee; }} header,.card {{ background:#2b2c2f;border-color:#555; }} button,select,input {{ background:#333;color:#eee;border-color:#666; }} }}
</style></head><body><header><div class="controls">
<strong>Buzz 高光候选</strong><select id="sort"><option value="score">按分数</option><option value="time">按时间</option><option value="status">按状态</option></select>
<select id="filter"><option value="all">全部状态</option><option value="unprocessed">只看未处理</option><option value="keep">只看保留</option><option value="ignore">只看忽略</option></select>
<input id="search" placeholder="筛选字幕关键词"><button id="ignoreAll">全部标记忽略</button><button id="export">导出已选</button>{render_button}</div><div id="summary"></div></header><main id="cards"></main>
<script>
const initial = {data}; const videoName = {video_name_json}; const renderEndpoint = {render_endpoint_json}; const key = 'buzz-highlight-selection:' + location.pathname; const saved = JSON.parse(localStorage.getItem(key) || '{{}}');
const state = initial.map(c => ({{...c, status: saved[c.id] || c.status || 'unprocessed', selected: (saved[c.id] || c.status) === 'keep'}}));
const esc = s => String(s ?? '').replace(/[&<>"']/g, x => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[x]));
const fmt = ms => {{ let s=Math.max(0,Math.round(ms))/1000; let h=Math.floor(s/3600);s%=3600;let m=Math.floor(s/60);s=(s%60).toFixed(3);return `${{String(h).padStart(2,'0')}}:${{String(m).padStart(2,'0')}}:${{String(s).padStart(6,'0')}}`; }};
function persist() {{ localStorage.setItem(key, JSON.stringify(Object.fromEntries(state.map(c => [c.id,c.status])))); }}
function download(name, content, type='application/json') {{ const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{{type}}));a.download=name;a.click();URL.revokeObjectURL(a.href); }}
function render() {{ const sort=document.querySelector('#sort').value, filter=document.querySelector('#filter').value, query=document.querySelector('#search').value.toLowerCase();
 let rows=state.filter(c => (filter==='all'||c.status===filter) && (!query||String(c.transcript).toLowerCase().includes(query)));
 rows.sort((a,b)=>sort==='time'?a.start_ms-b.start_ms:sort==='status'?a.status.localeCompare(b.status):b.score-a.score);
 document.querySelector('#summary').textContent=`显示 ${{rows.length}} / ${{state.length}}，保留 ${{state.filter(c=>c.status==='keep').length}} 个`;
 document.querySelector('#cards').innerHTML=rows.map(c=>`<article class="card ${{c.status==='keep'?'keep':c.status==='ignore'?'ignore':''}}"><img src="${{esc(c.thumbnail||'')}}" loading="lazy" alt="${{esc(c.id)}}"><div class="meta"><b>${{esc(c.id)}}</b><span>分数 ${{Number(c.score).toFixed(3)}}</span><b>主体</b><button class="timestamp" data-start="${{c.start_ms}}" data-end="${{c.end_ms}}">${{fmt(c.start_ms)}} → ${{fmt(c.end_ms)}}</button><b>预览</b><span>${{fmt(c.preview_start_ms)}} → ${{fmt(c.preview_end_ms)}}</span><b>原因</b><span>${{esc((c.reasons||[]).join(', '))}}</span></div>${{c.transcript?`<div class="transcript">${{esc(c.transcript)}}</div>`:''}}${{c.preview?`<video controls preload="metadata" src="${{esc(c.preview)}}"></video>`:'<small>预览生成失败或未生成</small>'}}<div class="actions"><button data-action="keep" data-id="${{c.id}}">保留</button><button data-action="ignore" data-id="${{c.id}}">忽略</button><button data-action="unprocessed" data-id="${{c.id}}">未处理</button></div></article>`).join('');
 document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{{state.find(c=>c.id===b.dataset.id).status=b.dataset.action;state.find(c=>c.id===b.dataset.id).selected=b.dataset.action==='keep';persist();render();}});
 document.querySelectorAll('.timestamp').forEach(b=>b.onclick=()=>navigator.clipboard?.writeText(`${{fmt(+b.dataset.start)}} --> ${{fmt(+b.dataset.end)}}`)); }}
['sort','filter','search'].forEach(id=>document.querySelector('#'+id).oninput=render);
document.querySelector('#ignoreAll').onclick=()=>{{state.filter(c=>c.status==='unprocessed').forEach(c=>c.status='ignore');persist();render();}};
if (renderEndpoint) document.querySelector('#render').onclick=async()=>{{const selected=state.filter(c=>c.status==='keep');if(!selected.length){{alert('请先保留至少一个片段');return;}}const button=document.querySelector('#render');button.disabled=true;button.textContent='正在生成...';try{{const response=await fetch(renderEndpoint,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{ids:selected.map(c=>c.id),statuses:Object.fromEntries(state.map(c=>[c.id,c.status]))}})}});const result=await response.json();if(!response.ok) throw new Error(result.error||'生成失败');window.location.href=result.url;}}catch(error){{alert(error.message);button.disabled=false;button.textContent='生成最终视频';}}}};
 document.querySelector('#export').onclick=()=>{{const selected=state.filter(c=>c.status==='keep');download('selected.json',JSON.stringify(selected,null,2));download('clips.txt',selected.map((c,i)=>`ffmpeg -y -ss ${{(c.start_ms/1000).toFixed(3)}} -i "${{esc(videoName)}}" -t ${{((c.end_ms-c.start_ms)/1000).toFixed(3)}} -c:v libx264 -c:a aac "clip_${{String(i+1).padStart(4,'0')}}.mp4"`).join('\\n')+'\\n','text/plain');}};
render();
</script></body></html>'''
    _atomic_text(path, html)


class _ResultRequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/render":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            ids = set(payload.get("ids", []))
            statuses = payload.get("statuses", {})
            candidates = self.server.candidates  # type: ignore[attr-defined]
            selected = [candidate for candidate in candidates if candidate.id in ids]
            for candidate in candidates:
                if candidate.id in statuses and statuses[candidate.id] in {"unprocessed", "keep", "ignore"}:
                    candidate.status = statuses[candidate.id]
                candidate.selected = candidate.id in ids
            if not selected:
                raise ValueError("select at least one candidate before rendering")
            output_path = render_selected_video(
                self.server.ffmpeg,  # type: ignore[attr-defined]
                self.server.video_path,  # type: ignore[attr-defined]
                selected,
                self.server.output_dir,  # type: ignore[attr-defined]
                has_audio=self.server.has_audio,  # type: ignore[attr-defined]
            )
            _json(self.server.output_dir / "selected.json", [candidate.to_dict() for candidate in candidates if candidate.selected])  # type: ignore[attr-defined]
            self._send_json({"url": output_path.name})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _send_json(self, value: dict[str, str], status: int = 200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def serve_result_page(
    output_dir: Path,
    candidates: list[Candidate],
    video_path: str,
    ffmpeg: str,
    has_audio: bool,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    class HighlightResultHandler(_ResultRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), HighlightResultHandler)
    server.candidates = candidates  # type: ignore[attr-defined]
    server.video_path = video_path  # type: ignore[attr-defined]
    server.output_dir = output_dir  # type: ignore[attr-defined]
    server.ffmpeg = ffmpeg  # type: ignore[attr-defined]
    server.has_audio = has_audio  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def export_outputs(
    output_dir: Path,
    candidates: list[Candidate],
    video: VideoInfo,
    config: HighlightConfig,
    video_path: str,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    render_endpoint: str | None = None,
) -> None:
    errors = errors or []
    warnings = warnings or []
    _json(output_dir / "candidates.json", {
        "video": video.to_dict(), "config": config.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "errors": errors, "warnings": warnings,
    })
    _json(output_dir / "selected.json", [candidate.to_dict() for candidate in candidates if candidate.selected])
    export_clips_txt(output_dir / "clips.txt", [c for c in candidates if c.selected], video_path)
    export_html(
        output_dir / "index.html",
        candidates,
        Path(video_path).name,
        video_path,
        render_endpoint=render_endpoint,
    )
    _json(output_dir / "manifest.json", {
        "input": {"path": str(Path(video_path).absolute()), "name": Path(video_path).name},
        "video": video.to_dict(), "config": config.to_dict(),
        "mode": "fixed_window" if config.no_scene_detection else "fixed_window+scene_change",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors, "warnings": warnings,
        "candidate_count": len(candidates),
    })
