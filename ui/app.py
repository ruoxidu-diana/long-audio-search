import os
import io
import wave

os.environ["GRADIO_SERVER_NAME"] = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ["GRADIO_SERVER_PORT"] = os.getenv("GRADIO_SERVER_PORT", "7860")

import pandas as pd
import numpy as np
import requests
import gradio as gr

API_URL_SERVER = os.getenv("API_URL", "http://api:8000")
API_URL_PUBLIC = os.getenv("API_URL_PUBLIC", API_URL_SERVER)

TABLE_COLS = ["id", "filename", "tags", "description", "duration", "url", "score"]


def _public_url(path_or_url: str) -> str:
    if not path_or_url:
        return ""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"{API_URL_PUBLIC}{path_or_url}"


def _player_html(public_url: str, track_id: str | None):
    if not public_url:
        return "<i>No selection</i>"
    safe = public_url.replace('"', "%22")
    dl = f"{API_URL_PUBLIC}/download/{track_id}" if track_id else safe
    return f"""
    <div style="display:flex;flex-direction:column;gap:8px">
      <audio controls src="{safe}" style="width:100%"></audio>
      <div>
        <a href="{dl}">⬇️ Descargar</a> &nbsp;|&nbsp;
        <code>{safe}</code>
      </div>
    </div>
    """


def api_search(query: str):
    query = (query or "").strip()
    if not query:
        empty_df = pd.DataFrame(columns=TABLE_COLS)
        return empty_df, [], gr.update(choices=[], value=None), gr.update(value=_player_html("", None))
    r = requests.get(f"{API_URL_SERVER}/search", params={"query": query, "limit": 30}, timeout=120)
    r.raise_for_status()
    data = r.json() or []
    items = []
    for it in data:
        items.append(
            {
                "id": it.get("id", ""),
                "filename": it.get("filename", ""),
                "tags": it.get("tags", "") or "",
                "description": it.get("description", "") or "",
                "duration": it.get("duration", 0.0),
                "url": _public_url(it.get("url", "")),
                "score": round(float(it.get("score", 0.0)), 6),
            }
        )
    df = pd.DataFrame(items, columns=TABLE_COLS)
    choices = [f"{row['filename']}  ·  {row['id']}" for _, row in df.iterrows()]
    first_player = _player_html(items[0]["url"], items[0]["id"]) if items else _player_html("", None)
    return df, items, gr.update(choices=choices, value=(choices[0] if choices else None)), gr.update(value=first_player)


def api_upload(file, tags, description, auto_toggle: bool):
    if file is None:
        return "No file selected."
    with open(file.name, "rb") as f:
        files = {"file": (os.path.basename(file.name), f.read())}
    data = {"tags": tags or "", "description": description or "", "auto": 1 if auto_toggle else 0}
    r = requests.post(f"{API_URL_SERVER}/upload", files=files, data=data, timeout=600)
    if r.status_code != 200:
        return f"Error {r.status_code}: {r.text}"
    it = r.json()
    return f"Uploaded: {it.get('filename','')} ({it.get('id','')})"


def api_update(track_id, tags, description):
    track_id = (track_id or "").strip()
    if not track_id:
        return "Missing Track ID."
    r = requests.patch(f"{API_URL_SERVER}/tracks/{track_id}", json={"tags": tags, "description": description}, timeout=120)
    if r.status_code != 200:
        return f"Error: {r.status_code} {r.text}"
    return "Updated."


def api_delete(track_id):
    track_id = (track_id or "").strip()
    if not track_id:
        return "Missing Track ID."
    r = requests.delete(f"{API_URL_SERVER}/tracks/{track_id}", timeout=120)
    return "Deleted." if r.status_code == 200 else f"Error: {r.status_code} {r.text}"


def api_rescan():
    r = requests.post(f"{API_URL_SERVER}/reindex-incremental", timeout=600)
    try:
        r.raise_for_status()
        return f"Rescan OK: {r.text}"
    except Exception:
        return f"Rescan error: {r.status_code} {r.text}"


def on_pick(selection_label: str, table_rows):
    if not selection_label:
        return gr.update(value=_player_html("", None)), "", "", ""
    if "·" in selection_label:
        track_id = selection_label.split("·")[-1].strip()
    else:
        track_id = selection_label.strip()
    if not table_rows:
        return gr.update(value=_player_html("", None)), "", "", ""
    row = next((r for r in table_rows if r.get("id") == track_id), None)
    if not row:
        return gr.update(value=_player_html("", None)), "", "", ""
    html = _player_html(row.get("url", ""), track_id)
    return html, track_id, row.get("tags", ""), row.get("description", "")


# ---- Segment Search ----
SEG_COLS = ["Filename", "Start", "End", "Score"]


def _fmt_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _seg_label(i: int, it: dict) -> str:
    return f"{i + 1}. {it['filename']}  {_fmt_time(it['start'])}–{_fmt_time(it['end'])}  ({it['score']:.3f})"


def api_search_segments(query: str):
    query = (query or "").strip()
    if not query:
        return pd.DataFrame(columns=SEG_COLS), [], gr.update(choices=[], value=None), None
    try:
        r = requests.get(f"{API_URL_SERVER}/search-segments", params={"query": query, "limit": 20}, timeout=120)
        r.raise_for_status()
    except Exception as e:
        raise gr.Error(f"Search failed: {e}")
    items = []
    for it in r.json() or []:
        items.append(
            {
                "filename": it.get("filename", ""),
                "start": float(it.get("start", 0.0)),
                "end": float(it.get("end", 0.0)),
                "score": float(it.get("score", 0.0)),
            }
        )
    rows = [[it["filename"], _fmt_time(it["start"]), _fmt_time(it["end"]), f"{it['score']:.3f}"] for it in items]
    df = pd.DataFrame(rows, columns=SEG_COLS)
    choices = [_seg_label(i, it) for i, it in enumerate(items)]
    return df, items, gr.update(choices=choices, value=None), None


def on_seg_pick(label: str, items):
    if not label or not items:
        return None
    idx = int(label.split(".", 1)[0]) - 1
    if idx < 0 or idx >= len(items):
        return None
    it = items[idx]
    try:
        r = requests.get(
            f"{API_URL_SERVER}/clip",
            params={"filename": it["filename"], "start": it["start"], "end": it["end"]},
            timeout=120,
        )
        r.raise_for_status()
    except Exception as e:
        raise gr.Error(f"Clip failed: {e}")
    with wave.open(io.BytesIO(r.content), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise gr.Error(f"Unsupported WAV sample width: {sampwidth}")
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return (sr, audio)


def on_seg_table_select(items, evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not items or row is None or row >= len(items):
        return gr.update()
    return gr.update(value=_seg_label(row, items[row]))


def api_rebuild_segments():
    try:
        r = requests.post(f"{API_URL_SERVER}/reindex-segments", timeout=3600)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


with gr.Blocks(title="Semantic Audio Search") as demo:
    gr.Markdown("# 🔎 Semantic Audio Search")

    with gr.Tab("Search"):
        with gr.Row():
            q = gr.Textbox(label="Search query", scale=5)
            btn = gr.Button("Search", variant="primary", scale=1)
        table = gr.Dataframe(
            headers=["ID", "Filename", "Tags", "Description", "Duration (s)", "URL", "Score"],
            value=[],
            datatype=["str", "str", "str", "str", "number", "str", "number"],
            interactive=False,
            wrap=True,
            height=360,
        )
        state_items = gr.State([])

        gr.Markdown("### Preview / Edit")
        pick = gr.Dropdown(label="Pick a result", choices=[])
        player = gr.HTML()
        track_id = gr.Textbox(label="Track ID")
        edit_tags = gr.Textbox(label="Tags")
        edit_desc = gr.Textbox(label="Description")
        with gr.Row():
            upd_btn = gr.Button("Save changes")
            del_btn = gr.Button("Delete")
            rescan_btn = gr.Button("Rescan library")

        upd_out = gr.Markdown()
        del_out = gr.Markdown()
        rescan_out = gr.Markdown()

        btn.click(api_search, inputs=q, outputs=[table, state_items, pick, player])
        pick.change(on_pick, inputs=[pick, state_items], outputs=[player, track_id, edit_tags, edit_desc])
        upd_btn.click(api_update, inputs=[track_id, edit_tags, edit_desc], outputs=upd_out)
        del_btn.click(api_delete, inputs=[track_id], outputs=del_out)
        rescan_btn.click(api_rescan, outputs=rescan_out)

    with gr.Tab("Manage Library"):
        up_file = gr.File(label="Audio file")
        up_tags = gr.Textbox(label="Tags")
        up_desc = gr.Textbox(label="Description")
        auto_toggle = gr.Checkbox(label="Auto-caption/auto-tags", value=True)
        up_btn = gr.Button("Upload", variant="primary")
        up_out = gr.Markdown()
        up_btn.click(api_upload, inputs=[up_file, up_tags, up_desc, auto_toggle], outputs=up_out)

    with gr.Tab("Segment Search"):
        with gr.Row():
            seg_q = gr.Textbox(label="Search query", scale=5)
            seg_btn = gr.Button("Search", variant="primary", scale=1)
        seg_table = gr.Dataframe(
            headers=SEG_COLS,
            value=[],
            datatype=["str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            height=360,
        )
        seg_items = gr.State([])

        seg_pick = gr.Dropdown(label="Pick a segment (or click a table row)", choices=[])
        seg_audio = gr.Audio(label="Clip", type="numpy", interactive=False)

        rebuild_btn = gr.Button("Rebuild segments")
        rebuild_out = gr.JSON(label="Rebuild result")

        seg_btn.click(api_search_segments, inputs=seg_q, outputs=[seg_table, seg_items, seg_pick, seg_audio])
        seg_q.submit(api_search_segments, inputs=seg_q, outputs=[seg_table, seg_items, seg_pick, seg_audio])
        seg_table.select(on_seg_table_select, inputs=[seg_items], outputs=seg_pick)
        seg_pick.change(on_seg_pick, inputs=[seg_pick, seg_items], outputs=seg_audio)
        rebuild_btn.click(api_rebuild_segments, outputs=rebuild_out)

demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    show_error=True,
)
