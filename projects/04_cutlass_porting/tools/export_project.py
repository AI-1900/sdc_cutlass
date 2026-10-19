"""Export visible project conversations and gather their visualization artifacts."""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODEX = pathlib.Path.home() / '.codex'
paths = subprocess.check_output(
    ['rg', '-l', '-F', ROOT.name, str(CODEX / 'sessions')], text=True
).splitlines()
out = ROOT / 'conversations'
out.mkdir(exist_ok=True)
index = []
for name in sorted(paths):
    rows = [json.loads(line) for line in pathlib.Path(name).read_text(encoding='utf-8').splitlines()]
    meta = next((r['payload'] for r in rows if r['type'] == 'session_meta'), {})
    if ROOT.name not in meta.get('cwd', '') or isinstance(meta.get('source'), dict):
        continue
    messages = []
    for row in rows:
        msg = row.get('payload', {})
        if row['type'] != 'response_item' or msg.get('type') != 'message':
            continue
        role = msg.get('role')
        if role not in ('user', 'assistant'):
            continue
        content = '\n'.join(c.get('text', '') for c in msg.get('content', []) if c.get('type') in ('input_text', 'output_text', 'text'))
        content = re.sub(r'<in-app-browser-context\b[^>]*>.*?</in-app-browser-context>', '', content, flags=re.S)
        if role == 'user' and content.lstrip().startswith(('# AGENTS.md', '<environment_context>', '<external_codex_apps_open_page>')):
            continue
        if content.strip():
            messages.append({'role': role, 'channel': msg.get('channel'), 'timestamp': row.get('timestamp'), 'text': content.strip()})
    if not messages:
        continue
    sid = meta['id']
    title = next((m['text'].splitlines()[0][:100] for m in messages if m['role'] == 'user'), sid)
    filename = pathlib.Path(name).stem.replace('rollout-', '') + '.md'
    chunks = [f'# {title}\n\n会话 ID：`{sid}`\n\n仅包含可见的用户问题和助手回复；工具输出、内部推理和环境注入信息不在此导出范围。\n']
    for number, msg in enumerate(messages, 1):
        label = '用户' if msg['role'] == 'user' else '助手'
        chunks.append(f'\n---\n\n## {number}. {label}\n\n{msg["text"]}\n')
    (out / filename).write_text(''.join(chunks), encoding='utf-8')
    index.append({'session_id': sid, 'title': title, 'file': filename, 'messages': len(messages)})
    # Each visualization folder is scoped to one verified project session.
    for folder in (CODEX / 'visualizations').glob(f'*/*/*/{sid}'):
        shutil.copytree(folder, ROOT / 'visualizations' / sid, dirs_exist_ok=True)
(out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(out / 'README.md').write_text('# 项目问答目录\n\n' + '\n'.join(f'- [{x["title"].replace("[", "（").replace("]", "）")}]({x["file"]})（{x["messages"]} 条消息）' for x in index) + '\n', encoding='utf-8')
files = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ('.git', '.sync', '__pycache__')) or p.name == 'archive-manifest.json':
        continue
    data = p.read_bytes()
    files.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
(ROOT / 'archive-manifest.json').write_text(json.dumps({'project': '04_cutlass_porting', 'scope': 'Current local project files and locally available project conversations/visualizations', 'files': files}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'conversations': len(index), 'messages': sum(x['messages'] for x in index), 'files': len(files), 'bytes': sum(x['bytes'] for x in files)}))
