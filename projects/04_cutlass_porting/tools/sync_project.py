"""Sync the verified archive into a dedicated branch without changing source files."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = 'https://github.com/AI-1900/sdc_cutlass.git'
BRANCH = '04_cutlass_porting'
PREFIX = 'projects/04_cutlass_porting/'
parser = argparse.ArgumentParser()
parser.add_argument('--push', action='store_true')
args = parser.parse_args()
gitdir = ROOT / '.sync' / 'git'
env = dict(os.environ, GIT_INDEX_FILE=str(ROOT / '.sync' / 'archive.index'))

def git(*argv, data=None):
    return subprocess.check_output(['git', '--git-dir=' + str(gitdir), *argv], input=data, env=env).decode().strip()

if not gitdir.exists():
    subprocess.check_call(['git', 'init', '--bare', str(gitdir)])
    git('remote', 'add', 'origin', REPO)
git('fetch', '--depth=1', 'origin', 'main')
refs = git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH)
if refs:
    git('fetch', '--depth=1', 'origin', BRANCH)
    parent = git('rev-parse', 'FETCH_HEAD')
else:
    parent = git('rev-parse', 'refs/remotes/origin/main')
git('read-tree', parent)
manifest = json.loads((ROOT / 'archive-manifest.json').read_text(encoding='utf-8'))
entries = []
for item in manifest['files'] + [{'path': 'archive-manifest.json'}]:
    path = ROOT / item['path']
    if 'sha256' in item:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Changed since export: ' + item['path']
    sha = git('hash-object', '-w', '--', str(path))
    entries.append(f'100644 {sha}\t{PREFIX}{item["path"]}\0')
git('update-index', '-z', '--index-info', data=''.join(entries).encode('utf-8'))
tree = git('write-tree')
if tree == git('rev-parse', parent + '^{tree}'):
    commit = parent
else:
    commit = git('commit-tree', tree, '-p', parent, data=b'Archive CUTLASS project conversations, guides, examples and visualizations\n')
git('update-ref', 'refs/heads/' + BRANCH, commit)
if args.push:
    git('push', '--set-upstream', 'origin', f'refs/heads/{BRANCH}:refs/heads/{BRANCH}')
    remote = git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).split()[0]
    assert remote == commit, 'Remote verification failed'
print(json.dumps({'branch': BRANCH, 'commit': commit, 'files': len(entries), 'pushed': args.push}))
