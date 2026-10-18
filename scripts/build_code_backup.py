"""Create a source-only backup with neutral headers and per-file checksums."""
import argparse
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRS = ('configs', 'docs', 'provenance', 'scripts', 'src', 'tests', 'site', 'assets', 'recipes', '.github')
ALLOWED_ROOT = ('README.md', 'AGENTS.md', '.gitignore', 'LICENSE', 'NOTICE', 'pyproject.toml', 'requirements.txt', 'requirements-linux-cu128-py311.txt', 'requirements-demo-linux-cu128-py311.txt', 'CONTRIBUTING.md', 'CHANGELOG.md', 'MANIFEST.in')
ALLOWED_SUFFIXES = {'.py', '.json', '.md', '.html', '.css', '.js', '.toml', '.txt', '.yaml', '.yml', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.csv', '.example', '.bib'}

INTERNAL_ONLY = {"AGENTS.md", "docs/backup-policy.md", "docs/release-readiness.md"}

def build(output, privacy_rules, release_candidate=False):
    rules = json.loads(privacy_rules.read_text())
    forbidden = [value.casefold().encode() for value in rules['forbidden_strings']]
    files = [ROOT / name for name in ALLOWED_ROOT if (ROOT / name).is_file()]
    for name in ALLOWED_DIRS:
        files.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and p.suffix in ALLOWED_SUFFIXES and '__pycache__' not in p.parts)
    entries, manifest = [], []
    for p in sorted(files):
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError('symlinks or external paths are forbidden')
        rel = p.relative_to(ROOT).as_posix()
        if release_candidate and rel in INTERNAL_ONLY:
            continue
        data = p.read_bytes()
        if release_candidate and rel == 'provenance/check-report.json':
            report = json.loads(data)
            report.get('checks', {}).pop('private_drive_backup', None)
            data = (json.dumps(report, indent=2) + '\n').encode()
        # Exact, manually reviewed upstream legal notices can be exempted in private rules.
        if rel not in rules.get('legal_notice_exceptions', []):
            scan = rel.encode().lower() + b'\n' + data.lower()
            if any(value in scan for value in forbidden):
                raise ValueError('privacy scan failed in ' + rel)
        manifest.append({'path':rel,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        entries.append((rel, data))
    created = int(time.time())
    index = json.dumps({'schema_version':1,'created_unix':created,'scope':'private source candidate; validation scope is recorded in provenance receipts','files':manifest},indent=2).encode() + b'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output, 'w:gz', format=tarfile.PAX_FORMAT) as archive:
        for rel,data in entries + [('BACKUP_MANIFEST.json',index)]:
            member = tarfile.TarInfo(rel)
            member.size = len(data); member.mode = 0o600; member.mtime = created
            member.uid = member.gid = 0; member.uname = member.gname = ''
            archive.addfile(member, io.BytesIO(data))
    output.chmod(0o600)
    print(json.dumps({'files':len(entries),'archive_bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--release-candidate',action='store_true',help='Exclude internal agent and backup-operation documents')
    parser.add_argument('--privacy-rules',type=Path,required=True)
    args = parser.parse_args()
    build(args.output,args.privacy_rules,args.release_candidate)
