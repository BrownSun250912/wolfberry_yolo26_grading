"""Check paper links and the initial handoff snapshot using Python's standard library.

After intentional manuscript edits, use --links-only to check navigation without
requiring the original manuscript hash. Frozen result CSVs still need a documented
correction process rather than silent edits.
"""
from pathlib import Path
from urllib.parse import unquote
import argparse,csv,hashlib,json,re,sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links-only',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'collaboration/PUBLISH_MANIFEST.json').read_text(encoding='utf-8'))
    errors=[];checked_links=0;files=manifest['files'];secrets=[]
    for item in files:
        p=ROOT/item['path']
        if not p.is_file():errors.append('Missing file: '+item['path']);continue
        if not args.links_only and hashlib.sha256(p.read_bytes()).hexdigest()!=item['published_sha256']:
            errors.append('Initial snapshot hash changed: '+item['path'])
        if p.suffix in {'.md','.csv','.json','.yaml','.yml','.txt','.py','.ps1'}:
            text=p.read_text(encoding='utf-8-sig')
            if re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):
                secrets.append(item['path'])
        if p.suffix!='.md':continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=unquote(target.strip().strip('<>'))
            if target.startswith(('http://','https://','mailto:','#','data:')):continue
            checked_links+=1
            if re.match(r'^[A-Za-z]:[/\\]',target) or target.startswith('file:'):
                errors.append('Machine-local link: '+item['path']);continue
            name=target.split('#',1)[0]
            if name and not (p.parent/name).resolve().exists():errors.append('Broken link: '+item['path']+' -> '+target)
    if secrets:errors.extend('Potential credential material in '+name for name in secrets)
    metrics=ROOT/'outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv'
    with metrics.open(encoding='utf-8-sig',newline='') as f:rows={r['arm']:r for r in csv.DictReader(f)}
    for arm,ap,correct in [('raw_YOLO',.7803632767397777,456),('original_defect_rank',.8072028356784946,481),('original_defect_rank_prototype',.8072028356784946,480)]:
        r=rows[arm]
        if abs(float(r['Mask_AP50'])-ap)>1e-12 or int(r['correct_grade'])!=correct or int(r['GT'])!=576 or int(r['localized'])!=560:
            errors.append('Frozen test endpoint changed: '+arm)
    zh=(ROOT/'paper/manuscript_20261025/MANUSCRIPT_ZH.md').read_text(encoding='utf-8')
    en=(ROOT/'paper/manuscript_20261025/MANUSCRIPT_EN.md').read_text(encoding='utf-8')
    for token in ['80.72','78.04','69.12','481/560','480/560']:
        if token not in zh or token not in en:errors.append('Bilingual final-test token missing: '+token)
    result=dict(status='PASS' if not errors else 'FAIL',mode='links_and_frozen_endpoints' if args.links_only else 'initial_handoff_snapshot',
        files=len(files),local_markdown_links_checked=checked_links,potential_credential_files=secrets,errors=errors,
        scope='Local navigation, bytes and frozen headline endpoints; not scientific peer review, model inference, or external URL availability.')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:sys.exit(1)


if __name__=='__main__':main()
