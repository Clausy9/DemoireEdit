"""Download the revision-pinned public base components and check their hashes."""
import argparse,concurrent.futures,hashlib,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=Path('configs/base-models.json'))
    parser.add_argument('--output',type=Path,default=Path('weights/base'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('workers must be between 1 and 8')
    from huggingface_hub import hf_hub_download
    manifest=json.loads(args.manifest.read_text())
    jobs=[]
    for model in manifest['models']:
        for file in model['files']:
            if Path(file['name']).is_absolute() or '..' in Path(file['name']).parts:raise ValueError('Unsafe model file path')
            jobs.append((model,file))
    def fetch(job):
        model,file=job
        path=Path(hf_hub_download(model['repo_id'],file['name'],revision=model['revision'],local_dir=args.output/model['repo_id']))
        if path.stat().st_size!=file['size']:raise ValueError('Base component size mismatch')
        if file['sha256']:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=file['sha256']:raise ValueError('Base component SHA-256 mismatch')
        return file['size']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        sizes=list(pool.map(fetch,jobs))
    print(json.dumps({'status':'verified','files':len(sizes),'bytes':sum(sizes)}))

if __name__=='__main__':main()
