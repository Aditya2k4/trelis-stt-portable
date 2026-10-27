"""Download the public Trelis release, verify its parts, and extract a full package."""
import argparse,concurrent.futures,hashlib,json,os,re,shutil,sys,time,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):digest.update(data)
    return digest.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-dir',type=Path,default=ROOT/'downloads')
    p.add_argument('--destination',type=Path,default=ROOT/'package')
    p.add_argument('--no-download',action='store_true',help='Use already downloaded local release parts')
    p.add_argument('--verify-only',action='store_true',help='Verify parts and reconstructed SHA256 without writing/extracting the ZIP')
    p.add_argument('--workers',type=int,choices=range(1,5),default=2)
    a=p.parse_args();manifest=json.loads((ROOT/'release-manifest.json').read_text(encoding='utf-8'))
    repository=manifest['repository'];tag=manifest['tag']
    if not repository or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):raise SystemExit('Repository identity missing or invalid in release manifest')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',tag):raise SystemExit('Invalid release tag')
    a.assets_dir=a.assets_dir.resolve();a.destination=a.destination.resolve()
    a.assets_dir.mkdir(parents=True,exist_ok=True)
    for item in manifest['parts']:
        if not re.fullmatch(r'Trelis-STT-FULL\.zip\.part\d{3}',item['name']):raise ValueError('Invalid part filename')
    def obtain(item):
        path=a.assets_dir/item['name']
        if path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']:
            print('Verified existing '+item['name'],flush=True);return path
        if a.no_download:raise SystemExit('Missing or mismatched local part: '+str(path))
        temp=path.with_suffix(path.suffix+'.download')
        url=f"https://github.com/{repository}/releases/download/{tag}/{item['name']}"
        for attempt in range(4):
            try:
                offset=temp.stat().st_size if temp.exists() else 0
                if offset>=item['bytes']:
                    if offset==item['bytes'] and sha(temp)==item['sha256']:
                        temp.replace(path);return path
                    temp.unlink();offset=0
                headers={'User-Agent':'Trelis-Public-Downloader'}
                if offset:headers['Range']=f'bytes={offset}-'
                req=urllib.request.Request(url,headers=headers)
                with urllib.request.urlopen(req,timeout=120) as response:
                    append=offset>0 and response.status==206
                    if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Server returned an unexpected byte range')
                    with temp.open('ab' if append else 'wb') as output:
                        shutil.copyfileobj(response,output,1024*1024)
                if temp.stat().st_size!=item['bytes'] or sha(temp)!=item['sha256']:
                    temp.unlink();raise RuntimeError('Downloaded part failed size/SHA256 verification')
                temp.replace(path);print('Downloaded and verified '+item['name'],flush=True);return path
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
        raise RuntimeError('Download did not complete')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(obtain,manifest['parts']))
    archive=a.assets_dir/manifest['archive_name']
    if archive.name!='Trelis-STT-FULL.zip':raise ValueError('Unexpected archive filename')
    digest=hashlib.sha256();total=0
    if a.verify_only:
        out=None
    else:
        out=archive.with_suffix('.zip.assembling').open('wb')
    try:
        for item in manifest['parts']:
            with (a.assets_dir/item['name']).open('rb') as f:
                for data in iter(lambda:f.read(4*1024*1024),b''):
                    digest.update(data);total+=len(data)
                    if out:out.write(data)
    finally:
        if out:out.close()
    if total!=manifest['archive_bytes'] or digest.hexdigest()!=manifest['archive_sha256']:
        raise RuntimeError('Reassembled full archive did not match the release checksum')
    print('Full package SHA256 verified: '+digest.hexdigest(),flush=True)
    if a.verify_only:return
    archive.with_suffix('.zip.assembling').replace(archive)
    a.destination.mkdir(parents=True,exist_ok=True)
    target=a.destination/manifest['extracted_folder']
    if target.exists():raise SystemExit('Archive verified. Extraction destination already exists; preserve it and choose --destination with a new folder, or use the existing installation.')
    # Stream ZIP64 entries; avoid PowerShell Expand-Archive size limitations.
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            relative=Path(member.filename)
            destination=(a.destination/relative).resolve()
            if relative.is_absolute() or not destination.is_relative_to(a.destination):raise ValueError('Unsafe archive path')
            if not relative.parts or relative.parts[0]!=manifest['extracted_folder']:raise ValueError('Unexpected archive layout')
            if member.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as source,destination.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
    import subprocess
    subprocess.run([sys.executable,str(target/'verify_package.py')],check=True)
    print('\nReady: '+str(target),flush=True)
    print('Read START HERE.txt and CHATGPT HANDOFF.txt in that folder.')
    print('Next: powershell -NoProfile -ExecutionPolicy Bypass -File "'+str(target/'setup-windows.ps1')+'"')
if __name__=='__main__':main()
