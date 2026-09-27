"""Bounded-memory continuous HTTP reads with source-bound resume offsets."""
import os
import re
import shutil
import time
import urllib.request

def install(w):
    def fetch(item):
        target=w.ROOT/item['destination']/item['path']
        if target.is_symlink():
            raise RuntimeError('target symlink')
        if target.exists():
            return {'path':item['path'],'bytes':item['size'],**w.verify(target,item),'status':'verified_existing'}
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_name(target.name+'.partial-20260927-001')
        if partial.is_symlink():
            raise RuntimeError('partial symlink')
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>item['size']:
            raise RuntimeError('oversized partial')
        if shutil.disk_usage(w.ROOT).free<item['size']-offset+w.RESERVE:
            raise RuntimeError('insufficient reserve')
        started=time.monotonic()
        previous_progress=offset//(256*1024*1024)
        for attempt in range(12):
            if offset==item['size']:
                break
            try:
                request=urllib.request.Request(item['url'],headers={'User-Agent':'rosetta-direct-staging/3','Range':f'bytes={offset}-'})
                with urllib.request.urlopen(request,timeout=90) as response:
                    if response.status==206:
                        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                        if not match:
                            raise RuntimeError('missing Content-Range')
                        first,last,total=map(int,match.groups())
                        if first!=offset or total!=item['size'] or last<first or last>=total:
                            raise RuntimeError('range offset or total mismatch')
                        response_end=last+1
                    elif response.status==200 and offset==0:
                        response_end=item['size']
                    else:
                        raise RuntimeError('resume refused; partial preserved')
                    with partial.open('ab' if offset else 'xb') as output:
                        while block:=response.read(8*1024*1024):
                            if offset+len(block)>response_end:
                                raise RuntimeError('response exceeded declared source size')
                            if shutil.disk_usage(w.ROOT).free<len(block)+w.RESERVE:
                                raise RuntimeError('capacity reserve reached')
                            output.write(block)
                            offset+=len(block)
                            progress=offset//(256*1024*1024)
                            if progress>previous_progress:
                                previous_progress=progress
                                w.emit(status='progress',dataset=item['dataset'],path=item['path'],bytes=offset,size=item['size'],elapsed=round(time.monotonic()-started,2))
                if offset!=response_end:
                    raise RuntimeError('short response; partial preserved')
            except Exception as exc:
                # Flush/close has completed before retry; the on-disk offset is authoritative.
                offset=partial.stat().st_size if partial.exists() else 0
                w.emit(status='retry',dataset=item['dataset'],path=item['path'],offset=offset,attempt=attempt+1,error_type=type(exc).__name__)
                if attempt==11:
                    raise
                time.sleep(min(30,3*(attempt+1)))
        if offset!=item['size']:
            raise RuntimeError('incomplete file after bounded retries')
        result=w.verify(partial,item)
        os.link(partial,target)
        partial.unlink()
        return {'path':item['path'],'bytes':item['size'],**result,'status':'downloaded'}
    w.fetch=fetch
