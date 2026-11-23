"""Bounded isolated D train-only native view build / independent native read."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rosetta_reality.diagnostics.d_train_native_view import build,verify_native_read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("build","verify"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--export-root",type=Path)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--normalization",type=Path)
    p.add_argument("--episodes",type=int,nargs="+")
    p.add_argument("--maximum-seconds",type=int,default=600)
    p.add_argument("--maximum-bytes",type=int,default=256*1024**2)
    p.add_argument("--resource-report",type=Path)
    p.add_argument("--proof-output",type=Path)
    p.add_argument("--unmodified-native-action-query",action="store_true")
    args=p.parse_args()
    started=time.monotonic();completed=False
    try:
        if args.mode=="build":
            if any(getattr(args,name) is None for name in ("export_root","source_root","normalization","episodes")):
                p.error("Build requires explicitly bound export/source/statistics/episodes")
            build(args.export_root,args.source_root,args.normalization,args.output,args.episodes,
                  max_seconds=args.maximum_seconds,max_bytes=args.maximum_bytes)
            print(json.dumps({"status":"prepared_unverified_native_read","model_loaded":False,"shutdown_requested":False}))
        else:print(json.dumps(verify_native_read(args.output,proof_output=args.proof_output,
                              action_tensor_cache=not args.unmodified_native_action_query),sort_keys=True))
        completed=True
    finally:
        if args.resource_report:
            import resource
            usage=resource.getrusage(resource.RUSAGE_SELF)
            with args.resource_report.open("x") as stream:
                json.dump({"completed":completed,"elapsed_seconds":time.monotonic()-started,
                    "maximum_rss_bytes":usage.ru_maxrss*1024,"user_cpu_seconds":usage.ru_utime,
                    "system_cpu_seconds":usage.ru_stime,"model_loaded":False,"optimizer_updates":0},stream)


if __name__=="__main__":main()
