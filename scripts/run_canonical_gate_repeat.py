"""New execution namespace for an authorized repeat, with unchanged Gate criteria."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
NAME = "canonical-fullframes-posttrain-20260916-005"


def main():
    from scripts import canonical_fullframes_runtime as runtime

    runtime.NAME = NAME
    module_name, *arguments = sys.argv[1:]
    allowed = {
        "scripts.run_canonical_fullframes_posttrain",
        "scripts.canonical_fullframes_runtime",
        "scripts.canonical_fullframes_stages",
        "scripts.canonical_fullframes_offline",
        "scripts.canonical_fullframes_sim_gate",
    }
    if module_name not in allowed:
        raise ValueError("Unregistered repeat stage")
    original_popen = subprocess.Popen

    def launch(args, *positional, **kwargs):
        args = list(args)
        if len(args) > 2 and args[0] == sys.executable:
            if args[1] == "-m" and args[2] in allowed:
                args = [args[0], __file__, *args[2:]]
            elif Path(args[1]).name == "run_canonical_fullframes_posttrain.py":
                args = [args[0], __file__, "scripts.run_canonical_fullframes_posttrain", *args[2:]]
            elif args[1:3] == ["-m", "pytest"]:
                args += ["tests/test_rollout_trace.py"]
        return original_popen(args, *positional, **kwargs)

    subprocess.Popen = launch
    module = importlib.import_module(module_name)
    if module_name == "scripts.run_canonical_fullframes_posttrain":
        module.NAME = NAME
    if module_name == "scripts.canonical_fullframes_sim_gate":
        from scripts import smolvla_sim_gate as engine
        from rosetta_reality.eval.rollout_trace import run_traced_rollout
        from rosetta_reality.eval.rollout_trace_verify import verify_trace

        original_rollout = engine._rollout

        def traced(policy, contract, instruction, **options):
            plan, job, reg = runtime.active()
            output = job / "traces" / (arguments[0] + "-" + str(options["seed"]))
            identity = {"kind": "registered_model", "run_id": NAME,
                        "sources": reg["sources"], "endpoint": plan["endpoint"],
                        "optimizer_steps": 0, "original_protocol": plan["gate_protocol_authority"]}
            engine._rollout = original_rollout
            try:
                result = run_traced_rollout(engine, policy, contract, instruction,
                    output=output, identity=identity, episode_index=options["seed"], **options)
                verify_trace(output)
                return result
            finally:
                engine._rollout = traced

        engine._rollout = traced
    sys.argv = [module.__file__, *arguments]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
