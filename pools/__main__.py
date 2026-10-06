"""Pool CLI.

    python -m pools build --panel data/processed/cn_csi500.npz --out runs/main [--scale 1.0] [--search]
                          [--authors configs/models.yaml]
    python -m pools import-p4 --framework alphaagent --path logs.json --out runs/main/p4.jsonl
    python -m pools canary --in runs/main/formulas.jsonl --out release/formulas.jsonl
    python -m pools check-library
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--panel", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--search", action="store_true", help="run GitHub / infini-gram novelty searches")
    b.add_argument("--authors", help="models YAML; models with role 'author' run P1/P2")
    b.add_argument("--windows")
    b.add_argument("--fast", action="store_true")
    i = sub.add_parser("import-p4")
    i.add_argument("--framework", default="generic", choices=["alphaagent", "rdagent", "generic", "alpha-r1"])
    i.add_argument("--path", required=True)
    i.add_argument("--out", required=True)
    c = sub.add_parser("canary")
    c.add_argument("--in", dest="inp", required=True)
    c.add_argument("--out", required=True)
    sub.add_parser("check-library")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from verify.__main__ import _ctx

        from .build import build_all

        ctx = _ctx(a.panel, a.fast, a.windows)
        authors = None
        if a.authors:
            from configs import models
            from narrate.clients import get_client

            authors = [get_client(m) for m in models(a.authors)["models"] if "author" in m.get("roles", [])]
        rep = build_all(ctx, a.out, a.scale, a.search, authors)
        print(json.dumps(rep["counts"], indent=1))
    elif a.cmd == "import-p4":
        from .in_the_wild import import_alpha_r1_profiles, import_framework_outputs
        from .records import write_jsonl

        recs, rej = (import_alpha_r1_profiles(a.path) if a.framework == "alpha-r1"
                     else import_framework_outputs(a.path, a.framework))
        write_jsonl(recs, a.out)
        print(f"imported {len(recs)} formulas, rejected {len(rej)}")
        if rej:
            print(json.dumps(rej[:10], indent=1))
    elif a.cmd == "canary":
        from .canary import embed_canary

        print(f"wrote {embed_canary(a.inp, a.out)} records with canary")
    elif a.cmd == "check-library":
        from .library import base_set, library, verify_transcriptions

        bad = verify_transcriptions()
        print(f"{len(library())} formulas, {len(base_set())} in the base set, {len(bad)} unparsable")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
