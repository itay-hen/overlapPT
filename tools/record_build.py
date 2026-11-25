#!/usr/bin/env python3
"""Write BUILD/build_metadata.json for a binary built by build_overlap.sh.

Usage: tools/record_build.py BUILD_DIR SRC_DIR -- COMPILER ARGS...

Records the exact compiler command (as run, MPI or serial), the version of that compiler,
SHA-256 of every source in BUILD_DIR (including the generated hamiltonian*.hpp) and of
overlap.bin, the git revision of SRC_DIR and whether its tree was clean. Paths in the command
are written as <MODEL>/... and <SRC>/... so the manifest carries no user-specific prefix.
tools/detach_run.py copies the manifest whose binary hash matches into each run directory.
"""
import datetime, hashlib, json, os, platform, subprocess, sys
from pathlib import Path

b, src = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
cmd = sys.argv[sys.argv.index('--') + 1:]
base = b.parent

def strip(a):
    """Replace the model and source prefixes by <MODEL> and <SRC> (no user-specific paths)."""
    return a.replace(str(base), '<MODEL>').replace(str(src), '<SRC>')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(args, cwd=None):
    try:
        return subprocess.check_output(args, text=True, cwd=cwd, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f'unavailable ({type(e).__name__})'

srcs = {p.name: sha(p) for p in sorted(list(b.glob('*.hpp')) + list(b.glob('*.cpp')))}
d = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command': [strip(a) for a in cmd],
    'compiler': cmd[0],
    'compiler_version': run([cmd[0], '--version']).splitlines()[0],
    'git_revision': run(['git', 'rev-parse', 'HEAD'], cwd=src),
    'git_tree_clean': run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=src) == '',
    'platform': platform.platform(),
    'sources_sha256': srcs,
    'binary_sha256': sha(b / 'overlap.bin'),
}
(b / 'build_metadata.json').write_text(json.dumps(d, indent=2) + '\n')
print('Build command, compiler, git revision and hashes recorded in build_metadata.json')
