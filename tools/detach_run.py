#!/usr/bin/env python3
# Usage: tools/detach_run.py RUNDIR path/to/overlap.bin [--seed N]
# Runs overlap.bin in RUNDIR fully detached (new session, no controlling tty), kept awake:
# survives the parent shell/session exiting (nohup alone does not); caffeinate -i blocks idle
# sleep (macOS).
#
# Provenance: before launching, appends a record to RUNDIR/run_manifest.json with the binary's
# SHA-256 and the build manifest whose binary hash matches (build_metadata.json next to the
# binary, in its build/ subdirectory, or in ../build/; "unmatched" otherwise). The program itself appends its
# RNG seed to RUNDIR/run_seed_r<rank>.txt. --seed N sets OVERLAP_SEED for a replayable run.
#
# Log: log.txt in RUNDIR, opened in append mode so a resumed run keeps the earlier output.
import datetime, hashlib, json, os, sys

args = sys.argv[1:]
seed = None
if '--seed' in args:
    i = args.index('--seed'); seed = args[i + 1]; del args[i:i + 2]
rundir, binary = os.path.abspath(args[0]), os.path.abspath(args[1])

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

bsha = sha(binary)
meta = 'unmatched'
for cand in (os.path.join(os.path.dirname(binary), 'build_metadata.json'),
             os.path.join(os.path.dirname(binary), 'build', 'build_metadata.json'),
             os.path.join(os.path.dirname(binary), '..', 'build', 'build_metadata.json')):
    if os.path.exists(cand):
        m = json.load(open(cand))
        if m.get('binary_sha256') == bsha:
            meta = m; break
mpath = os.path.join(rundir, 'run_manifest.json')
recs = json.load(open(mpath)) if os.path.exists(mpath) else []
recs.append({'launch_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'binary': os.path.basename(binary), 'binary_sha256': bsha,
             'resumed': os.path.exists(os.path.join(rundir, 'log.txt')),
             'overlap_seed': seed, 'build': meta})
with open(mpath, 'w') as f:
    json.dump(recs, f, indent=2); f.write('\n')

if os.fork(): sys.exit(0)
os.setsid()
if os.fork(): os._exit(0)
os.chdir(rundir)
if seed is not None:
    os.environ['OVERLAP_SEED'] = seed
fd = os.open('log.txt', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.dup2(os.open('/dev/null', os.O_RDONLY), 0)
os.execvp('caffeinate', ['caffeinate', '-i', '/usr/bin/time', '-p', binary])
