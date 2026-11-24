#!/usr/bin/env python3
# Usage: tools/detach_run.py RUNDIR path/to/overlap.bin   (log goes to RUNDIR/log.txt)
# Survives the parent shell/session exiting (nohup alone does not); caffeinate -i blocks idle sleep (macOS).
# Run overlap.bin in RUNDIR fully detached (new session, no controlling tty), kept awake.
import os, sys
rundir, binary = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
if os.fork(): sys.exit(0)
os.setsid()
if os.fork(): os._exit(0)
os.chdir(rundir)
fd = os.open('log.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.dup2(os.open('/dev/null', os.O_RDONLY), 0)
os.execvp('caffeinate', ['caffeinate', '-i', '/usr/bin/time', '-p', binary])
