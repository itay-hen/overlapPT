from pathlib import Path
import sys,json,hashlib,platform,datetime,subprocess
b=Path(sys.argv[1]); sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in b.glob('*.hpp')}
sources.update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in b.glob('*.cpp')})
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'flags':sys.argv[2:],'sources_sha256':sources,'platform':platform.platform(),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],'binary_sha256':hashlib.sha256((b/'overlap.bin').read_bytes()).hexdigest()}
(b/'build_metadata.json').write_text(json.dumps(d,indent=2)+'\n')
print('Build settings and source hashes recorded in build_metadata.json')
