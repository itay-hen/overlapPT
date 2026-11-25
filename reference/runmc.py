"""Reference PT--MBAR runs on the four-site ring, beta = 4, nine-ensemble ladder (paper Sec.
"End-to-end validation", Table tab:mc, Fig. 3).  Usage: python3 runmc.py NSWEEP SEED[,SEED...] [BETA]   (BETA defaults to 4; Table tab:beta uses 2, 4, 6)
Writes res_<seed>.npz per run in the current directory; fig_mc.py aggregates them."""
import numpy as np, time, sys
from scipy.linalg import expm
import ptmbar as P
beta=float(sys.argv[3]) if len(sys.argv)>3 else 4.0; grid=[0.0,0.25,0.5,0.75,1.0]
ladder=[(0.0,lb) for lb in grid]+[(la,1.0) for la in grid[1:]]
i00,i01,i11=0,4,len(ladder)-1
plus=np.ones(P.DIM)/np.sqrt(P.DIM); Aex={l:expm(-beta*P.H_dense(l)/2) for l in grid}
O=lambda a,b: float(plus@Aex[a]@Aex[b]@plus)
f_ex=np.array([-np.log(O(a,b)/O(0.,0.)) for a,b in ladder])
F_ex=O(0.,1.)**2/(O(0.,0.)*O(1.,1.))
ns=int(sys.argv[1]); seeds=[int(x) for x in sys.argv[2].split(',')]
Fs=[]; errs=[]
for sd in seeds:
    t=time.time()
    samples,acc,rt,qser=P.run_pt(ladder,beta,nsweep=ns,nburn=ns//10,stride=60,seed=sd)
    idx=np.array([s[0] for s in samples]); cfg=[s[1] for s in samples]
    u=np.empty((len(ladder),len(cfg)))
    for n,c in enumerate(cfg):
        y=P.states(c)
        for k,(la,lb) in enumerate(ladder): u[k,n]=-P.logW(c,la,lb,beta,y=y)
    Nc=np.array([(idx==k).sum() for k in range(len(ladder))])
    f=P.mbar(u,Nc); lnF=-(2*f[i01]-f[i00]-f[i11]); Fs.append(np.exp(lnF)); errs.append(f-f_ex)
    cvec=np.zeros(len(ladder)); cvec[i00]+=1; cvec[i01]-=2; cvec[i11]+=1
    try:
        Th=P.mbar_cov(u,Nc,f); sd_lnF=float(np.sqrt(max(cvec@Th@cvec,0.0)))
    except Exception as e:
        sd_lnF=float('nan')
    tau=P.tau_int(qser)
    # raw data kept so every number in Table tab:mc and Fig. 3 can be recomputed (fig_mc.py):
    # u[k, n] = -ln W_k of stored configuration n, idx[n] = rung it was drawn at (time order).
    np.savez_compressed(f'res_{sd}.npz',F=Fs[-1],lnF=lnF,sd_lnF=sd_lnF,f=f,f_ex=f_ex,F_ex=F_ex,
             acc=acc,ns=ns,rt=rt,tau=tau,q=np.array([len(c.G) for c in cfg]),nconf=len(cfg),
             u=u,idx=idx,qser=qser,seed=sd,ladder=np.array(ladder))
    print(f'   lnF={lnF:+.5f} (MBAR sd {sd_lnF:.5f})  roundtrips={rt}  tau_int(q)={tau:.2f} strides',flush=True)
    q=np.array([len(c.G) for c in cfg])
    print(f" seed {sd}: F={Fs[-1]:.5f}  max|f-f_ex|={np.abs(f-f_ex).max():.4f}  "
          f"swap={acc.min():.2f}-{acc.max():.2f}  <q>={q.mean():.2f}  [{time.time()-t:.0f}s]",flush=True)
Fs=np.array(Fs)
print(f"\nF_beta  PT-MBAR = {Fs.mean():.5f} +/- {Fs.std(ddof=1)/np.sqrt(len(Fs)):.5f}  ({len(Fs)} runs)")
print(f"F_beta  exact   = {F_ex:.5f}")
print(f"deviation = {(Fs.mean()-F_ex)/F_ex*100:+.2f}%")
print("mean |f-f_ex| per ensemble:",np.round(np.abs(np.mean(errs,axis=0)),4))
