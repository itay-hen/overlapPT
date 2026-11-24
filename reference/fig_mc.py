import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':8,'font.family':'serif'})
d=np.load('mc_summary.npz'); Fs,fs,fex,Fex=d['Fs'],d['fs'],d['fex'],float(d['Fex'])
lab=['(0,0)','(0,¼)','(0,½)','(0,¾)','(0,1)','(¼,1)','(½,1)','(¾,1)','(1,1)']
x=np.arange(len(fex))
fig,axs=plt.subplots(1,2,figsize=(6.6,2.6))
axs[0].plot(x,fex,'k-',lw=1,label='exact')
axs[0].errorbar(x,fs.mean(0),yerr=fs.std(0,ddof=1),fmt='o',ms=4,capsize=2,
                color='#b03a2e',label='PT--MBAR')
axs[0].set_xticks(x); axs[0].set_xticklabels(lab,rotation=45,fontsize=6)
axs[0].set_ylabel(r'$f_k=-\ln O(\lambda_a,\lambda_b)$  (rel.)')
axs[0].set_xlabel('ladder ensemble'); axs[0].legend(frameon=False,fontsize=7)
axs[0].set_title('(a) free energies along the ladder',fontsize=8)
ax2=axs[1]
ax2.axhline(Fex,color='k',lw=1,label='exact')
ax2.errorbar([0],[Fs.mean()],yerr=[Fs.std(ddof=1)/np.sqrt(len(Fs))],fmt='s',ms=6,
             capsize=3,color='#1e8449',label='mean of 4 runs')
ax2.plot(np.linspace(1,4,4),Fs,'o',ms=4,color='#b03a2e',label='individual runs')
ax2.set_xticks([0,1,2,3,4]); ax2.set_xticklabels(['mean','21','22','23','24'],fontsize=7)
ax2.set_xlabel('seed'); ax2.set_ylabel(r'$F_\beta$')
ax2.legend(frameon=False,fontsize=7); ax2.set_title(r'(b) $F_\beta$ vs exact',fontsize=8)
fig.tight_layout(pad=.3); fig.savefig('mc_validation.pdf'); print('ok')
