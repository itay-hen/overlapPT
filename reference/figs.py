import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':8,'font.family':'serif'})

def grid(ax,L,cuts=(),lw=.8):
    for y in range(L):
        for x in range(L):
            if x+1<L and not any(c=='v' and x+1==p for c,p in cuts): ax.plot([x,x+1],[y,y],'-',c='#4a6fa5',lw=lw,zorder=1)
            if y+1<L and not any(c=='h' and y+1==p for c,p in cuts): ax.plot([x,x],[y,y+1],'-',c='#4a6fa5',lw=lw,zorder=1)
    ax.plot(*np.meshgrid(range(L),range(L)),'o',c='#2c3e50',ms=2.2,zorder=2)
    ax.set_aspect('equal'); ax.axis('off')

# --- Fig: severed.pdf  (L=8 -> four 4x4) ---
fig,ax=plt.subplots(figsize=(3.2,3.2))
grid(ax,8,cuts=(('v',4),('h',4)))
ax.plot([3.5,3.5],[-.6,7.6],'--',c='k',lw=1.3); ax.plot([-.6,7.6],[3.5,3.5],'--',c='k',lw=1.3)
ax.plot([1.5,1.5],[3.7,7.6],':',c='#e07b39',lw=1.3); ax.plot([-.6,3.3],[5.5,5.5],':',c='#e07b39',lw=1.3)
fig.tight_layout(pad=.1); fig.savefig('severed.pdf'); plt.close(fig)

# --- Fig: severed3.pdf  four stages ---
fig,axs=plt.subplots(1,4,figsize=(7.0,1.95))
stages=[(('v',2),('v',4),('v',6),('h',2),('h',4),('h',6)),
        (('v',2),('v',6),('h',2),('h',6)),
        (('v',4),('h',4)), ()]
lab=[r'(a) $L^{(3)}$',r'(b) $L^{(2)}$',r'(c) $L^{(1)}$',r'(d) $L^{(0)}$']
for ax,st,t in zip(axs,stages,lab):
    grid(ax,8,cuts=st); ax.set_title(t,fontsize=8,pad=2)
fig.tight_layout(pad=.2); fig.savefig('severed3.pdf'); plt.close(fig)

# --- Legacy ED panel (superseded): the manuscript's ed_scaling.pdf is written by fig2.py ---
Lx=[4,6,8,10,12,14,16]
d1=[0.1155,0.8252,0.1952,0.3769,0.2602,0.4979,0.3214]
d2=[0.2918,0.3900,0.4654]; d3=[0.5416,0.0803]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.5))
axs[0].plot(Lx,d1,'o-',ms=4,label=r'$\ell_{\rm cut}=1$')
axs[0].plot(Lx[:3],d2,'s-',ms=4,label=r'$\ell_{\rm cut}=2$')
axs[0].plot(Lx[:2],d3,'^-',ms=4,label=r'$\ell_{\rm cut}=3$')
axs[0].set_xlabel(r'bulk length $L_x$'); axs[0].set_ylabel(r'$-\ln|\langle g_1|g_2\rangle|^2$')
axs[0].set_ylim(0,1.0); axs[0].legend(frameon=False,fontsize=7)
axs[0].set_title('flat in bulk size',fontsize=8)
m=[np.mean(d1),np.mean(d2),np.mean(d3)]; s=[np.std(d1),np.std(d2),np.std(d3)]
axs[1].errorbar([1,2,3],m,yerr=s,fmt='o',ms=5,capsize=3,color='#b03a2e')
c=np.polyfit([1,2,3],m,1)
axs[1].plot([0.5,3.5],np.polyval(c,[0.5,3.5]),'--',c='gray',lw=1)
axs[1].set_xlabel(r'cut length $\ell_{\rm cut}$ (bonds removed)')
axs[1].set_ylabel(r'$\overline{-\ln|\langle g_1|g_2\rangle|^2}$')
axs[1].set_xticks([1,2,3]); axs[1].set_title('grows with cut length',fontsize=8)
fig.tight_layout(pad=.3); fig.savefig('ed_scaling_legacy.pdf'); plt.close(fig)
print("ok")
