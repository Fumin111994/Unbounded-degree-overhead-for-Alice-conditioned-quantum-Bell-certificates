"""The two proved square-root degree laws; no asymmetric numerical fit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,
                     'pdf.fonttype':42,'ps.fonttype':42})
data=json.loads((ROOT/'artifacts/prl/fejer/numerical_saturation.json').read_text())
if not data['complete']:raise RuntimeError('Incomplete numerical survey')
fig,(ax,precision)=plt.subplots(1,2,figsize=(7.0,2.85),layout='constrained')
ax.set_title('(a) Orientation and exact degree',fontsize=8,loc='left')
# d_os >= k+1 for epsilon<t_k. Points at jumps are excluded on the
# stronger side; the staircase has no filled endpoint markers.
k=np.arange(1,501,dtype=np.int64)
t=4*(4*k*k+1)/(8*k**4+8*k*k+1)
ax.step(t,k+1,where='post',color='#7b3294',lw=1.5,
        label='Exact-degree lower bound')
epsilon=np.geomspace(1e-4,2-6/np.sqrt(23),250)
ax.plot(epsilon,12/np.sqrt(epsilon)+3,color='#346f91',lw=1.5,
        label=r'Proved upper bound: $12/\sqrt{\epsilon}+3$')
# The stronger endpoint estimate is only drawn over its proved range.
epsilon_endpoint=np.geomspace(1e-4,2-30/np.sqrt(287),120)
ax.plot(epsilon_endpoint,9/np.sqrt(epsilon_endpoint)+3,color='#346f91',
        lw=1.1,ls=':',label=r'Endpoint refinement: $9/\sqrt{\epsilon}+3$')
ax.axhline(2,color='#246a9b',lw=1.5,label='Standard: exact degree 2')
ax.axhline(1,color='#9b3c36',lw=1.7,ls='--',
           label='Bob tilt: conditioned degree 1')
ax.plot([.5,.7],[3,3],color='#22834c',lw=4,solid_capstyle='butt',
        label='Conditioned: exact degree 3')
e=np.array([r['epsilon'] for r in data['summary']])
d=np.array([r['numerical_saturation_level'] for r in data['summary']])
ax.plot(e,d,'o',ms=4,mfc='white',mec='#c36713',mew=1.1,zorder=5,
        label=r'Numerical saturation ($10^{-6}$)')
ax.set_xscale('log');ax.set_yscale('log',base=2)
ax.set_xlim(.9,1e-4);ax.set_ylim(.8,16384)
ax.set_yticks([1,2,8,32,128,512,2048,8192])
ax.set_yticklabels(['1','2','8','32','128','512','2048','8192'])
ax.set_xlabel(r'Distance to the endpoint, $\epsilon=2-\alpha$')
ax.set_ylabel('Native certificate level')
ax.grid(True,which='major',alpha=.17,lw=.5)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',frameon=False,handlelength=1.8,fontsize=6.2,
          labelspacing=.28)
ax.text(.97,.14,r'Alice tilt: $d_{\rm os}=\Theta(\epsilon^{-1/2})$',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=7,
        bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})


# The asymmetric panel uses only proved analytic bounds and the exact threshold.
delta_star=np.sqrt(5)/2-1
delta=np.geomspace(1e-4,delta_star,300)
lam=1+delta
g=1-1/(lam*lam)
lower=(1+np.sqrt((lam*lam+1)/(lam*lam-1)))/2
upper=15/(2*np.sqrt(g))+3
precision.plot(delta,lower,color='#7b3294',lw=1.6,label='Proved Markov lower bound')
precision.plot(delta,upper,color='#346f91',lw=1.5,
               label=r'Proved upper bound: $15/(2\sqrt{g})+3$')
precision.axvspan(delta_star,.4,color='#22834c',alpha=.10)
precision.plot([delta_star,.4],[2,2],color='#22834c',lw=3,
               label=r'Exactly $K_*=2$: $\lambda\geq\sqrt{5}/2$')
precision.axvline(delta_star,ymax=.55,color='#22834c',ls=':',lw=1)
precision.set_xscale('log');precision.set_yscale('log',base=2)
precision.set_xlim(.4,1e-4);precision.set_ylim(.8,16384)
precision.set_yticks([1,2,8,32,128,512,2048,8192])
precision.set_yticklabels(['1','2','8','32','128','512','2048','8192'])
precision.set_xlabel(r'Distance to symmetric weights, $\lambda-1$')
precision.set_ylabel(r'Uniform exact level $K_*(\lambda)$')
precision.set_title('(b) Contact separation and uniform degree',fontsize=8,loc='left')
precision.grid(True,which='major',alpha=.17,lw=.5)
precision.spines[['top','right']].set_visible(False)
precision.legend(loc='upper left',frameon=False,fontsize=6.7)
precision.text(.97,.68,r'$K_*=\Theta((\lambda-1)^{-1/2})$'+'\n'+r'$g=1-\lambda^{-2}$',
               transform=precision.transAxes,ha='right',fontsize=7,
               bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
fig.savefig(ROOT/'figures/prl_contact_costs.pdf')
fig.savefig(ROOT/'artifacts/prl/contact_transition/degree_costs.png',dpi=220)
