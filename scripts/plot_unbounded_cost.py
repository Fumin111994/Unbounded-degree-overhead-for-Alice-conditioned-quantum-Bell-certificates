"""Exact degree bounds and the Bell advantage missed by finite truncation."""
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

# These are analytic physical quantities, not optimal SDP errors.
levels=np.arange(1,65,dtype=np.int64)
r=1/(8*levels.astype(float)**2+2)
alpha=2-8*r
w=4-(1-2*r)*8*r
q=np.sqrt(8+2*alpha*alpha)
lower=64*r*r*(1-2*r)**2/(w+q)
local=2+alpha
# Rationalized differences avoid cancellation near the endpoint.
advantage=(8*r)**2/(q+local)
if not np.all((lower>0)&(advantage>0)):
    raise RuntimeError('Physical scales must be positive')
precision.plot(levels,levels.astype(float)**4*advantage,color='#246a9b',lw=1.5,
               label=r'Quantum violation: $k^4[q-(2+\alpha_k)]$')
precision.plot(levels,levels.astype(float)**4*lower,color='#7b3294',lw=1.5,
               label=r'Witness excess: $k^4(w_k-q)$')
precision.axhline(1/8,ls='--',lw=.9,color='0.45',label=r'Common limit: $1/8$')
precision.set_xscale('log',base=2)
precision.set_xlim(1,64)
precision.set_ylim(.05,.135)
precision.set_xticks([1,2,4,8,16,32,64])
precision.set_xticklabels(['1','2','4','8','16','32','64'])
precision.set_xlabel(r'Conditioned level $k$')
precision.set_ylabel(r'Bell-value difference multiplied by $k^4$')
precision.set_title('(b) Obstruction versus quantum advantage',fontsize=8,loc='left')
precision.grid(True,which='major',alpha=.17,lw=.5)
precision.spines[['top','right']].set_visible(False)
precision.legend(loc='lower left',frameon=False,handlelength=2)
fig.savefig(ROOT/'figures/prl_unbounded_cost.pdf')
fig.savefig(ROOT/'artifacts/prl/fejer/degree_figure.png',dpi=220)
