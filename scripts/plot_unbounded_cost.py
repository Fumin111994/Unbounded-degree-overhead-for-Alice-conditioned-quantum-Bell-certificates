"""Plot rigorous exact-degree bounds separately from numerical saturation."""
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
fig,ax=plt.subplots(figsize=(3.4,2.8),layout='constrained')
# d_os >= k+1 for epsilon<t_k. Points at jumps are excluded on the
# stronger side; the staircase has no filled endpoint markers.
k=np.arange(1,501,dtype=np.int64)
t=4*(4*k*k+1)/(8*k**4+8*k*k+1)
ax.step(t,k+1,where='post',color='#7b3294',lw=1.5,
        label='Conditioned: exact-degree lower bound')
ax.axhline(2,color='#246a9b',lw=1.5,label='Standard: exact degree 2')
ax.plot([.5,.7],[3,3],color='#22834c',lw=4,solid_capstyle='butt',
        label='Conditioned: exact degree 3')
e=np.array([r['epsilon'] for r in data['summary']])
d=np.array([r['numerical_saturation_level'] for r in data['summary']])
ax.plot(e,d,'o',ms=4,mfc='white',mec='#c36713',mew=1.1,zorder=5,
        label=r'Numerical saturation ($10^{-6}$)')
fitx=np.geomspace(e.min(),e.max(),150)
ax.plot(fitx,data['log_log_fit_prefactor']*fitx**data['log_log_fit_slope'],
        '--',lw=.9,color='#c36713',alpha=.8,zorder=2)
ax.annotate(r'fit: $2.23\epsilon^{-0.492}$',xy=(.04,11),xytext=(.035,30),
            ha='center',fontsize=7,color='#a9570b',
            arrowprops={'arrowstyle':'-','lw':.7,'color':'#c36713'})
ax.set_xscale('log');ax.set_yscale('log',base=2)
ax.set_xlim(.9,1e-4);ax.set_ylim(1.7,165)
ax.set_yticks([2,4,8,16,32,64,128]);ax.set_yticklabels(['2','4','8','16','32','64','128'])
ax.set_xlabel(r'Distance to the endpoint, $\epsilon=2-\alpha$')
ax.set_ylabel('Native certificate level')
ax.grid(True,which='major',alpha=.17,lw=.5)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',frameon=False,handlelength=2)
fig.savefig(ROOT/'figures/prl_unbounded_cost.pdf')
fig.savefig(ROOT/'artifacts/prl/fejer/degree_figure.png',dpi=220)
