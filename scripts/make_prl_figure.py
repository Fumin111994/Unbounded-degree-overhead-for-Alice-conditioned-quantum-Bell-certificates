"""Render the Letter figure from stored numerical data and exact certificates."""
import json
from pathlib import Path
from fractions import Fraction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'artifacts/figures/fig1_data.json').read_text())
fan=json.loads((ROOT/'artifacts/m65/tangent_fan.json').read_text())
zoom=json.loads((ROOT/'artifacts/figures/prl_zoom_numerical.json').read_text())['records']
alphas=np.array(data['alphas'])
gap=np.array(data['gap'])
lo,hi=fan['covered_interval_float']
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':6.8,
                     'axes.linewidth':0.65,'lines.linewidth':1.2,'pdf.fonttype':42})
fig=plt.figure(figsize=(3.42,3.85))
grid=fig.add_gridspec(3,1,height_ratios=[1.1,.18,1])
axes=[fig.add_subplot(grid[0]),fig.add_subplot(grid[2])]
closure=fig.add_subplot(grid[1],sharex=axes[0])
for index,axis in enumerate(axes):
    axis.axvspan(lo,hi,color='#D7E7F6',zorder=0)
    axis.axhline(0,color='#666666',lw=.65)
    xx,yy=(alphas,gap) if index==0 else ([r['alpha'] for r in zoom],[r['gap'] for r in zoom])
    axis.plot(xx,yy,'--',color='#454545',label=r'OS$_2$ (numerical)')
    if index == 1:
        axis.axvline(1.28427,color='#7956A5',ls=':',lw=1)
    axis.grid(alpha=.18,lw=.4)
axes[0].plot([1,1.25,41/32],[0,0,0],'D',color='#C86A18',ms=4,
             label=r'OS$_2=q$ (exact)',zorder=5)
axes[0].set(xlim=(.98,2.01),ylim=(-.00035,.0155),ylabel=r'$\omega_2^{\rm os}-q$')
axes[0].set_yticks([0,.005,.010,.015])
axes[0].legend(loc='upper left',bbox_to_anchor=(.10,1.01),frameon=False,handlelength=2)
axes[0].text(.025,.95,'(a)',transform=axes[0].transAxes,va='top',fontweight='bold')
axes[0].text(1.61,.006,'certified\nnon-closure',color='#245D8B',fontsize=7)
axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
# A separate categorical strip: these are level-three closure results,
# never points on the level-two-gap ordinate.
closure.set_ylim(0,1)
closure.set_yticks([])
closure.tick_params(axis='x',bottom=False,labelbottom=False)
for spine in closure.spines.values():
    spine.set_visible(False)
closure.text(.99,.6,'level 3: exact',fontsize=7,color='#21805E',va='center')
closure.plot([1.3,1.5],[.6,.6],color='#21805E',lw=3,solid_capstyle='butt')
closure.plot([1.3,1.5],[.6,.6],'|',color='#21805E',ms=6)
closure.text(1.3,-.10,r'$13/10$',fontsize=6.5,ha='center',color='#21805E')
closure.text(1.5,-.10,r'$3/2$',fontsize=6.5,ha='center',color='#21805E')
# Keep the raw numerical values, including tiny negative solver residuals.
axes[1].plot([1.25,41/32],[0,0],'D',color='#C86A18',ms=4,zorder=6)
axes[1].set(xlim=(1.235,1.312),ylim=(-3e-6,9e-5),
            xlabel=r'tilt $\alpha$',ylabel=r'$\omega_2^{\rm os}-q$')
axes[1].set_yticks([0,2.5e-5,5e-5,7.5e-5])
axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
axes[1].text(.025,.95,'(b)',transform=axes[1].transAxes,va='top',fontweight='bold')
axes[1].annotate(r'$5/4$',(1.25,0),xytext=(1.247,2.0e-5),fontsize=7)
axes[1].annotate(r'$41/32$',(41/32,0),xytext=(1.267,2.0e-5),fontsize=7,
                 arrowprops={'arrowstyle':'-','lw':.55,'color':'#C86A18'})
axes[1].annotate(r'$\alpha^*$ (numerical)',(1.28427,5.5e-5),xytext=(1.248,7.0e-5),
                 fontsize=7,color='#7956A5',arrowprops={'arrowstyle':'->','lw':.6})
fig.subplots_adjust(left=.19,right=.98,bottom=.12,top=.96,hspace=.45)
fig.savefig(ROOT/'figures/prl_quantum_gap.pdf')
(ROOT/'tmp/prl-theorem').mkdir(parents=True,exist_ok=True)
fig.savefig(ROOT/'tmp/prl-theorem/prl_quantum_gap.png',dpi=180)
plt.close(fig)
(ROOT/'artifacts/figures/prl_quantum_gap_data.json').write_text(json.dumps({
    'numerical_source':'artifacts/figures/fig1_data.json',
    'zoom_numerical_source':'artifacts/figures/prl_zoom_numerical.json',
    'numerical_values_modified':False,
    'exact_level_two_points':['1','5/4','41/32'],
    'exact_level_three_points':['13/10','3/2'],
    'exact_level_three_interval':['13/10','3/2'],
    'level_three_display':'separate categorical closure strip',
    'interval_certificate':'artifacts/prl/interval/level3_interval.json',
    'exact_certificate_directory':'artifacts/prl/quantum_face',
    'certified_nonclosure_interval':fan['covered_interval'],
    'candidate_boundary_numerical':1.28427,
    'note':'No interval of level-two exact closure or unique transition is claimed.'},indent=2))
