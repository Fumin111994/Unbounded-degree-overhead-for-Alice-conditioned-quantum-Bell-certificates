"""Publication figures from proved formulas and verified certificate metadata.

No fitted convergence levels or numerical SDP results enter these figures.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
BLUE, ORANGE, GREEN, INK = '#21618c', '#bb572a', '#237759', '#182b3a'
MUTED, RULE = '#586977', '#cbd4db'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'axes.labelsize':9, 'axes.titlesize':10,
                     'pdf.fonttype':42, 'ps.fonttype':42,
                     'mathtext.fontset':'dejavusans', 'savefig.facecolor':'white'})

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(OUT / (name + '.png'), dpi=220, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)

def resources():
    fig = plt.figure(figsize=(7.25, 4.2))
    ax = fig.add_axes([.02,.04,.96,.93]); ax.axis('off')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    cols = [.015,.355,.695]
    labels = ['Standard proof', 'One global direction', 'Mixed directions']
    formulas = [r'$\sum_i f_i(A,B)^\dagger f_i(A,B)$',
                r'$P\in\mathsf{O}^{A}_k\ \mathrm{or}\ \mathsf{O}^{B}_k$',
                r'$P=P_A+P_B$']
    notes = ['Cross-question factors allowed',
             'Choose the better party once',
             'Choose the party term by term']
    for x,lab,formula,note,color in zip(cols,labels,formulas,notes,[INK,BLUE,GREEN]):
        ax.text(x,.98,lab,ha='left',va='top',weight='bold',color=color,fontsize=10)
        ax.text(x,.855,formula,ha='left',va='center',fontsize=10,color=color)
        ax.text(x,.75,note,ha='left',va='top',fontsize=8.2,color=MUTED)
    ax.text(.015,.62,'PROVED DEGREES AT THE EXACT QUANTUM TARGET',color=INK,
            weight='bold',fontsize=8.7)
    positions=[.015,.405,.575,.755,.93]
    heads=['Bell family','Standard','Alice','Bob','Mixed']
    for x,t in zip(positions,heads):
        ax.text(x,.53,t,ha='left' if x==positions[0] else 'center',weight='bold',fontsize=8.7)
    ax.plot([.015,.99],[.495,.495],color=INK,lw=.8)
    rows=[('Single tilt  (2 questions each)', '2', r'$\Theta(\epsilon^{-1/2})$', '1', '1'),
          ('Coupled tests  (4 questions each)', '2', r'$\Theta_c(\epsilon^{-1/2})$',
           r'$\Theta_c(\epsilon^{-1/2})$', '1'),
          ('Symmetric interval  (2 each)', '3', r'$\geq 8$', r'$\geq 8$', '8')]
    ax.axhspan(.20,.33,xmin=.005,xmax=.995,color='#edf4fa',zorder=-1)
    for y,row in zip([.405,.265,.125],rows):
        for j,(x,t) in enumerate(zip(positions,row)):
            ax.text(x,y,t,ha='left' if j==0 else 'center',va='center',fontsize=8.6,
                    color=INK if j<2 else (GREEN if j==4 else BLUE),
                    weight='bold' if j==4 else 'normal')
        ax.plot([.015,.99],[y-.065,y-.065],color=RULE,lw=.55)
    save(fig,'direction_resources')

def coupling():
    fig,(ax,plot)=plt.subplots(1,2,figsize=(7.25,3.15),gridspec_kw={'width_ratios':[1,1.15]})
    fig.subplots_adjust(left=.02,right=.985,bottom=.18,top=.83,wspace=.32)
    ax.set_xlim(-.45,1.5); ax.set_ylim(-.16,1.12); ax.axis('off')
    ax.set_title('a  Coupled Bell test',loc='left',fontweight='bold',pad=17)
    yy=[.94,.68,.24,-.02]
    apos={i:(0,y) for i,y in enumerate(yy)}; bpos={i:(1,y) for i,y in enumerate(yy)}
    for indices in [(0,1),(2,3)]:
        for i in indices:
            for j in indices:
                negative=i==indices[1] and j==indices[1]
                ax.plot([0,1],[yy[i],yy[j]],color=BLUE,lw=1.2,
                        ls='--' if negative else '-',alpha=.82,zorder=1)
    ax.plot([0,1],[yy[0],yy[2]],color=ORANGE,lw=2.8,zorder=2)
    ax.text(.64,.55,r'$c\epsilon$',color=ORANGE,fontsize=11,weight='bold',
            bbox={'facecolor':'white','edgecolor':'none','pad':1},zorder=5)
    for party,coords in [('A',apos),('B',bpos)]:
        for i,(x,y) in coords.items():
            ax.scatter([x],[y],s=200,c='white',edgecolors=INK,linewidths=1,zorder=4)
            ax.text(x,y,fr'${party}_{i}$',ha='center',va='center',fontsize=8,zorder=5)
    ax.text(-.26,.93,r'$\alpha A_0$',ha='right',va='center',color=BLUE,fontsize=9)
    ax.text(1.26,.24,r'$\alpha B_2$',ha='left',va='center',color=BLUE,fontsize=9)
    ax.text(.5,-.23,'One entangled qubit pair attains the bound',ha='center',fontsize=8,color=MUTED)
    c=np.linspace(.001,.999,600)
    ratio=c*(2-c); lower=np.sqrt(2*(1-c)/(1+2*c-c*c))
    plot.plot(c,ratio,color=BLUE,lw=2,label=r'Independent-resource loss: $c(2-c)$')
    plot.plot(c,lower,color=ORANGE,lw=2,ls='--',label=r'Degree lower prefactor: $L(c)$')
    plot.scatter([.9],[.99],c=BLUE,s=25,zorder=5)
    plot.annotate('0.99 at c = 0.9',xy=(.9,.99),xytext=(.33,1.19),fontsize=8,color=BLUE,
                  arrowprops={'arrowstyle':'-','lw':.8,'color':BLUE})
    plot.set_title('b  Coupling effect and certificate cost',loc='left',fontweight='bold',pad=17)
    plot.set_xlabel(r'Fixed bridge ratio, $c$');plot.set_ylabel('Asymptotic quantity')
    plot.set_xlim(0,1);plot.set_ylim(0,1.5)
    plot.spines[['top','right']].set_visible(False);plot.grid(alpha=.15,lw=.5)
    plot.legend(loc='lower left',fontsize=6.8,frameon=False)
    save(fig,'coupling_and_cost')

def exact():
    # Read the same certificate metadata used by the exact proof.
    data=json.loads((ROOT/'research/mixed_divergence/M8_upper_exact.json').read_text())
    sizes=[len(g) for g in data['Grams']]
    assert sizes==[15,17,16,16] and sum(sizes)==64
    base=json.loads((ROOT/'research/mixed_divergence/D3_not_M7_exact.json').read_text())
    assert len(base['standard_Gram'])==23 and base['mixed_level']==7
    fig,ax=plt.subplots(figsize=(7.25,3.45));fig.subplots_adjust(left=.02,right=.99,bottom=.04,top=.98)
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.02,.965,'ONE ATTAINED TARGET, FOUR EXACT CERTIFICATES',fontsize=10,weight='bold',color=INK)
    ax.text(.36,.82,'Lower bound: a feasible witness',ha='center',color=ORANGE,fontsize=9)
    ax.text(.79,.82,'Upper bound: an SOS identity',ha='center',color=GREEN,fontsize=9)
    for y,name,left,right,ldetail,rdetail,result in [
        (.64,'Standard','Level 2 excluded','Degree 3 reaches target',
         '13 x 13 Gram; 13 positive pivots\nUniform score excess > 0.0015',
         '23 x 23 Gram; 23 positive pivots\nFactors satisfy both deterministic zeros',r'$d_{\rm std}=3$'),
        (.29,'Mixed','Level 7 excluded','Degree 8 reaches target',
         'Eight 15 x 15 Grams; 120 positive pivots\nShared Bell moments; excess > 0.000008',
         'Four Grams + party exchange\nSizes 15, 17, 16, 16; 64 positive pivots',r'$d_{\rm mix}=8$')]:
        ax.text(.02,y,name,color=INK,fontsize=10,weight='bold',va='center')
        ax.text(.02,y-.095,result,color=INK,fontsize=10,va='center')
        ax.text(.36,y,left,ha='center',va='center',color=ORANGE,weight='bold',fontsize=10)
        ax.text(.79,y,right,ha='center',va='center',color=GREEN,weight='bold',fontsize=10)
        ax.text(.36,y-.09,ldetail,ha='center',va='top',fontsize=7.7,linespacing=1.65,color=MUTED)
        ax.text(.79,y-.09,rdetail,ha='center',va='top',fontsize=7.7,linespacing=1.65,color=MUTED)
        ax.add_patch(FancyArrowPatch((.575,y),(.605,y),arrowstyle='->',mutation_scale=10,color=INK,lw=.8))
    ax.plot([.02,.99],[.44,.44],color=RULE,lw=.8)
    ax.text(.5,.015,r'All bounds hold for $0\leq t\leq10^{-5}$; both deterministic strategies attain $\beta_t$.',
            ha='center',fontsize=8,color=INK)
    save(fig,'exact_degree_evidence')

def relative():
    c=1/8
    eps=np.geomspace(1e-5,.1,260)
    def witness(e,k):
        r=1/(8*k*k+2);h=1-2*r
        alpha=2-e;t=alpha+c*e*h;s=e*(1-c*h)
        V=2*e*e/(np.sqrt(8+2*alpha*alpha)+alpha+2)
        D=2*(1-c)*r*e+s*s/(np.sqrt(8+2*t*t)+t+2)-V
        return D/V
    fig,(left,right)=plt.subplots(1,2,figsize=(7.25,3.5))
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.82,wspace=.40)
    for k,color in zip([2,4,8,16],[INK,BLUE,GREEN,ORANGE]):
        ratio=witness(eps,k)
        left.loglog(eps,np.where(ratio>0,ratio,np.nan),color=color,lw=1.5,label=fr'$k={k}$')
    left.axhline(.5,color='#6d7480',lw=1,ls='--')
    left.text(1.4e-5,.65,r'Error budget $\rho=1/2$',fontsize=7.5,color=MUTED)
    left.set(xlabel=r'Endpoint distance, $\epsilon$',ylabel=r'Witness lower bound on $E_k/V$',
             xlim=(1e-5,.1),ylim=(.03,1e5))
    left.set_title('a  Fixed levels lose relative resolution',loc='left',weight='bold',fontsize=9.5,pad=16)
    left.legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    lower=[]
    for e in eps:
        lo=0;hi=int(np.floor(9/np.sqrt(e)+3))
        # The witness score decreases with k: q'(t)<1 and c<1.
        while hi-lo>1:
            k=(lo+hi)//2
            if witness(e,k)>.5:lo=k
            else:hi=k
        lower.append(lo+1)
    lower=np.array(lower)
    upper=np.floor(9/np.sqrt(eps)+3)
    right.fill_between(eps,lower,upper,color='#edf4fa')
    right.loglog(eps,upper,color=BLUE,lw=1.6,label='Exact-certificate upper bound')
    right.loglog(eps,lower,color=ORANGE,lw=1.6,label='Feasible-witness lower bound')
    right.set(xlabel=r'Endpoint distance, $\epsilon$',ylabel=r'Required level, $K_{1/2}$',
              xlim=(1e-5,.1),ylim=(1,5000))
    right.set_title('b  Relative accuracy has a degree cost',loc='left',weight='bold',fontsize=9.5,pad=16)
    right.text(4e-4,100,r'$\Theta(\epsilon^{-1/2})$',fontsize=9.5,color=INK)
    right.legend(frameon=False,fontsize=7.4,loc='upper right')
    for ax in [left,right]:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(which='major',alpha=.15,lw=.5)
    save(fig,'relative_precision')
    # Independent high-precision evaluation checks every plotted threshold.
    from decimal import Decimal, localcontext
    margins=[]
    with localcontext() as ctx:
        ctx.prec=80
        for e,k in zip(eps,lower):
            e=Decimal(str(float(e)));cdec=Decimal(1)/8
            def excess(n):
                r=Decimal(1)/(8*int(n)**2+2);h=1-2*r
                alpha=2-e;t=alpha+cdec*e*h
                q=lambda x:(8+2*x*x).sqrt()
                V=2*e*e/(q(alpha)+alpha+2)
                return 4-h*e+q(t)-(2*q(alpha)+cdec*e)-V/2
            assert excess(int(k))<0
            margins.append(float(abs(excess(int(k)))))
            if k>1:
                assert excess(int(k)-1)>0
                margins.append(float(abs(excess(int(k)-1))))
    report={'status':'PASS','points':len(eps),'precision_digits':80,
            'minimum_threshold_margin':min(margins),'SDP_optimization':False}
    (ROOT/'relative_precision_plot_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    resources();coupling();relative();exact()
    print('Wrote 3 main figures and 1 supplementary certificate figure; relative thresholds checked at 80-digit precision.')
