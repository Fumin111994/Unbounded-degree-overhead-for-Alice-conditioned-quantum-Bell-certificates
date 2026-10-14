"""Independent Fraction-only verification of the finite mixed upper bound."""
from pathlib import Path
from fractions import Fraction as F
import copy
import hashlib
import json
from verify_exact_D3_M6 import reduce_word,positive_pivots,verify as verify_lower

HERE=Path(__file__).resolve().parent

def verify(data):
    k=data['level'];assert isinstance(k,int) and k==8
    assert len(data['basis'])==len(data['Grams'])==4
    got={};pivots=[];word_count=0
    for block,(raw,rawgram) in enumerate(zip(data['basis'],data['Grams'])):
        basis=[]
        for f in raw:
            polynomial={}
            for w,c in f:
                w=tuple(w);assert all(x in (0,1) for x in w)
                assert reduce_word(w)==w and len(w)<=k
                polynomial[w]=polynomial.get(w,F(0))+F(c);word_count+=1
            basis.append(polynomial)
        gram=[[F(x) for x in row] for row in rawgram]
        assert len(gram)==len(basis);pivots.extend(positive_pivots(gram))
        for i,f in enumerate(basis):
            for j,g in enumerate(basis):
                for u,a in f.items():
                    for v,b in g.items():
                        w=reduce_word(u[::-1]+v);c=a*b*gram[i][j]/2
                        for aw,sign in [((),1),((block//2,),(-1)**block)]:
                            # Keep full reduced words, not their star orbits.
                            for term in [(aw,w),(w,aw)]:
                                got[term]=got.get(term,F(0))+sign*c
    theta=list(map(F,data['coefficients']));beta=F(data['target']);assert len(theta)==8
    atoms=[((x,),()) for x in range(2)]+[((),(x,)) for x in range(2)]+[((x,),(y,)) for x in range(2) for y in range(2)]
    goal={((),()):beta};goal.update({w:-c for w,c in zip(atoms,theta)})
    clean=lambda p:{w:c for w,c in p.items() if c}
    assert clean(got)==clean(goal),'Full coefficient identity failed'
    lower=json.loads((HERE/'D3_not_M7_exact.json').read_text())
    assert data['coefficients']==lower['coefficients'] and data['target']==lower['target']
    lower_report=verify_lower(lower)
    assert lower_report['status']=='PASS'
    return dict(status='PASS',mixed_level=k,positive_LDL_pivots=len(pivots),
                outcome_blocks=8,independent_Grams=4,checked_factor_terms=word_count,
                full_noncommutative_identity=True,linked_lower_checks=lower_report['checks'],
                theorem='d_mix(G_t)=8 for every 0 <= t <= 1/100000',
                scope='Exact finite degree; this entire interval has a uniform mixed upper bound.')

def main():
    path=HERE/'M8_upper_exact.json';data=json.loads(path.read_text());report=verify(data)
    rejected=[]
    for kind in ('coefficient','Gram','degree'):
        bad=copy.deepcopy(data)
        if kind=='coefficient':bad['coefficients'][4]=str(F(bad['coefficients'][4])+F(1,10000))
        elif kind=='Gram':bad['Grams'][0][0][0]='-1'
        else:bad['basis'][0][0][0][0]=[0,1]*5
        try:verify(bad)
        except AssertionError:rejected.append(kind)
        else:raise AssertionError('Undetected mutation: '+kind)
    report['rejected_mutations']=rejected
    report['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'M8_upper_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
