"""Standard-library-only exact check of the standard degree-two lower bound."""
from pathlib import Path
from fractions import Fraction as F
import copy
import hashlib
import json
from verify_exact_D3_M6 import ball,reduce_word,positive_pivots,verify as verify_upper
HERE=Path(__file__).resolve().parent

def verify(data):
    assert data['degree']==2
    words=[(tuple(a),tuple(b)) for a,b in data['word_basis']]
    assert words==[(a,b) for a in ball(2) for b in ball(2) if len(a)+len(b)<=2]
    moments={(tuple(a),tuple(b)):F(v) for a,b,v in data['moments']}
    assert len(moments)==len(data['moments']) and moments[((),())]==1
    canonical=lambda w:min(w,(w[0][::-1],w[1][::-1]))
    products=[[canonical((reduce_word(a[::-1]+c),reduce_word(b[::-1]+d))) for c,d in words] for a,b in words]
    assert set(moments)=={w for row in products for w in row}
    pivots=positive_pivots([[moments[w] for w in row] for row in products])
    atoms=[((x,),()) for x in range(2)]+[((),(x,)) for x in range(2)]+[((x,),(y,)) for x in range(2) for y in range(2)]
    theta=list(map(F,data['coefficients']));beta=F(data['target']);assert len(theta)==8
    gap=sum(c*moments[w] for c,w in zip(theta,atoms))-beta
    slope=2-moments[((0,),())]-moments[((),(0,))]
    end=gap-slope/F(100000);assert slope>=0 and end>F(15,10000)
    upper=json.loads((HERE/'D3_not_M7_exact.json').read_text())
    assert data['coefficients']==upper['coefficients'] and data['target']==upper['target']
    verify_upper(upper)
    return dict(status='PASS',standard_positive_LDL_pivots=len(pivots),exact_gap=str(gap),
                exact_interval_endpoint_gap=str(end),theorem='d_std(G_t)=3 for 0 <= t <= 1/100000',
                scope='Exact finite degree, no assertion of mixed divergence.')

def main():
    path=HERE/'D2_lower_exact.json';data=json.loads(path.read_text());report=verify(data);bad=copy.deepcopy(data)
    bad['moments'][0][2]='-1'
    try:verify(bad)
    except AssertionError:report['rejected_mutations']=['normalization']
    else:raise AssertionError('Mutation not rejected')
    report['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'D2_lower_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
