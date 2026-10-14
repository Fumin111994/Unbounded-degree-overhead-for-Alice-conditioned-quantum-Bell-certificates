"""Independent exact checker; Python standard library only, no SDP calls."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import copy
import hashlib
import itertools
import json

HERE=Path(__file__).resolve().parent


def reduce_word(w):
    result=[]
    for x in w:
        if result and result[-1]==x:result.pop()
        else:result.append(x)
    return tuple(result)


def ball(k):
    return [()]+[tuple((s+j)%2 for j in range(n)) for n in range(1,k+1) for s in (0,1)]


def positive_pivots(matrix):
    n=len(matrix);assert all(len(row)==n for row in matrix)
    assert all(matrix[i][j]==matrix[j][i] for i in range(n) for j in range(n))
    work=[row[:] for row in matrix];pivots=[]
    for k in range(n):
        pivot=work[k][k];assert pivot>0, f'Nonpositive exact LDL pivot {k}'
        pivots.append(pivot)
        for i in range(k+1,n):
            for j in range(i,n):
                work[i][j]-=work[i][k]*work[k][j]/pivot
                work[j][i]=work[i][j]
    return pivots


def verify(data):
    checks=[]
    def check(ok,label):
        assert ok,label
        checks.append(label)
    check(data['standard_degree']==3 and data['mixed_level'] in (6,7),'standard degree three and specified mixed level')
    theta=list(map(F,data['coefficients']));beta=F(data['target'])
    check(len(theta)==8 and beta>0,'eight Bell coefficients and positive target')
    check(theta[0]==theta[2] and theta[1]==theta[3] and theta[5]==theta[6],
          'exact symmetry under exchange of the two parties')
    atoms=[((x,),()) for x in range(2)]+[((),(y,)) for y in range(2)]
    atoms += [((x,),(y,)) for x in range(2) for y in range(2)]
    goal={((),()):beta}
    for w,c in zip(atoms,theta):goal[w]=-c
    basis=[]
    for entries in data['standard_basis']:
        poly={}
        for u,v,c in entries:
            u,v=tuple(u),tuple(v)
            check(reduce_word(u)==u and reduce_word(v)==v and len(u)+len(v)<=3,'standard factor has native degree at most three')
            poly[u,v]=poly.get((u,v),F(0))+F(c)
        basis.append(poly)
    gram=[[F(v) for v in row] for row in data['standard_Gram']]
    check(len(gram)==len(basis),'standard Gram dimension matches factors')
    pivots=positive_pivots(gram)
    got={}
    for i,f in enumerate(basis):
        for j,g in enumerate(basis):
            for (a,b),c in f.items():
                for (u,v),d in g.items():
                    # Match every reduced word, without quotienting by star.
                    w=(reduce_word(a[::-1]+u),reduce_word(b[::-1]+v))
                    got[w]=got.get(w,F(0))+gram[i][j]*c*d
    clean=lambda p:{w:c for w,c in p.items() if c}
    check(clean(got)==clean(goal),'full noncommutative SOS coefficient identity')
    scores=[]
    for a0,a1,b0,b1 in itertools.product((-1,1),repeat=4):
        p=[a0,a1,b0,b1,a0*b0,a0*b1,a1*b0,a1*b1]
        scores.append(sum(c*v for c,v in zip(theta,p)))
    check(max(scores)==beta,'exact local maximum equals certified quantum upper bound')
    for assignment in data['assignments']:
        av=list(assignment[:2]);bv=list(assignment[2:])
        p=av+bv+[a*b for a in av for b in bv]
        check(sum(c*v for c,v in zip(theta,p))==beta,'stated deterministic strategy attains target')
    k=data['mixed_level'];words=[tuple(w) for w in data['word_basis']]
    check(words==ball(k),'complete reduced word ball at specified mixed level')
    keys=[tuple(w) for w in data['moment_keys']]
    canonical=lambda w:min(reduce_word(w),reduce_word(w)[::-1])
    check(set(keys)=={canonical(u[::-1]+v) for u in words for v in words} and len(keys)==len(set(keys)),
          'all required moments through degree twice the mixed level, without duplicate keys')
    index={w:i for i,w in enumerate(keys)}
    moments=list(map(F,data['shared_Bell_moments']));ma,mb=moments[:2],moments[2:4]
    corr=[moments[4:6],moments[6:8]];ct=[list(v) for v in zip(*corr)]
    mixed_pivots=[]
    for name,first,second,cc in [('left',ma,mb,corr),('right',mb,ma,ct)]:
        table=[[F(v) for v in row] for row in data[name]]
        check(len(table)==4 and all(len(row)==len(keys) for row in table),'four complete outcome blocks '+name)
        check(all(table[0][j]+table[1][j]==table[2][j]+table[3][j] for j in range(len(keys))), 'all common sums '+name)
        check(table[0][index[()]]+table[1][index[()]]==1,'normalization '+name)
        for row in range(4):
            x,sign=row//2,(-1)**row
            check(table[row][index[()]]==(1+sign*first[x])/2,'shared outcome mass '+name+str(row))
            check(all(table[row][index[(y,)]]==(second[y]+sign*cc[x][y])/2 for y in range(2)),
                  'shared Bell moments '+name+str(row))
            matrix=[[table[row][index[canonical(u[::-1]+v)]] for v in words] for u in words]
            mixed_pivots.extend(positive_pivots(matrix))
    score=sum(c*v for c,v in zip(theta,moments));gap=score-beta
    threshold=F(1,20000) if k==6 else F(1,100000)
    check(gap>threshold,'strict mixed witness gap greater than '+str(threshold))
    interval_gap=None
    if k==7:
        slope=2-moments[0]-moments[2]
        interval_gap=gap-F(1,100000)*slope
        check(slope>=0 and interval_gap>F(8,1000000),
              'uniform mixed separation on the exact-target interval 0 <= t <= 1/100000')
    # Reflection and raw PVM balls have an invertible triangular change
    # of basis, diagonal 2^{-length}; no rows or outcomes are discarded.
    transform=[]
    for w in words:
        expansion={():F(1)}
        for letter in w:
            nxt={}
            for u,c in expansion.items():
                for v in (u,reduce_word(u+(letter,))):nxt[v]=nxt.get(v,F(0))+c/2
            expansion=nxt
        transform.append([expansion.get(v,F(0)) for v in words])
    check(all(transform[i][i]==F(1,2**len(words[i])) for i in range(len(words))) and
          all(transform[i][j]==0 for i in range(len(words)) for j in range(i+1,len(words))),
          'invertible triangular conversion to full native PVM words')
    return dict(status='PASS',checks=len(checks),standard_positive_LDL_pivots=len(pivots),
        mixed_positive_LDL_pivots=len(mixed_pivots),exact_target=str(beta),exact_gap=str(gap),
        exact_interval_endpoint_gap=str(interval_gap) if interval_gap is not None else None,
        gap_display=float(gap),theorem=f'beta-G is in D_3 and outside M_{k}, at beta=omega_Q=omega_L',
        minimal_standard_degree='not determined',mixed_upper_degree='not determined',
        scope='Finite-level separation only. Does not prove unbounded mixed degree.',passed_checks=checks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mutation-checks',action='store_true')
    parser.add_argument('--level',type=int,choices=[6,7],default=6);args=parser.parse_args()
    path=HERE/f'D3_not_M{args.level}_exact.json';payload=json.loads(path.read_text())
    report=verify(payload);report['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if args.mutation_checks:
        rejected=[]
        for name in ('coefficient','Gram','witness'):
            bad=copy.deepcopy(payload)
            if name=='coefficient':bad['coefficients'][4]=str(F(bad['coefficients'][4])+F(1,1000))
            elif name=='Gram':bad['standard_Gram'][0][0]='-1'
            else:bad['left'][0][0]=str(F(bad['left'][0][0])+F(1,1000))
            try:verify(bad)
            except AssertionError:rejected.append(name)
            else:raise AssertionError('Undetected mutation '+name)
        report['rejected_mutations']=rejected
    (HERE/f'D3_not_M{args.level}_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='passed_checks'},indent=2))


if __name__=='__main__':main()
