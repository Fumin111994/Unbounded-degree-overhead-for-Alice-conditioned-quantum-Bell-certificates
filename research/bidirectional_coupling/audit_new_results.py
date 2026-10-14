"""Independent rational reconstruction, mutation checks, and manuscript guard.

This supplements the analytic proofs. It is not a numerical SDP test.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re

import sympy as sp

from verify_mixed_separation import verify


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    payload=json.loads((HERE/'mixed_Gstar_level2_exact.json').read_text())
    keys=[tuple(w) for w in payload['moment_keys']]
    words=[tuple(w) for w in payload['word_basis']]
    def simplify_word(seq):
        # Independent iterative cancellation, separate from discovery code.
        old=None
        seq=tuple(seq)
        while old!=seq:
            old=seq
            for i in range(len(seq)-1):
                if seq[i]==seq[i+1]:
                    seq=seq[:i]+seq[i+2:]
                    break
        return min(seq,seq[::-1])
    behavior=[]
    determinants=0
    for direction in ('left','right'):
        table=[[sp.Rational(v) for v in row] for row in payload[direction]]
        matrices=[sp.Matrix(5,5,lambda i,j:
            row[keys.index(simplify_word(words[i][::-1]+words[j]))]) for row in table]
        for matrix in matrices:
            for mask in range(1,32):
                ids=[i for i in range(5) if mask & (1<<i)]
                assert matrix.extract(ids,ids).det(method='domain-ge')>0
                determinants+=1
        common=matrices[0]+matrices[1]
        assert common==matrices[2]+matrices[3] and common[0,0]==1
        first=[matrices[2*x][0,0]-matrices[2*x+1][0,0] for x in range(2)]
        second=[common[0,1+y] for y in range(2)]
        corr=sp.Matrix(2,2,lambda x,y:matrices[2*x][0,1+y]-matrices[2*x+1][0,1+y])
        behavior.append((first,second,corr))
    left,right=behavior
    assert left[0]==right[1] and left[1]==right[0] and left[2]==right[2].T
    recomputed=left[0]+left[1]+list(left[2])
    assert recomputed==list(map(sp.Rational,payload['shared_Bell_moments']))
    score=sum(c*v for c,v in zip(map(sp.Rational,payload['coefficients']),recomputed))
    assert score-sp.Rational(501,200)>sp.Rational(79,100000)

    mutants=[]
    wrong=deepcopy(payload)
    wrong['shared_Bell_moments'][0]=str(sp.Rational(wrong['shared_Bell_moments'][0])+sp.Rational(1,10000))
    mutants.append(('mismatched shared Bell behavior',wrong))
    wrong=deepcopy(payload)
    for row in (0,3):
        wrong['left'][row][6]=str(sp.Rational(wrong['left'][row][6])+100)
    mutants.append(('indefinite higher moment with common sums preserved',wrong))
    wrong=deepcopy(payload)
    wrong['right'][3][5]=str(sp.Rational(wrong['right'][3][5])+sp.Rational(1,10000))
    mutants.append(('inconsistent higher-word outcome sum',wrong))
    rejections=[]
    for name,bad in mutants:
        try:
            verify(bad,archived=False)
        except AssertionError as error:
            rejections.append(dict(mutation=name,rejected_at=str(error)))
        else:
            raise AssertionError('mutation was not rejected: '+name)

    revision=json.loads((ROOT/'revision_manifest.json').read_text(encoding='utf-8-sig'))
    original=Path(revision['source_root'])
    snapshot=revision['preserved_sources']
    preserved={path:hashlib.sha256((original/path).read_bytes()).hexdigest()==digest
               for path,digest in snapshot.items()}
    assert all(preserved.values())
    fragments=['proof.tex','bridge_strength.tex','mixed_directions.tex','quantum_target_family.tex']
    full=(ROOT/'supplementary.tex').read_text(encoding='utf-8')+'\n'+ '\n'.join(
        (HERE/f).read_text(encoding='utf-8') for f in fragments)
    labels=set(re.findall(r'\\label\{([^}]+)\}',full))
    tex_audits={}
    for name in fragments:
        text=(HERE/name).read_text(encoding='utf-8')
        stack=[]
        for action,env in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
            if action=='begin':
                stack.append(env)
            else:
                assert stack and stack.pop()==env,(name,env)
        assert not stack,name
        refs=[r for group in re.findall(r'\\(?:eqref|cref|ref)\{([^}]+)\}',text)
              for r in group.split(',')]
        assert all(r in labels for r in refs),(name,[r for r in refs if r not in labels])
        assert ']longrightarrow' not in text
        tex_audits[name]=dict(environments='balanced',references='resolved across research fragments and SM',
                              compilation='integrated into supplementary.tex; compilation checked separately')
    report=dict(status='PASS',all_principal_minors_positive=determinants,
        independent_score=str(score),mutation_rejections=rejections,
        previous_manuscript_sources_unchanged=preserved,fragment_audits=tex_audits)
    (HERE/'new_results_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
