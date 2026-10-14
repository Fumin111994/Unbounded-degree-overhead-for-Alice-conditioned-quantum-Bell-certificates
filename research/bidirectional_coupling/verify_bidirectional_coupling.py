"""Exact research audit of a coupled, bidirectionally hard Bell family.

This is a research addition, not part of the published v0.2.0 archive.
Run with Python 3 and SymPy. No optimization solver is imported or used.

All-level arguments are in proof.tex. Finite matrix audits below check the
tensor-lifting construction on EVERY four-generator word at levels 1, 2, 3;
they are not a substitute for the universal tensor-positivity argument.
"""

from functools import lru_cache
from pathlib import Path
import json

import sympy as sp


CHECKS = []


def require(condition, name):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def zero(expr, name):
    require(sp.factor(expr) == 0, name)


def reduced(word):
    result = []
    for letter in word:
        if result and result[-1] == letter:
            result.pop()
        else:
            result.append(letter)
    return tuple(result)


def words(generators, level):
    result = [()]
    front = [()]
    for _ in range(level):
        front = [w + (y,) for w in front for y in range(generators)
                 if not w or w[-1] != y]
        result.extend(front)
    return result


def clean(poly):
    return {w: sp.expand(c) for w, c in poly.items() if sp.expand(c) != 0}


def add(*polys):
    result = {}
    for poly in polys:
        for w, c in poly.items():
            result[w] = result.get(w, 0) + c
    return clean(result)


def scale(poly, c):
    return clean({w: c * v for w, v in poly.items()})


def multiply(left, right):
    result = {}
    for (a, b), c in left.items():
        for (aa, bb), d in right.items():
            key = (reduced(a + aa), reduced(b + bb))
            result[key] = result.get(key, 0) + c * d
    return clean(result)


def algebraic_audit():
    alpha, q, eta = sp.symbols('alpha q eta', real=True)
    unit = {((), ()): sp.S.One}
    alice = [{((i,), ()): sp.S.One} for i in range(4)]
    bob = [{((), (i,)): sp.S.One} for i in range(4)]

    def chsh(a0, a1, b0, b1):
        return add(multiply(a0, add(b0, b1)),
                   multiply(a1, add(b0, scale(b1, -1))))

    s1 = chsh(alice[0], alice[1], bob[0], bob[1])
    s2 = chsh(alice[2], alice[3], bob[2], bob[3])
    f1 = add(scale(alice[0], alpha), s1)
    f2 = add(scale(bob[2], alpha), s2)
    r1 = add(scale(unit, q), scale(f1, -1))
    r2 = add(scale(unit, q), scale(f2, -1))
    t1 = add(scale(alice[1], alpha),
             scale(chsh(alice[0], alice[1], bob[0], scale(bob[1], -1)), -1))
    t2 = add(scale(bob[3], alpha),
             scale(chsh(bob[2], bob[3], alice[2], scale(alice[3], -1)), -1))
    bridge = multiply(alice[0], bob[2])
    g = add(f1, f2, scale(bridge, eta))
    gap = add(scale(unit, 2*q + eta), scale(g, -1))
    j = add(alice[0], scale(bob[2], -1))
    identity = add(multiply(r1, r1), multiply(t1, t1),
                   multiply(r2, r2), multiply(t2, t2),
                   scale(multiply(j, j), eta*q), scale(gap, -2*q))
    relation = q*q - 8 - 2*alpha*alpha
    require(all(sp.rem(c, relation, q) == 0 for c in identity.values()),
            'full noncommutative degree-two SOS identity, symbolic parameters')
    require(max(len(a)+len(b) for p in (r1, r2, t1, t2)
                for a, b in p) <= 2, 'all standard SOS factors have degree at most two')

    def exchanged(poly):
        result = {}
        swap = {0: 2, 1: 3, 2: 0, 3: 1}
        for (a, b), c in poly.items():
            key = (tuple(swap[i] for i in b), tuple(swap[i] for i in a))
            result[key] = result.get(key, 0) + c
        return clean(result)

    require(exchanged(g) == g, 'exact party-exchange symmetry with setting permutation')
    links = {(a[0], b[0]) for (a, b), c in g.items() if a and b}
    nodes = {('A', 0)}
    for _ in range(8):
        for a, b in links:
            if ('A', a) in nodes or ('B', b) in nodes:
                nodes.update({('A', a), ('B', b)})
    require(len(nodes) == 8 and len(links) == 9,
            'nine correlator edges connect all eight measurement settings')


def scalar_audit():
    alpha, q = sp.symbols('alpha q', positive=True)
    relation = q*q - 8 - 2*alpha*alpha

    def qzero(expr, name):
        num = sp.together(expr).as_numer_denom()[0]
        zero(sp.rem(sp.expand(num), relation, q), name)

    c = 2*alpha/q
    b = q/4
    d2 = (4-alpha*alpha)/8
    s2 = 1-c*c
    qzero(b*b + d2 - 1, 'attaining measurement reflection normalization')
    qzero(d2 - b*b*s2, 'optimal angle and Schmidt correlation identity')
    qzero(alpha*c + 2*b + 2*d2/b - q, 'both qubit subtests attain q')
    qzero(1-c*c - (4-alpha*alpha)/(4+alpha*alpha),
          'Schmidt bias is strictly subunit in the subcritical range')

    r, u = sp.symbols('r u', real=True)
    a = 2-8*r
    q2 = 8+2*a*a
    w = 4-(1-2*r)*8*r
    aa = w-r
    cc = q2-2*r*a*(1-2*r)
    p = 496*r**4-680*r**3+341*r*r-86*r+8
    zero(q2*aa*aa-cc*cc-64*r*r*p, 'all-level coupled gap polynomial identity')
    positive_p = (496*u**4 + sp.Rational(2408, 5)*u**3
                  + sp.Rational(4169, 25)*u*u + sp.Rational(4527, 125)*u
                  + sp.Rational(5449, 2500))
    zero(p.subs(r, sp.Rational(1, 10)-u)-positive_p,
         'gap polynomial has positive coefficients in u=1/10-r')
    require(all(v > 0 for v in sp.Poly(positive_p, u).all_coeffs()),
            'uniform polynomial positivity for every 0<r<=1/10')
    # A>0, C>0 and q>0 make the squared comparison a valid signed comparison.
    require(4-sp.Rational(9, 10) > 0,
            'positive A=w-r throughout 0<r<=1/10')
    require(16-sp.Rational(68, 10)-sp.Rational(32, 1000) > 0,
            'positive C throughout 0<r<=1/10')
    require(9-sp.Rational(16, 10) > 0, 'A<=4 by r*(9-16*r)>=0')
    require(68-sp.Rational(152, 10) > 0, 'C<=16 by r*(68-152*r+32*r**2)>=0')
    zero(q2-16+64*r-128*r*r, 'q<4 for 0<r<=1/10')
    zero((sp.Rational(5449, 2500)/2)-sp.Rational(5449, 5000),
         'explicit witness gap constant 5449/5000 times r squared')

    # The all-epsilon degree lower bound uses k=floor(v/sqrt(epsilon)).
    z, cc0 = sp.symbols('z cc0', positive=True)
    rr = z*z/(8+2*z*z)
    e = cc0*z*z
    qq = sp.sqrt(8+2*(2-e)**2)
    delta = 4-(1-2*rr)*e-qq-e/8*(1-(1-2*rr)*2*(2-e)/qq)
    zero(sp.limit(delta/z**4, z, 0)-cc0*(7-5*cc0)/32,
         'positive asymptotic gap for epsilon*k squared < 7/5')

    zed = sp.Matrix([1, 0])
    ex = sp.Matrix([0, 1])
    plus = (zed+ex)/sp.sqrt(2)
    minus = (zed-ex)/sp.sqrt(2)
    av = [zed, ex, plus, minus]
    bv = [plus, minus, zed, ex]
    objective = alpha*(zed.dot(av[0])+zed.dot(bv[2]))
    objective += sum((-1 if i == j == 1 else 1)*av[i].dot(bv[j])
                     for i in (0, 1) for j in (0, 1))
    objective += sum((-1 if i == j == 1 else 1)*av[i+2].dot(bv[j+2])
                     for i in (0, 1) for j in (0, 1))
    eta = sp.symbols('eta', positive=True)
    objective += eta*av[0].dot(bv[2])
    zero(objective-4*sp.sqrt(2)-2*alpha-eta,
         'feasible standard level-one Gram has value 4sqrt(2)+2alpha+eta')
    zero((2*sp.sqrt(2)+alpha)**2-(8+2*alpha*alpha)
         -alpha*(4*sp.sqrt(2)-alpha),
         'standard level one strictly exceeds the exact quantum bound')

    # Physical size of the bridge: optimize over independent test resources,
    # allowing each subtest to sacrifice its individual optimum.
    bias, e = sp.symbols('bias e', real=True)
    zero(sp.diff(2*sp.sqrt(2-bias*bias), bias, 2)
         +4/(2-bias*bias)**sp.Rational(3, 2),
         'fixed-bias CHSH boundary is strictly concave')
    mstar = 1-sp.Rational(7, 32)*e
    stationarity = 2*bias/sp.sqrt(2-bias*bias)-(2-e)-e*bias/8
    zero(sp.diff(stationarity, bias).subs({e: 0, bias: 1})-4,
         'independent optimum has a nondegenerate critical derivative')
    zero(sp.series(stationarity.subs(bias, mstar), e, 0, 2).removeO(),
         'independent optimum bias is 1-7epsilon/32+O(epsilon squared)')
    independent = 2*(2-e)*mstar+4*sp.sqrt(2-mstar*mstar)+e*mstar*mstar/8
    beta = 2*sp.sqrt(8+2*(2-e)**2)+e/8
    loss_coefficient = sp.limit((beta-independent)/e**2, e, 0)
    zero(loss_coefficient-sp.Rational(15, 256),
         'best independent test resources lose 15epsilon squared/256 at leading order')
    local = 4+2*(2-e)+e/8
    advantage_coefficient = sp.limit((beta-local)/e**2, e, 0)
    zero(advantage_coefficient-sp.Rational(1, 4),
         'coupled nonlocal violation is epsilon squared/4 at leading order')
    zero(loss_coefficient/advantage_coefficient-sp.Rational(15, 64),
         'independence penalty is 15/64 of the nonlocal violation at critical scale')


def rational_psd(matrix):
    """Exact Schur-complement positivity; no floating-point eigenvalues."""
    m = sp.Matrix(matrix)
    require(m == m.T, 'rational PSD input is symmetric')
    while m.rows:
        diagonal = [m[i, i] for i in range(m.rows)]
        if any(v < 0 for v in diagonal):
            return False
        pivots = [i for i, v in enumerate(diagonal) if v > 0]
        if not pivots:
            return m == sp.zeros(m.rows)
        i = pivots[0]
        rest = [j for j in range(m.rows) if j != i]
        column = m.extract(rest, [i])
        m = m.extract(rest, rest)-column*column.T/m[i, i]
    return True


def lifted_gram_audit(level):
    old = words(2, level)
    native = words(4, level)
    old_index = {w: i for i, w in enumerate(old)}
    r = sp.Rational(1, 8*level*level+2)

    @lru_cache(None)
    def tau(w):
        w = reduced(w)
        if len(w) % 2:
            return sp.S.Zero
        return max(sp.S.Zero, 1-sp.Rational(len(w), 2*level))

    def ly(y, w):
        return (tau(w)+tau(reduced((y,)+w)))/2

    def moments(w):
        l0, l1 = ly(0, w), ly(1, w)
        atom = r*(-1)**len(w)
        return [l0+l1-atom, atom, l0, l1, l0+l1]

    base = [sp.Matrix(len(old), len(old), lambda i, j:
                      moments(reduced(tuple(reversed(old[i]))+old[j]))[t])
            for t in range(5)]
    for i, matrix in enumerate(base):
        require(rational_psd(matrix), f'k={level}: original rational Gram {i} PSD')

    eye = sp.eye(2)
    z = sp.diag(1, -1)
    x = sp.Matrix([[0, 1], [1, 0]])
    # Rational control assemblage audits the UNIVERSAL lifting map exactly.
    # The optimal parameter-dependent assemblage is addressed in scalar_audit
    # and in the proof; this control is not presented as the separating score.
    dd = sp.diag(sp.Rational(3, 5), sp.Rational(4, 5))
    aa = [(3*z+4*x)/5, (3*z-4*x)/5]
    factors = [(dd, i) for i in range(4)]
    for observable in aa:
        require(observable*observable == eye, 'rational control Alice reflection')
        factors.extend([(dd*(eye+observable)/2, 4),
                        (dd*(eye-observable)/2, 4)])

    def image(w):
        ow = reduced(tuple(i for i in w if i < 2))
        mat = eye
        for i in w:
            if i >= 2:
                mat = mat*(z if i == 2 else x)
        return ow, mat

    images = [image(w) for w in native]
    grams = []
    for block, (factor, base_index) in enumerate(factors):
        rho = factor*factor.T
        compression = sp.zeros(4*len(old), len(native))
        for j, (ow, mat) in enumerate(images):
            coefficient = mat*factor
            start = 4*old_index[ow]
            for v in range(4):
                compression[start+v, j] = coefficient[v]
        congruence = compression.T*sp.kronecker_product(base[base_index], sp.eye(4))*compression
        direct = sp.Matrix(len(native), len(native), lambda i, j:
                           moments(reduced(tuple(reversed(images[i][0]))+images[j][0]))[base_index]
                           *sp.trace(rho*images[i][1].T*images[j][1]))
        require(congruence == direct,
                f'k={level}: every four-generator word Gram in block {block} is a PSD compression')
        grams.append(direct)
    common = grams[0]+grams[1]
    require(all(grams[2*j]+grams[2*j+1] == common for j in range(4)),
            f'k={level}: exact common outcome sums on every Gram entry')
    require(common[0, 0] == 1, f'k={level}: common identity moment normalized to one')

    # Invertible four-generator PVM change of basis, not a kept-outcome slice.
    transform = sp.zeros(len(native))
    indices = {w: i for i, w in enumerate(native)}
    for i, word in enumerate(native):
        expansion = {(): sp.S.One}
        for y in word:
            following = {}
            for w, c in expansion.items():
                for v in (w, reduced(w+(y,))):
                    following[v] = following.get(v, 0)+c/2
            expansion = following
        for w, c in expansion.items():
            transform[i, indices[w]] = c
    require(transform.is_lower,
            f'k={level}: four-generator PVM conversion is triangular')
    require(all(transform[i, i] == sp.Rational(1, 2)**len(w)
                for i, w in enumerate(native)),
            f'k={level}: PVM conversion is invertible and degree preserving')
    converted = [transform*g*transform.T for g in grams]
    require(all(converted[2*j]+converted[2*j+1] == converted[0]+converted[1]
                for j in range(4)), f'k={level}: common sums preserved in raw PVM convention')
    return {'level': level, 'native_Bob_words': len(native), 'outcome_blocks': 8,
            'arithmetic': 'exact rational', 'positivity': 'explicit PSD congruence',
            'optimal_separating_score': 'proved symbolically, not this rational control'}


def actual_level_one_witness():
    """Independent full Gram audit at the actual separating parameter.

    All entries lie in Q(sqrt(17)). All principal minors, including zeros,
    are checked exactly. This is an audit of one case, not the all-level proof.
    """
    level = 1
    r = sp.Rational(1, 10)
    alpha = sp.Rational(6, 5)
    eta = r
    q = 4*sp.sqrt(17)/5
    c, b = 2*alpha/q, q/4
    ds = (4-alpha*alpha)/(2*q)
    eye = sp.eye(2)
    z = sp.diag(1, -1)
    x = sp.Matrix([[0, 1], [1, 0]])
    sigma = (eye+c*z)/2
    conditional = [(eye+c*z+t*(alpha*eye/2+b*z+u*ds*x))/4
                   for u in (1, -1) for t in (1, -1)]
    rho = [sigma]*4+conditional
    native = words(4, 1)

    def tau(w):
        w = reduced(w)
        return sp.S.Zero if len(w) % 2 else max(sp.S.Zero, 1-sp.Rational(len(w), 2))

    def moment(w, block):
        l0 = (tau(w)+tau((0,)+w))/2
        l1 = (tau(w)+tau((1,)+w))/2
        atom = r*(-1)**len(reduced(w))
        return [l0+l1-atom, atom, l0, l1, l0+l1][block]

    def image(w):
        ow = reduced(tuple(y for y in w if y < 2))
        mat = eye
        for y in w:
            if y >= 2:
                mat *= z if y == 2 else x
        return ow, mat

    images = [image(w) for w in native]
    grams = [sp.Matrix(len(native), len(native), lambda i, j:
                       sp.simplify(moment(tuple(reversed(images[i][0]))+images[j][0],
                                          block if block < 4 else 4)
                                   *sp.trace(rho[block]*images[i][1].T*images[j][1])))
             for block in range(8)]
    for block, g in enumerate(grams):
        minors = []
        for mask in range(1, 1 << len(native)):
            indices = [i for i in range(len(native)) if mask & (1 << i)]
            minors.append(sp.simplify(g.extract(indices, indices).det(method='domain-ge')))
        require(all(v >= 0 for v in minors),
                f'actual alpha=6/5: all 31 principal minors of block {block} are nonnegative exactly')
    common = grams[0]+grams[1]
    require(all(grams[2*j]+grams[2*j+1] == common for j in range(4)),
            'actual alpha=6/5: all eight blocks have the same outcome sum')
    value = alpha*(grams[0][0, 0]-grams[1][0, 0])+alpha*common[0, 3]
    value += sum(((-1 if a == y == 1 else 1)
                  *(grams[2*a][0, 1+y]-grams[2*a+1][0, 1+y]))
                 for a in (0, 1) for y in (0, 1))
    value += sum(((-1 if a == y == 1 else 1)
                  *(grams[4+2*a][0, 3+y]-grams[5+2*a][0, 3+y]))
                 for a in (0, 1) for y in (0, 1))
    value += eta*(grams[0][0, 3]-grams[1][0, 3])
    gap = sp.simplify(value-2*q-eta)
    zero(value-(4-(1-2*r)*(2-alpha)+q+eta*(1-2*r)*c),
         'actual alpha=6/5: score reconstructed from full Gram entries')
    require(gap > sp.Rational(5449, 5000)*r*r,
            'actual alpha=6/5: strictly separating gap exceeds the stated rational bound')
    return {'level': level, 'alpha': str(alpha), 'eta': str(eta),
            'exact_gap': str(gap), 'PSD_test': 'all 248 principal minors; exact algebraic arithmetic'}


def main():
    algebraic_audit()
    scalar_audit()
    records = [lifted_gram_audit(k) for k in (1, 2, 3)]
    actual = actual_level_one_witness()
    report = {
        'status': 'PASS', 'checks': len(CHECKS), 'failed': 0,
        'scope': 'four binary questions per party; bridge eta=(2-alpha)/8',
        'quantum_bound': '2*sqrt(8+2*alpha**2)+(2-alpha)/8',
        'standard_exact_degree': 2,
        'both_conditioning_degrees': 'Theta((2-alpha)**(-1/2))',
        'liminf_lower_constant': 'sqrt(7/5)',
        'all_level_sequence': 'r=1/(8*k**2+2); alpha=2-8*r; eta=r',
        'sequence_gap_lower_bound': '5449*r**2/5000',
        'independent_test_resource_penalty': '15*(2-alpha)**2/256 + O((2-alpha)**3)',
        'penalty_over_nonlocal_violation_limit': '15/64',
        'not_proved': ['two-question-per-party version', 'nonvanishing bridge at critical tilt',
                       'cryptographic failure or a lower bound for all compiled proof methods'],
        'finite_matrix_audits': records, 'actual_separating_full_Gram_audit': actual,
        'passed_checks': CHECKS,
    }
    output = Path(__file__).with_name('verification_report.json')
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'checks', 'failed', 'scope',
                                           'quantum_bound', 'both_conditioning_degrees')}, indent=2))


if __name__ == '__main__':
    main()
