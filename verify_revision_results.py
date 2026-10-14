"""Run the accompanying exact checks; any failed command fails this driver."""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
COMMANDS=[
 ['research/bidirectional_coupling/verify_bidirectional_coupling.py'],
 ['research/bidirectional_coupling/verify_scope_extensions.py'],
 ['research/bidirectional_coupling/verify_mixed_separation.py'],
 ['research/bidirectional_coupling/verify_quantum_target_family.py'],
 ['research/mixed_divergence/verify_degree_pair.py','--mutation-checks'],
 ['research/mixed_divergence/verify_mixed_upper.py'],
 ['research/mixed_divergence/verify_standard_lower.py'],
 ['research/mixed_divergence/verify_corner_obstruction.py'],
]

if __name__=='__main__':
    records=[]
    for args in COMMANDS:
        result=subprocess.run([sys.executable,*args],cwd=HERE,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:
            print(result.stdout);print(result.stderr,file=sys.stderr)
            raise SystemExit(result.returncode)
        records.append(dict(command='python '+' '.join(args),status='PASS'))
        print(json.dumps(records[-1]),flush=True)
    report=dict(status='PASS',commands=records,scope='Accompanying exact proofs; no claim of all-level mixed divergence.')
    (HERE/'exact_checks_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
