"""Verify the degree-three upper certificate and common level-seven witness."""
from pathlib import Path
import runpy
import sys

target=Path(__file__).with_name('verify_exact_D3_M6.py')
sys.argv=[str(target),'--level','7',*sys.argv[1:]]
runpy.run_path(str(target),run_name='__main__')
