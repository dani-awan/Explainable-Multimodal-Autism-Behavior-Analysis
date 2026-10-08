"""Inspect packaged scripts and existing E6 project inputs without executing cells."""
import argparse
import ast
import importlib.util
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
args = parser.parse_args()
project = args.project.resolve()
folder = Path(__file__).resolve().parent
missing = []
for script in sorted(folder.glob('*.py')):
    source = script.read_text(encoding='utf-8')
    ast.parse(source, filename=str(script))
    print('PASS syntax:', script.name)
required = [
    'dataset/external/asd_feat/derived/e6_inputs/development_combined_X.csv',
    'dataset/external/asd_feat/derived/e6_inputs/development_metadata.csv',
    'dataset/external/asd_feat/derived/e6_inputs/development_y.csv',
    'dataset/external/asd_feat/derived/e6_inputs/input_schema.json',
    'docs/ASD_FEAT_FULL_AUDIO_COVERAGE.json',
]
for relative in required:
    if (project / relative).is_file():
        print('PASS input:', relative)
    else:
        missing.append(relative)
        print('MISSING input:', relative)
for module in ['numpy', 'pandas', 'sklearn', 'scipy', 'joblib', 'threadpoolctl', 'matplotlib']:
    if importlib.util.find_spec(module) is None:
        missing.append('Python package '+module)
        print('MISSING package:', module)
    else:
        print('PASS package:', module)
print('Inspection only; no training, extraction or changes to existing files.')
if missing:
    sys.exit(1)
print('E6 project preflight: PASS')
