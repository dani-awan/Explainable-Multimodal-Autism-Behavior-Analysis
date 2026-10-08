"""Read-only preflight and fresh-output execution for standalone E6 stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def stage_main(script, function, spec):
    parser = argparse.ArgumentParser(description=function.__doc__ or spec['name'])
    parser.add_argument('--project', type=Path, default=Path(script).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Inspect paths and package presence only (default).')
    mode.add_argument('--run', action='store_true', help='Execute this stage explicitly.')
    if spec['writes']:
        parser.add_argument('--output', type=Path, help='New output directory inside the project; it must not exist.')
    for field in spec['fields']:
        parser.add_argument('--'+field.replace('_', '-'), type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Do not use python -O: source validation must remain enabled.')
    args.project = args.project.resolve()
    if not args.project.is_dir():
        parser.error('Project directory does not exist.')
    for field, default in spec['fields'].items():
        path = getattr(args, field) or args.project / default
        setattr(args, field, path.resolve())
    failures = []
    for field, relative in spec['requires']:
        path = getattr(args, field) / relative
        if not path.is_file():
            failures.append('Missing input: '+str(path))
        else:
            print('PASS input:', path)
    for module in spec['packages']:
        if importlib.util.find_spec(module) is None:
            failures.append('Missing Python package: '+module)
    if spec['source_cell'] == 120:
        coverage = args.project / 'docs/ASD_FEAT_FULL_AUDIO_COVERAGE.json'
        if coverage.is_file():
            try:
                fold_source = args.project / json.loads(coverage.read_text(encoding='utf-8'))['fold_source']
                if not fold_source.is_file():
                    failures.append('Missing fold source: '+str(fold_source))
            except (KeyError, ValueError, TypeError) as error:
                failures.append('Invalid coverage evidence: '+str(error))
    if spec['writes']:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.output = (args.output or args.project / 'results/asd_feat/e6_reproduction' / spec['name'] / stamp).resolve()
        if not args.output.is_relative_to(args.project) or args.output == args.project:
            failures.append('Output must be a new subdirectory inside the project.')
        if args.output.exists():
            failures.append('Output already exists; choose a new --output directory.')
        protected = [args.project / 'dataset', args.project / 'models', args.project / 'docs', args.project / 'scripts', args.project / 'dashboard', args.project / 'notebooks']
        if any(args.output.is_relative_to(folder) for folder in protected):
            failures.append('Output must not be inside source data, models, code, notebooks or documentation.')
        print('Planned output:', args.output)
    if failures:
        parser.error('\n'.join(failures))
    if not args.run:
        print('Preflight: PASS. Presence checks only; no arrays loaded, models fitted or files changed.')
        return
    # Source-derived numerical code executes only after explicit --run.
    function(args)
    if spec['writes']:
        if not args.output.is_dir():
            raise RuntimeError('Stage did not create its declared output folder.')
        receipt = {
            'stage': spec['name'], 'source_cell': spec['source_cell'],
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'script_sha256': hashlib.sha256(Path(script).read_bytes()).hexdigest(),
            'project': str(args.project), 'output': str(args.output),
            'selected_inputs': {field: str(getattr(args, field)) for field in spec['fields']},
            'scope': 'Standalone development stage execution; no official test processing',
        }
        (args.output / 'standalone_execution.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
