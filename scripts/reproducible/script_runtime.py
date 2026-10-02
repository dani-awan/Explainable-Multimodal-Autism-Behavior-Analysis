"""Shared command-line preflight and overwrite handling; standard library only."""
import argparse
import glob
from pathlib import Path
import shutil
import sys


def resolve_matches(project, pattern):
    return sorted(Path(p) for p in glob.glob(str(project / pattern)))


def run_stage(script, function, requires, produces, external=(), originals=False):
    parser = argparse.ArgumentParser(description=function.__module__ + ': notebook-derived stage')
    parser.add_argument('--project', type=Path, default=Path(script).resolve().parents[2])
    parser.add_argument('--check', action='store_true', help='Check required paths and tools without executing the stage.')
    parser.add_argument('--overwrite', action='store_true', help='Allow this stage to update existing generated outputs.')
    if originals:
        parser.add_argument('--original-videos', type=Path, help='Folder containing the completed original videos.')
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('Do not use python -O: validation assertions must remain enabled.')
    project = args.project.resolve()
    if not project.is_dir():
        parser.error('The project directory does not exist.')
    failures = []
    for pattern in requires:
        found = resolve_matches(project, pattern)
        if not found:
            failures.append('Missing input: ' + pattern)
        elif args.check:
            print('PASS input: ' + pattern)
    for tool in external:
        if not shutil.which(tool):
            failures.append('Required tool not found on PATH: ' + tool)
        elif args.check:
            print('PASS tool: ' + tool)
    if originals:
        if args.original_videos is None or not args.original_videos.is_dir():
            failures.append('Provide --original-videos with an existing directory.')
    if failures:
        parser.error('\n'.join(failures))
    existing = sorted({str(p.relative_to(project)) for pattern in produces
                       for p in resolve_matches(project, pattern) if p.is_file()})
    if args.check:
        print('Preflight: PASS (stage execution was not performed).')
        print(f'Existing protected outputs: {len(existing)}')
        return
    if existing and not args.overwrite:
        parser.error('Generated outputs already exist. Use --check for inspection or '
                     '--overwrite for an intentional rerun. Examples: ' + ', '.join(existing[:5]))
    if originals:
        function(project, original_videos=args.original_videos.resolve())
    else:
        function(project)
