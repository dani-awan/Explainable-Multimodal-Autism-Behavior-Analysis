"""Inspect stage inputs and project coverage without changing any data."""
import argparse
import json
from pathlib import Path
import sys

from script_runtime import resolve_matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stage', type=int, help='Inspect a single numbered stage.')
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = project / 'docs/SCRIPT_SOURCE_MAP.json'
    if not manifest.is_file():
        parser.error('docs/SCRIPT_SOURCE_MAP.json is missing; copy the complete package into the project.')
    stages = json.loads(manifest.read_text(encoding='utf-8'))['stages']
    if args.stage is not None:
        stages = [stage for stage in stages if int(stage['script'][:2]) == args.stage]
        if not stages:
            parser.error('Unknown stage number.')
    missing_count = 0
    for stage in stages:
        missing = [pattern for pattern in stage['inputs'] if not resolve_matches(project, pattern)]
        missing_count += len(missing)
        print(('MISSING' if missing else 'READY') + ': ' + stage['script'])
        for pattern in missing:
            print('  Input: ' + pattern)
    print('\nFile-presence inspection only; models were not trained and no results were changed.')
    return 1 if missing_count else 0


if __name__ == '__main__':
    sys.exit(main())
