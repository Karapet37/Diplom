#!/usr/bin/env python3
"""
Build coordinate vector datasets from all available sources and write to
DataSets/coordinate_vectors/{train,test}.jsonl

Usage:
    python scripts/build_coordinate_datasets.py [--max-empathetic N] [--max-archive N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agent_system.dataset_layer import (
    build_coordinate_dataset,
    coordinate_dataset_stats,
)

_OUTPUT = _ROOT / 'DataSets' / 'coordinate_vectors'


def write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build P-coordinate training datasets')
    parser.add_argument('--max-sessions',   type=int, default=300)
    parser.add_argument('--max-empathetic', type=int, default=3000)
    parser.add_argument('--max-archive',    type=int, default=2000)
    parser.add_argument('--no-sessions',    action='store_true')
    parser.add_argument('--no-empathetic',  action='store_true')
    parser.add_argument('--no-archive',     action='store_true')
    args = parser.parse_args()

    for split in ('train', 'test'):
        print(f'\n[{split}] Building coordinate dataset...')
        records = build_coordinate_dataset(
            split=split,
            include_sessions=(not args.no_sessions),
            include_empathetic=(not args.no_empathetic),
            include_archive=(not args.no_archive),
            include_corrections=True,
            max_sessions=args.max_sessions,
            max_empathetic=args.max_empathetic,
            max_archive=args.max_archive,
        )
        out_path = _OUTPUT / f'{split}.jsonl'
        write_jsonl(records, out_path)

        stats = coordinate_dataset_stats(records)
        print(f'  total records : {stats["total"]}')
        print(f'  by source     : {stats["by_source"]}')
        print(f'  by role       : {stats["by_role"]}')

        # Show top-5 non-trivial coordinates (those with >1 label in data)
        diverse = {
            pid: labels
            for pid, labels in stats['top_labels_per_coord'].items()
            if len(labels) > 1
        }
        print(f'  diverse coords: {len(diverse)} / {len(stats["top_labels_per_coord"])}')
        for pid in sorted(diverse)[:10]:
            print(f'    {pid}: {diverse[pid]}')

        print(f'  → written to {out_path}')


if __name__ == '__main__':
    main()
