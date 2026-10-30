"""Agreguje wyniki tekstowe; F0.5 jest liczone z TP, FP i FN."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('wyniki/tekst_podsumowanie.csv'))
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding='utf-8'))
    groups, identities = {}, set()
    for record in records:
        identity = (record['variant'], record['percentage'], record['seed'])
        if identity in identities:
            raise ValueError(f'Powtórzony przebieg: {identity}')
        identities.add(identity)
        t = record['test_results']
        denominator = 1.25*t['tp'] + t['fp'] + .25*t['fn']
        value = 100*1.25*t['tp']/denominator if denominator else 0.0
        groups.setdefault(identity[:2], []).append(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['variant', 'budget', 'runs', 'f05_mean_percent', 'f05_sd_pp'])
        writer.writeheader()
        for (variant, budget), values in sorted(groups.items()):
            writer.writerow(dict(variant=variant, budget=budget, runs=len(values),
                                 f05_mean_percent=statistics.mean(values),
                                 f05_sd_pp=statistics.stdev(values) if len(values)>1 else ''))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
