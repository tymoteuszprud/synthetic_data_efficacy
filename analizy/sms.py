"""Koncentracja słownictwa i początków, długości oraz duplikaty wiadomości."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import re

import numpy as np

TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokens(text):
    return TOKEN.findall(text.casefold().replace('’', "'"))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concentration(counter):
    values = np.array(sorted(counter.values(), reverse=True), dtype=float)
    return np.cumsum(values) / values.sum() * 100 if len(values) else values


def analyze(real, synth):
    statistics, ranks = [], []
    for source, path in (('Rzeczywiste', real), ('Syntetyczne', synth)):
        with path.open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        for label in ('HAM', 'SPAM'):
            messages = [r['message'] for r in rows if r['label'].strip().upper() == label]
            if not messages:
                raise ValueError(f'{source}: brak wiadomości klasy {label}.')
            split = [tokens(text) for text in messages]
            first = Counter(words[0] for words in split if words)
            words = Counter(word for entry in split for word in entry)
            chars = Counter(char for text in messages for char in text.casefold())
            duplicate = Counter(text.strip().casefold() for text in messages)
            row = dict(zbior=source, klasa=label, n=len(messages),
                       duplikaty_proc=100*sum(count-1 for count in duplicate.values())/len(messages),
                       top5_poczatkow_proc=100*sum(c for _, c in first.most_common(5))/len(messages),
                       top50_slow_proc=100*sum(c for _, c in words.most_common(50))/sum(words.values()))
            unique_first = Counter(entry[0] for text in sorted(duplicate) if (entry := tokens(text)))
            row['top5_poczatkow_bez_duplikatow_proc'] = 100*sum(c for _, c in unique_first.most_common(5))/len(duplicate)
            for unit, values in (('znaki', [len(m) for m in messages]), ('wyrazy', [len(w) for w in split])):
                values = np.array(values)
                row.update({f'{unit}_{key}': float(value) for key, value in
                            [('srednia', values.mean()), ('sd', values.std(ddof=1) if len(values)>1 else 0),
                             ('q25', np.quantile(values, .25)), ('mediana', np.median(values)),
                             ('q75', np.quantile(values, .75)), ('maksimum', values.max())]})
            statistics.append(row)
            for kind, counts in (('poczatek', first), ('slowa', words), ('znaki', chars)):
                total = len(messages) if kind == 'poczatek' else sum(counts.values())
                cumulative = 0
                for rank, (term, count) in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0])), 1):
                    cumulative += count
                    ranks.append(dict(zbior=source, klasa=label, miara=kind, ranga=rank,
                                      element=term, wystapienia=count, skumulowany_udzial_proc=100*cumulative/total))
    return statistics, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('wyniki/sms'))
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    summaries, ranks = analyze(args.real, args.synthetic)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'podsumowanie.csv', summaries)
    write_csv(args.output / 'koncentracja.csv', ranks)
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for kind in ('poczatek', 'slowa', 'znaki'):
            figure, axes = plt.subplots(1, 2, figsize=(7, 2.8))
            for axis, label in zip(axes, ('HAM', 'SPAM')):
                for source, color in (('Rzeczywiste', '#1f77b4'), ('Syntetyczne', '#d62728')):
                    group = [r for r in ranks if r['miara'] == kind and r['klasa'] == label and r['zbior'] == source]
                    group = group[:50 if kind != 'znaki' else 30]
                    axis.plot([r['ranga'] for r in group], [r['skumulowany_udzial_proc'] for r in group], label=source, color=color)
                axis.set(title=label, xlabel='Liczba najczęstszych elementów', ylabel='Skumulowany udział [%]', ylim=(0, 100))
                axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(args.output / (kind + '.pdf'), bbox_inches='tight')
            plt.close(figure)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
