"""Średnie podobieństwo obrazów w klasach i pomiędzy źródłami."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from obrazy.dane import DEFAULT_CLASSES, selected_classes


def intra(v):
    n = len(v)
    return float((np.dot(v.sum(axis=0), v.sum(axis=0)) - n) / (n * (n - 1)))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        required=True)
    parser.add_argument('--output', type=Path, default=Path('wyniki/podobienstwo'))
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    selected = selected_classes(args.manifest, args.class_file)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as data:
        vectors = data['vectors'].astype(np.float64)
        sources, classes, paths = (data[k].copy() for k in ('sets', 'classes', 'paths'))
    keep = np.isin(classes, selected)
    vectors, sources, classes, paths = vectors[keep], sources[keep], classes[keep], paths[keep]
    if set(classes) != set(selected):
        raise ValueError('Plik cech nie zawiera wszystkich wybranych klas.')
    assert vectors.ndim == 2 and vectors.shape[1] == 512
    assert len(vectors) == len(sources) == len(classes) == len(paths)
    assert set(sources) == {'rzeczywiste', 'syntetyczne'}
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    assert (norms > 0).all()
    vectors /= norms
    rows, maximum_error = [], 0.0
    for label in sorted(set(classes)):
        a = vectors[(classes == label) & (sources == 'rzeczywiste')]
        b = vectors[(classes == label) & (sources == 'syntetyczne')]
        assert len(a) > 1 and len(b) > 1


        cross = float(a.mean(axis=0) @ b.mean(axis=0))
        aa, bb = intra(a), intra(b)

        direct_cross = float((a @ b.T).mean())
        ma, mb = a @ a.T, b @ b.T
        direct_a = float(ma[np.triu_indices(len(a), k=1)].mean())
        direct_b = float(mb[np.triu_indices(len(b), k=1)].mean())
        error = max(abs(cross - direct_cross), abs(aa - direct_a), abs(bb - direct_b))
        assert error < 1e-10, (label, error)
        maximum_error = max(maximum_error, error)
        rows.append({'klasa': str(label), 'n_rzeczywiste': len(a),
                     'n_syntetyczne': len(b), 'pary_real_syn': len(a) * len(b),
                     'real_real': aa, 'syn_syn': bb, 'real_syn': cross})
    with (args.output / 'wyniki_klas.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'n_klas': len(rows), 'n_rzeczywiste': int(sum(r['n_rzeczywiste'] for r in rows)),
               'n_syntetyczne': int(sum(r['n_syntetyczne'] for r in rows)),
               'pary_real_syn': int(sum(r['pary_real_syn'] for r in rows))}
    for key in ('real_real', 'syn_syn', 'real_syn'):
        values = np.array([r[key] for r in rows])
        summary[key] = {'srednia_klas': float(values.mean()),
                        'sd_miedzy_klasami': float(values.std(ddof=1)),
                        'min_klas': float(values.min()), 'max_klas': float(values.max())}
    summary['real_syn_nizsze_od_syn_syn'] = sum(r['real_syn'] < r['syn_syn'] for r in rows)
    summary['real_syn_nizsze_od_real_real'] = sum(r['real_syn'] < r['real_real'] for r in rows)
    summary['kontrola'] = {'metoda': 'Wszystkie pary w klasie, kazda klasa z jednakowa waga',
                          'maksymalny_blad_wzgledem_jawnych_macierzy': maximum_error,
                          'plik_cech': str(args.input.resolve()),
                          'sha256_cech': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                          'numpy': np.__version__, 'model_uruchomiony': False}
    (args.output / 'podsumowanie.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
