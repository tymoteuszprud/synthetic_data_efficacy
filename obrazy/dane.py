"""Wspólne listy klas, referencji i rozłącznych podziałów obrazów, bez bibliotek GPU."""
import csv
import hashlib
import json
from pathlib import Path
import random

DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / 'konfiguracje/wybrane_klasy.txt'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
BUDGETS = (1, 3, 10, 30, 100)


def load_classes(path=DEFAULT_CLASSES, names=None):
    if Path(path).suffix.lower() == '.csv':
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        weights = {r['sport_name']: int(r['count']) for r in rows}
    else:
        entries = [line.strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
                   if line.strip() and not line.lstrip().startswith('#')]
        if len(entries) != len(set(entries)):
            raise ValueError('Lista klas zawiera powtórzenia.')
        weights = dict.fromkeys(entries, 1)
    selected = list(names) if names else list(weights)
    if len(set(selected)) != len(selected) or not selected or set(selected) - weights.keys():
        raise ValueError('Wybrane nazwy klas muszą być różne i należeć do wykazu klas.')
    if any(Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name for name in selected):
        raise ValueError('Klasa musi być pojedynczą nazwą folderu.')
    return {name: weights[name] for name in selected}


def allocate(capacities, total, minimum=None):
    """Rozdziela dokładną liczebność proporcjonalnie, z pokryciem wszystkich klas."""
    total = int(total)
    if total < 0 or total > sum(capacities.values()):
        raise ValueError(f'Niedostateczna pula: wymagane {total}, dostępne {sum(capacities.values())}.')
    result = {k: int((minimum or {}).get(k, 0)) for k in capacities}
    if not minimum and total >= len(capacities):
        result = {k: min(1, n) for k, n in capacities.items()}
    if any(result[k] > capacities[k] for k in result) or sum(result.values()) > total:
        raise ValueError('Nieprawidłowe minimalne liczebności klas.')
    weight = sum(capacities.values())
    while sum(result.values()) < total:
        candidates = [k for k in capacities if result[k] < capacities[k]]
        key = max(candidates, key=lambda k: (total * capacities[k] / weight - result[k], k))
        result[key] += 1
    return result


def inventory(root, classes):
    root = Path(root).resolve()
    result = {}
    for name in classes:
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Brak klasy {name}: {folder}')
        result[name] = [p.relative_to(root).as_posix() for p in sorted(folder.iterdir())
                        if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if not result[name]:
            raise ValueError(f'Pusta klasa: {name}')
    return result


def create_manifest(real_root, classes=None, class_file=DEFAULT_CLASSES, seeds=(42, 43, 44),
                    synthetic_total=None, reference_total=None):
    weights = load_classes(class_file, classes)
    files = inventory(real_root, weights)
    total_real = sum(len(paths) for paths in files.values())
    if reference_total is None:
        reference_total = max(1, round(total_real * .01))
    refs_n = allocate({k: len(v) for k, v in files.items()}, reference_total)
    refs = {k: v[:refs_n[k]] for k, v in files.items()}
    targets = ({k: len(v) for k, v in files.items()} if synthetic_total is None
               else allocate({k: len(v) for k, v in files.items()}, synthetic_total))
    manifest = dict(version=1, classes=list(weights), real_root=str(Path(real_root).resolve()),
                    synthetic_targets=targets, references=refs, seeds={})
    for seed in seeds:
        rng = random.Random(seed)
        phases = {phase: {} for phase in ('train', 'val', 'test')}
        for name, paths in files.items():
            remaining = [p for p in paths if p not in set(refs[name])]
            rng.shuffle(remaining)
            n = int(len(paths) * .15)
            if len(remaining) < 2*n:
                raise ValueError(f'Zbyt mało próbek poza referencjami: {name}')
            phases['test'][name] = remaining[:n]
            phases['val'][name] = remaining[n:2*n]
            phases['train'][name] = refs[name] + remaining[2*n:]
        budgets = {}
        capacities = {k: len(v) for k, v in phases['train'].items()}
        totals = {budget: (sum(capacities.values()) if budget == 100 else round(total_real * budget / 100))
                  for budget in BUDGETS}
        for budget, total in totals.items():
            counts = allocate(capacities, total, refs_n)
            budgets[str(budget)] = {k: phases['train'][k][:counts[k]] for k in weights}
        manifest['seeds'][str(seed)] = dict(**phases, budgets=budgets)
    return manifest


def save_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = json.loads(path.read_text(encoding='utf-8'))
        if current != manifest:
            raise FileExistsError('Manifest już istnieje i opisuje inne dane lub ustawienia.')
        return
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def read_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('version') != 1:
        raise ValueError('Nieobsługiwana wersja manifestu obrazów.')
    references = {p for paths in data['references'].values() for p in paths}
    for state in data['seeds'].values():
        groups = {phase: {p for paths in state[phase].values() for p in paths}
                  for phase in ('train', 'val', 'test')}
        if (groups['train'] & (groups['val'] | groups['test']) or groups['val'] & groups['test']
                or not references <= groups['train'] or references & (groups['val'] | groups['test'])):
            raise ValueError('Manifest zawiera nakładające się podziały lub referencje poza treningiem.')
        for subset in state['budgets'].values():
            selected = {p for paths in subset.values() for p in paths}
            if not references <= selected or not selected <= groups['train']:
                raise ValueError('Wycinek budżetu wykracza poza trening lub pomija referencje.')
    return data


def split_from_manifest(sources, manifest, seed, variant, budget):
    """Zwraca wybrane próbki w formacie oczekiwanym przez aplikację."""
    labels = manifest['classes']
    state = manifest['seeds'][str(seed)]
    real_source = next(s for s in sources if s['domain'] == 'Real')
    synth_source = next((s for s in sources if s['domain'] == 'Synthetic'), None)
    splits = {k: ([], [], []) for k in ('pretrain', 'train', 'val', 'test')}
    def add(phase, root, grouped):
        for label in labels:
            for relative in grouped[label]:
                path = Path(root) / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                splits[phase][0].append(str(path))
                splits[phase][1].append(label)
                splits[phase][2].append((False, False))
    for phase in ('val', 'test'):
        add(phase, real_source['path'], state[phase])
    if variant != 'E':
        add('train', real_source['path'], state['budgets'][str(budget)])
    if synth_source:
        available = inventory(synth_source['path'], labels)
        targets = {k: len(v) for k, v in available.items()}
        rng = random.Random(seed)
        for paths in available.values():
            rng.shuffle(paths)
        total = (sum(targets.values()) if variant in ('A', 'D') else
                 (sum(len(v) for v in state['train'].values()) if int(budget) == 100 else round(sum(targets.values()) * int(budget) / 100)))
        counts = targets if variant in ('A', 'D') else allocate(targets, total)
        chosen = {k: available[k][:counts[k]] for k in labels}
        add('pretrain' if variant == 'A' else 'train', synth_source['path'], chosen)
    return splits, labels, {k: i for i, k in enumerate(labels)}


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_classes(manifest=None, class_file=DEFAULT_CLASSES):
    return read_manifest(manifest)['classes'] if manifest else list(load_classes(class_file))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--classes', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--synthetic-total', type=int, help='Opcjonalne nadpisanie globalnego celu; domyślnie liczebności rzeczywistych klas.')
    args = parser.parse_args()
    data = create_manifest(args.real, args.classes, args.class_file, args.seeds, args.synthetic_total)
    save_manifest(args.output, data)
    print(args.output)


if __name__ == '__main__':
    main()
