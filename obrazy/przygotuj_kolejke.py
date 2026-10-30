"""Przygotowuje kolejkę Streamlit z ustawień zapisanych w archiwum eksperymentów."""
import argparse
import csv
import json
from pathlib import Path
from obrazy.dane import DEFAULT_CLASSES, create_manifest, save_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'konfiguracje/obrazy.json')
    parser.add_argument('--variants', nargs='+', choices=list('ABCDE'), default=list('ABCDE'))
    parser.add_argument('--budgets', nargs='+', type=int, default=[1, 3, 10, 30, 100])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--classes', nargs='+', help='Nazwy klas z wykazu; domyślnie wszystkie 30.')
    parser.add_argument('--manifest', type=Path, help='Wspólny manifest; domyślnie manifest_obrazow.json w katalogu roboczym.')
    args = parser.parse_args()
    configs = json.loads(args.config.read_text(encoding='utf-8'))
    jobs = []
    for job in configs:
        if job['variant'] not in args.variants or job['budget'] not in args.budgets:
            continue
        jobs.append({k: v for k, v in job.items() if k != 'configuration_note'})
        print(job['exp_id'], job['runs_total'], 'uruchomienia')
    if not jobs:
        parser.error('Brak konfiguracji spełniających warunki.')
    if not args.write:
        print('Nie zapisano kolejki. Dodaj --write, aby przygotować job_queue.csv.')
        return
    args.workspace.mkdir(parents=True, exist_ok=True)
    path = args.workspace / 'job_queue.csv'
    if path.exists():
        raise FileExistsError('Plik job_queue.csv już istnieje; kolejka nie została nadpisana.')
    for job in jobs:
        for source in job['data_sources']:
            if not (args.workspace / source['path']).is_dir():
                raise FileNotFoundError(f"Brak źródła obrazów: {source['path']}")
    real = next(s for s in jobs[0]['data_sources'] if s['domain'] == 'Real')
    manifest_path = (args.manifest or args.workspace / 'manifest_obrazow.json').resolve()
    seeds = sorted({j['seed_base'] + i for j in jobs for i in range(j['runs_total'])})
    data = create_manifest(args.workspace / real['path'], args.classes, args.class_file, seeds)
    save_manifest(manifest_path, data)
    for job in jobs:
        job['manifest_path'] = str(manifest_path)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(jobs[0]))
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in job.items()})
    print(path)


if __name__ == '__main__':
    main()
