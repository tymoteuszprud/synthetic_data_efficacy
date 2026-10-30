"""Uruchamianie eksperymentów tekstowych; domyślnie tylko kontrola danych."""
import argparse
from collections import Counter
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

from .dane import split_real_dataset, get_training_subset, prepare_training_stages, pretrain_cache_path
from .balansowanie import attach_identifiers

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(folder):
    import pandas as pd
    real = pd.read_csv(folder / 'wiadomosci_rzeczywiste.csv')
    synth = pd.read_csv(folder / 'wiadomosci_syntetyczne.csv')
    refs = pd.read_csv(folder / 'indeksy_referencji.csv')
    for source, frame in (('real', real), ('synthetic', synth)):
        if not {'label', 'message'}.issubset(frame.columns):
            raise ValueError(f'{source}: wymagane kolumny label i message.')
        if frame[['label', 'message']].isna().any().any():
            raise ValueError(f'{source}: brakujące wiadomości lub etykiety.')
        frame['label'] = frame['label'].astype(str).str.strip().str.upper()
        frame['message'] = frame['message'].astype(str)
        if not frame['label'].isin(('HAM', 'SPAM')).all():
            raise ValueError(f'{source}: nieznane etykiety.')
    indices = refs['df_index'].tolist()
    if len(indices) != len(set(indices)) or any(not isinstance(i, int) or i < 0 or i >= len(real) for i in indices):
        raise ValueError('Nieprawidłowe indeksy referencji.')
    return real, synth, refs


def audit(frames, variants, seeds, budgets, balanced):
    real, synth, refs = frames
    real = attach_identifiers(real, 'real')
    synth = attach_identifiers(synth, 'synthetic')
    plans = []
    for seed in seeds:
        split = split_real_dataset(real, refs, seed)
        val, test = set(split['val_indices']), set(split['test_indices'])
        for budget in budgets:
            subset = get_training_subset(real, split, budget, seed)
            train_ids = [int(i.split(':')[1]) for i in subset['_sample_id']]
            if val & test or set(train_ids) & (val | test):
                raise ValueError('Zbiory treningowy, walidacyjny i testowy nakładają się.')
            for variant in variants:
                training, pretrain, sampling, pretrain_sampling = prepare_training_stages(
                    subset, synth, variant, seed, budget, balanced)
                plan = dict(variant=variant, seed=seed, percentage=budget,
                                  train_rows=training['_sample_id'].tolist(),
                                  train_counts=dict(Counter(training['label'])),
                                  val_indices=list(map(int, split['val_indices'])),
                                  test_indices=list(map(int, split['test_indices'])),
                                  sampling=sampling)
                if variant == 'A':
                    plan.update(pretrain_rows=pretrain['_sample_id'].tolist(),
                                pretrain_counts=dict(Counter(pretrain['label'])),
                                pretrain_sampling=pretrain_sampling)
                plans.append(plan)
    return plans


def f05(metrics):
    denominator = 1.25*metrics['tp'] + metrics['fp'] + .25*metrics['fn']
    return 1.25*metrics['tp']/denominator if denominator else 0.0


def training_stage(variant, config_calls, cache):
    if variant != 'A':
        return 'main'
    cached = (cache / 'pretrain_history.json').exists()
    return 'pretrain' if config_calls == 0 and not cached else 'finetune'


def train(frames, plans, config, output, identity, retry):
    # Ciężkie biblioteki są importowane dopiero po jawnej zgodzie na trening.
    core = importlib.import_module('wiadomosci.rdzen')
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    core.MODEL_NAME = config['model']
    core.EPOCHS_STANDARD = config['epochs']
    core.EPOCHS_FINETUNE = config['finetune_epochs']
    core.EPOCHS_PRETRAIN = config['pretrain_epochs']
    core.LR_STANDARD = config['learning_rate']
    core.LR_FINETUNE = config['finetune_learning_rate']
    core.LR_PRETRAIN = config['pretrain_learning_rate']
    core.MAX_SEQ_LENGTH = config['max_seq_length']
    core.BALANCED = config['balanced']
    core._RUN_FINGERPRINT, core._RUN_PROVENANCE = fingerprint, identity
    real, synth, refs = frames
    real, synth = attach_identifiers(real, 'real'), attach_identifiers(synth, 'synthetic')
    work = output / 'work'
    work.mkdir(exist_ok=True)
    core._CACHE_ROOT = str(work)
    base_config = core.SFTConfig
    base_callback = core.CustomEvalCallback
    base_evaluate = core.evaluate_classification
    state = {}

    def configure(*a, **kw):
        # Stan jest odczytywany ponownie po ewentualnym usunięciu uszkodzonego modelu.
        stage = training_stage(state['variant'], state['config_calls'], state['cache'])
        state['config_calls'] += 1
        cap = config.get('microbatch_100')
        if cap and state['budget'] == 100 and stage != 'pretrain':
            before = kw['per_device_train_batch_size']
            effective = before * kw['gradient_accumulation_steps']
            after = min(before, cap)
            if effective % after:
                raise ValueError('Zmiana partii nie zachowuje efektywnego rozmiaru.')
            kw['per_device_train_batch_size'] = after
            kw['gradient_accumulation_steps'] = effective // after
        state['stage'] = stage
        kw['output_dir'] = str(state['folder'] / ('trainer_' + stage))
        resolved = base_config(*a, **kw)
        save(state['folder'] / ('config_' + stage + '.json'), resolved.to_dict())
        return resolved

    def evaluate(*a, **kw):
        result = base_evaluate(*a, **kw)
        state['last_metrics'] = result
        return result

    class Callback(base_callback):
        def on_evaluate(self, args, trainer_state, control, metrics=None, **kwargs):
            super().on_evaluate(args, trainer_state, control, metrics=metrics, **kwargs)
            if self.epoch_logs:
                result = state['last_metrics']
                self.epoch_logs[-1]['val_f05'] = f05(result)
                save(state['folder'] / ('validation_' + state['stage'] + '.json'), self.epoch_logs)

    core.SFTConfig, core.CustomEvalCallback, core.evaluate_classification = configure, Callback, evaluate
    try:
        records = {}
        for saved in sorted(output.glob('*_seed_*_budget_*/result.json')):
            record = read(saved)
            if record.get('provenance') != identity:
                raise RuntimeError(f'Inne pochodzenie zapisanego wyniku: {saved}')
            key = (record['variant'], record['seed'], record['percentage'])
            if key in records:
                raise RuntimeError(f'Powtórzony zapis wyniku: {key}')
            records[key] = record
        for plan in plans:
            variant, seed, budget = plan['variant'], plan['seed'], plan['percentage']
            key = (variant, seed, budget)
            folder = output / f'{variant}_seed_{seed}_budget_{budget}'
            result_file = folder / 'result.json'
            if result_file.exists():
                print(f'Pominięto ukończony przebieg: {folder.name}', flush=True)
                continue
            if folder.exists() and not retry:
                raise RuntimeError(f'Przerwany przebieg: {folder.name}. Dodaj --retry-incomplete, aby zacząć go od początku.')
            folder.mkdir(exist_ok=True)
            save(folder / 'data.json', plan)
            cache = pretrain_cache_path(work, fingerprint, seed, config['balanced'])
            core._EXPECTED_PLAN = plan
            state.clear()
            state.update(folder=folder, variant=variant, budget=budget, config_calls=0,
                         cache=cache)
            core.RESULTS_FILE = str(folder / 'original_result.json')
            tracker = []
            previous = Path.cwd()
            try:
                os.chdir(work)
                split = core.split_real_dataset(real, refs, seed)
                core.run_single_experiment(variant, seed, budget, real, split, synth, tracker)
            finally:
                os.chdir(previous)
            record = tracker[0]
            record['test_results']['f0_5'] = f05(record['test_results'])
            for name in ('targets', 'predictions'):
                if name in state['last_metrics']:
                    record['test_results'][name] = state['last_metrics'][name]
            record['configuration'] = config
            record['provenance'] = identity
            save(result_file, record)
            records[key] = record
            save(output / 'results.json', list(records.values()))
            print(f"Gotowe {folder.name}: F0.5={record['test_results']['f0_5']:.6f}", flush=True)
        save(output / 'results.json', list(records.values()))
    finally:
        core.SFTConfig, core.CustomEvalCallback, core.evaluate_classification = base_config, base_callback, base_evaluate
        core._EXPECTED_PLAN = None



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'wyniki/tekst')
    parser.add_argument('--config', type=Path, default=ROOT / 'konfiguracje/tekst.json')
    parser.add_argument('--variants', nargs='+', choices=list('ABCDE'), default=list('ABCDE'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--budgets', nargs='+', type=int, choices=[1, 3, 10, 30, 100], default=[1, 3, 10, 30, 100])
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    config['balanced'] = args.balanced
    folder = args.data_dir.resolve()
    frames = load_data(folder)
    plans = audit(frames, args.variants, args.seeds, args.budgets, args.balanced)
    input_names = ('wiadomosci_rzeczywiste.csv', 'wiadomosci_syntetyczne.csv', 'indeksy_referencji.csv')
    identity = dict(config=config, inputs={n: sha(folder / n) for n in input_names},
                    code={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'wiadomosci').glob('*.py'))})
    if args.train:
        identity['versions'] = {n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'torch', 'transformers', 'trl', 'unsloth', 'datasets')}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Nowa konfiguracja lub dane otrzymują osobny katalog, również dla cache A.
    output = args.output_dir.resolve() / fingerprint[:16]
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'audit.json', dict(provenance=identity, plans=plans, training_requested=args.train))
    for plan in plans:
        print(f"{plan['variant']} seed={plan['seed']} budget={plan['percentage']}% N={len(plan['train_rows'])} {plan['train_counts']}")
    print(f'Katalog wyników: {output}')
    if not args.train:
        print('Kontrola zakończona. Nie ładowano modeli; trening wymaga --train.')
        return
    lock = output / 'training.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        train(frames, plans, config, output, identity, args.retry_incomplete)
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == '__main__':
    main()
