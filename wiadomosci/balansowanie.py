"""Losowanie klas przy zachowaniu liczby wystąpień i udziału źródeł."""

import argparse

import ast

import csv

import hashlib

import importlib.metadata

import json

import math

import random

import shutil

from collections import Counter

from pathlib import Path

METHOD = 'per_source_equal_probability_with_replacement_fixed_length_v1'

def remove_generated_directory(path):
    root = (Path.cwd() / 'balanced_experiment_outputs').resolve()
    target = Path(path).resolve()
    if root not in target.parents or target == root:
        raise ValueError(f'Odmowa usunięcia folderu poza katalogiem wyników eksperymentów z losowaniem klas: {target}')
    shutil.rmtree(target)

def attach_identifiers(frame, source):
    frame = frame.copy()
    frame['label'] = frame['label'].astype(str).str.strip().str.upper()
    if not frame['label'].isin(['HAM', 'SPAM']).all():
        raise ValueError(f'Nieznane etykiety w zbiorze {source}.')
    if frame['message'].isna().any():
        raise ValueError(f'Brakujące wiadomości w zbiorze {source}.')
    frame['message'] = frame['message'].astype(str)
    frame['_source'] = source
    frame['_sample_id'] = [f'{source}:{i}' for i in range(len(frame))]
    return frame

def _counts(values):
    counts = Counter(values)
    return {label: int(counts.get(label, 0)) for label in ('HAM', 'SPAM')}

def balance_dataframe(frame, seed, variant, percentage, stage):


    if frame.empty:
        raise ValueError('Nie można balansować pustego wycinka.')
    labels = frame['label'].tolist()
    sources = frame['_source'].tolist()
    identifiers = frame['_sample_id'].tolist()
    selected = list(range(len(frame)))
    source_audit = {}

    sampling_percentage = None if stage == 'pretrain' else percentage
    key = f'{METHOD}|{seed}|{variant}|{sampling_percentage}|{stage}'
    sampling_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
    rng = random.Random(sampling_seed)
    for source in sorted(set(sources)):
        positions = [i for i, value in enumerate(sources) if value == source]
        by_class = {label: [i for i in positions if labels[i] == label]
                    for label in ('HAM', 'SPAM')}
        if not all(by_class.values()):
            raise ValueError(f'{variant}/{seed}/{percentage}/{stage}: źródło {source} nie zawiera obu klas.')
        before = _counts(labels[i] for i in positions)
        changed = before['HAM'] != before['SPAM']
        if changed:
            for destination in positions:
                chosen_class = 'HAM' if rng.random() < 0.5 else 'SPAM'
                selected[destination] = rng.choice(by_class[chosen_class])
        chosen_ids = [identifiers[selected[i]] for i in positions]
        after = _counts(labels[selected[i]] for i in positions)
        source_audit[source] = {
            'resampled': changed, 'rows_before': len(positions), 'rows_after': len(positions),
            'class_counts_before': before, 'class_counts_after': after,
            'expected_class_probability': {'HAM': 0.5, 'SPAM': 0.5},
            'unique_rows_available': len(set(identifiers[i] for i in positions)),
            'unique_rows_used': len(set(chosen_ids)),
            'repeat_occurrences': len(chosen_ids) - len(set(chosen_ids)),
        }
    result = frame.iloc[selected].reset_index(drop=True)
    selected_ids = result['_sample_id'].tolist()
    audit = {
        'method': METHOD, 'stage': stage, 'sampling_seed': sampling_seed,
        'rows_before': len(frame), 'rows_after': len(result),
        'class_counts_before': _counts(labels), 'class_counts_after': _counts(result['label']),
        'unique_rows_available': len(set(identifiers)), 'unique_rows_used': len(set(selected_ids)),
        'repeat_occurrences': len(result) - len(set(selected_ids)),
        'selection_sha256': hashlib.sha256('\n'.join(selected_ids).encode()).hexdigest(),
        'sources': source_audit,
    }
    assert len(result) == len(frame)
    assert Counter(result['_source']) == Counter(sources)
    for source, values in source_audit.items():
        if not values['resampled']:
            positions = [i for i, value in enumerate(sources) if value == source]
            assert all(selected[i] == i for i in positions)
    return result, audit

def save_manifest(frame, audit, folder, variant, seed, percentage):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f'{variant}_seed_{seed}_budget_{percentage}_{audit["stage"]}.csv'
    path = folder / name
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['position', 'sample_id', 'source', 'label'])
        for i, (_, row) in enumerate(frame.iterrows()):
            writer.writerow([i, row['_sample_id'], row['_source'], row['label']])
    audit['manifest'] = str(path.resolve())
