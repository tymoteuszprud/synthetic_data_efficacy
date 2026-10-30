"""Kontrole wyboru danych i losowania klas, bez modeli."""
from collections import Counter
import ast
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from wiadomosci.dane import split_real_dataset, get_training_subset, training_data, pretrain_cache_path
from wiadomosci.balansowanie import attach_identifiers, balance_dataframe
from wiadomosci.eksperymenty import load_data, audit


class DataTests(unittest.TestCase):
    def setUp(self):
        self.real = pd.DataFrame(dict(label=['HAM']*850 + ['SPAM']*150,
                                      message=[f'wiadomość {i}' for i in range(1000)]))
        self.refs = pd.DataFrame(dict(df_index=[0, 1, 2, 3, 4, 850, 851, 852, 853, 854]))
        self.synth = pd.DataFrame(dict(label=['HAM']*500 + ['SPAM']*500,
                                       message=[f'syntetyczna {i}' for i in range(1000)]))

    def test_split_is_disjoint_and_reference_is_preserved(self):
        identified = attach_identifiers(self.real, 'real')
        for seed in (42, 43, 44):
            split = split_real_dataset(identified, self.refs, seed)
            val, test = set(split['val_indices']), set(split['test_indices'])
            self.assertFalse(val & test)
            self.assertFalse(set(self.refs.df_index) & (val | test))
            for budget, size in ((1, 10), (3, 30), (10, 100), (30, 300), (100, 700)):
                frame = get_training_subset(identified, split, budget, seed)
                ids = {int(value.split(':')[1]) for value in frame['_sample_id']}
                self.assertEqual(len(frame), size)
                self.assertFalse(ids & (val | test))
                self.assertTrue(set(self.refs.df_index) <= ids)

    def test_variants_preserve_sources_and_sizes(self):
        real = attach_identifiers(self.real.iloc[:30], 'real')
        synth = attach_identifiers(self.synth, 'synthetic')
        for variant, size in (('A', 30), ('B', 30), ('C', 60), ('D', 1030), ('E', 30)):
            frame = training_data(real, synth, variant, 42)
            self.assertEqual(len(frame), size)
            if variant == 'C':
                self.assertEqual(Counter(frame['_source']), dict(real=30, synthetic=30))
            if variant == 'E':
                self.assertEqual(set(frame['_source']), {'synthetic'})
            pd.testing.assert_frame_equal(frame, training_data(real, synth, variant, 42))

    def test_all_budgets_are_nested_for_each_seed(self):
        real = attach_identifiers(self.real, 'real')
        synth = attach_identifiers(self.synth, 'synthetic')
        for seed in (42, 43, 44):
            split = split_real_dataset(real, self.refs, seed)
            previous_real, previous_synth = [], []
            for budget in (1, 3, 10, 30, 100):
                subset = get_training_subset(real, split, budget, seed)
                ids = subset['_sample_id'].tolist()
                self.assertEqual(ids[:len(previous_real)], previous_real)
                synthetic = training_data(subset, synth, 'E', seed)['_sample_id'].tolist()
                self.assertEqual(synthetic[:len(previous_synth)], previous_synth)
                mixed = training_data(subset, synth, 'C', seed)
                self.assertEqual(set(mixed.loc[mixed['_source'] == 'synthetic', '_sample_id']), set(synthetic))
                self.assertEqual(set(mixed.loc[mixed['_source'] == 'real', '_sample_id']), set(ids))
                # Wynik nie zależy od wcześniejszego wywołania innego budżetu.
                pd.testing.assert_frame_equal(subset, get_training_subset(real, split, budget, seed))
                previous_real, previous_synth = ids, synthetic

    def test_shared_core_selects_audited_stages_in_both_modes(self):
        real = attach_identifiers(self.real, 'real')
        synth = attach_identifiers(self.synth, 'synthetic')
        path = Path(__file__).resolve().parents[1] / 'wiadomosci/rdzen.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'dane']
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('prepare_experiment_data', 'verify_training_plan')]
        selected = ast.fix_missing_locations(ast.Module(body=imports + functions, type_ignores=[]))
        ns = dict(__name__='wiadomosci.test_selection', __package__='wiadomosci')
        exec(compile(selected, str(path), 'exec'), ns)
        for balanced in (False, True):
            ns['BALANCED'] = balanced
            plans = audit((self.real, self.synth, self.refs), list('ABCDE'), [42, 43, 44], [1, 3, 10, 30, 100], balanced)
            for plan in plans:
                variant, seed, budget = plan['variant'], plan['seed'], plan['percentage']
                split = split_real_dataset(real, self.refs, seed)
                actual, pretrain, sampling, pretrain_sampling = ns['prepare_experiment_data'](
                    real, split, synth, variant, seed, budget)
                ns['_EXPECTED_PLAN'] = plan
                ns['verify_training_plan'](actual, pretrain, sampling, pretrain_sampling)
                self.assertEqual(actual['_sample_id'].tolist(), plan['train_rows'])
                if variant == 'A':
                    self.assertEqual(pretrain['_sample_id'].tolist(), plan['pretrain_rows'])
                    self.assertEqual(pretrain_sampling, plan['pretrain_sampling'])
                else:
                    self.assertIsNone(pretrain)
            wrong = actual.copy()
            wrong.loc[0, '_sample_id'] = 'inny:0'
            with self.assertRaises(ValueError):
                ns['verify_training_plan'](wrong, pretrain, sampling, pretrain_sampling)

    def test_pretrain_cache_isolated_by_mode_identity_and_seed(self):
        standard = pretrain_cache_path('work', 'identity1', 42, False)
        paths = {standard, pretrain_cache_path('work', 'identity1', 42, True),
                 pretrain_cache_path('work', 'identity2', 42, False),
                 pretrain_cache_path('work', 'identity1', 43, False)}
        self.assertEqual(len(paths), 4)

    def test_input_files_use_public_names(self):
        parent = Path(__file__).resolve().parents[1] / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            folder = Path(folder)
            self.real.to_csv(folder / 'wiadomosci_rzeczywiste.csv', index=False)
            self.synth.to_csv(folder / 'wiadomosci_syntetyczne.csv', index=False)
            self.refs.to_csv(folder / 'indeksy_referencji.csv', index=False)
            real, synth, refs = load_data(folder)
            pd.testing.assert_frame_equal(real, self.real)
            pd.testing.assert_frame_equal(synth, self.synth)
            pd.testing.assert_frame_equal(refs, self.refs)

    def test_audit_includes_both_stages_of_a(self):
        for balanced in (False, True):
            plans = audit((self.real, self.synth, self.refs), ['A'], [42], [1, 3], balanced)
            self.assertEqual(plans[0]['pretrain_rows'], plans[1]['pretrain_rows'])
            self.assertEqual(len(plans[0]['pretrain_rows']), len(self.synth))
            self.assertEqual(len(plans[0]['train_rows']), 10)
            if balanced:
                self.assertIsNotNone(plans[0]['pretrain_sampling'])

    def test_balancing_preserves_length_sources_and_random_state(self):
        frame = pd.concat([attach_identifiers(self.real, 'real'),
                           attach_identifiers(self.synth, 'synthetic')]).reset_index(drop=True)
        before = np.random.get_state()
        balanced, audit = balance_dataframe(frame, 42, 'D', 30, 'standard')
        self.assertEqual(len(frame), len(balanced))
        self.assertEqual(Counter(frame['_source']), Counter(balanced['_source']))
        self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
        self.assertFalse(audit['sources']['synthetic']['resampled'])
        self.assertLess(abs(audit['sources']['real']['class_counts_after']['HAM']-500), 100)
        repeated, repeated_audit = balance_dataframe(frame, 42, 'D', 30, 'standard')
        pd.testing.assert_frame_equal(balanced, repeated)
        self.assertEqual(audit['selection_sha256'], repeated_audit['selection_sha256'])


if __name__ == '__main__':
    unittest.main()
