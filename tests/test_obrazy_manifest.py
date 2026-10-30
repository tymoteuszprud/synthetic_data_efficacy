"""Regresje wyboru klas, liczebności i ochrony referencji bez modeli i GPU."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from obrazy.dane import (DEFAULT_CLASSES, BUDGETS, allocate, create_manifest,
                         inventory, load_classes, save_manifest, read_manifest,
                         split_from_manifest)


class ImageManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = Path(__file__).resolve().parents[1] / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        cls.folder = tempfile.TemporaryDirectory(dir=parent)
        cls.root = Path(cls.folder.name)
        cls.real, cls.synth = cls.root/'real', cls.root/'synth'
        cls.weights = load_classes(DEFAULT_CLASSES.with_name('klasy_obrazow.csv'))
        for root in (cls.real, cls.synth):
            for name, count in cls.weights.items():
                (root/name).mkdir(parents=True)
                for i in range(count):
                    (root/name/f'{i:05}.jpg').touch()
            (root/'nie_wybrana_klasa').mkdir()
            (root/'nie_wybrana_klasa'/'extra.jpg').touch()
        cls.manifest = create_manifest(cls.real)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_default_thirty_names_ignore_extra(self):
        self.assertEqual(len(self.manifest['classes']), 30)
        self.assertNotIn('nie_wybrana_klasa', self.manifest['classes'])
        self.assertEqual(self.manifest['synthetic_targets'], self.weights)
        self.assertEqual(sum(self.manifest['synthetic_targets'].values()), 4683)
        with self.assertRaises(ValueError):
            load_classes(names=['nie_wybrana_klasa'])

    def test_reference_size_is_one_percent_of_actual_inventory(self):
        files = {'archery': [f'archery/{i}.jpg' for i in range(200)]}
        with patch('obrazy.dane.inventory', return_value=files):
            data = create_manifest(self.real, classes=['archery'])
        self.assertEqual(len(data['references']['archery']), 2)
        for state in data['seeds'].values():
            self.assertFalse(set(data['references']['archery']) & set(state['val']['archery']))
            self.assertFalse(set(data['references']['archery']) & set(state['test']['archery']))

    def test_references_and_all_seed_splits(self):
        refs = {p for paths in self.manifest['references'].values() for p in paths}
        self.assertEqual(len(refs), 47)
        for state in self.manifest['seeds'].values():
            phases = {k:{p for paths in state[k].values() for p in paths} for k in ('train','val','test')}
            self.assertEqual([len(phases[k]) for k in ('train','val','test')], [3305,689,689])
            self.assertTrue(refs <= phases['train'])
            self.assertFalse(refs & (phases['val']|phases['test']))
            self.assertFalse(phases['train'] & (phases['val']|phases['test']))
            self.assertFalse(phases['val'] & phases['test'])
            for budget,total in {1:47,3:140,10:468,30:1405,100:3305}.items():
                grouped=state['budgets'][str(budget)]
                values={p for paths in grouped.values() for p in paths}
                self.assertEqual(len(values),total)
                self.assertTrue(all(grouped.values()))
                self.assertTrue(refs <= values)

    def test_variant_sizes_and_shared_synthetic_source(self):
        expected = {1:(47,47),3:(140,140),10:(468,468),30:(1405,1405),100:(3305,3305)}
        sources=[dict(path=str(self.real),domain='Real'),dict(path=str(self.synth),domain='Synthetic')]
        for budget,(r,s) in expected.items():
            for variant in 'ABCDE':
                selected=sources[:1] if variant=='B' else sources
                phases,labels,_=split_from_manifest(selected,self.manifest,42,variant,budget)
                real_count=0 if variant=='E' else r
                synth_count=4683 if variant in 'AD' else (3305 if variant=='E' and budget==100 else s)
                self.assertEqual(len(phases['pretrain'][0]),4683 if variant=='A' else 0)
                self.assertEqual(len(phases['train'][0]),real_count+(0 if variant in 'AB' else synth_count))
                self.assertEqual(len(labels),30)
                self.assertTrue(all(not any(flag) for phase in phases.values() for flag in phase[2]))

    def test_manifest_no_silent_replacement(self):
        path=self.root/'manifest.json'
        save_manifest(path,self.manifest)
        save_manifest(path,self.manifest)
        self.assertEqual(read_manifest(path),self.manifest)
        different=dict(self.manifest, version=2)
        with self.assertRaises(FileExistsError):
            save_manifest(path,different)

    def test_read_rejects_reference_in_test(self):
        broken = json.loads(json.dumps(self.manifest))
        label = broken['classes'][0]
        broken['seeds']['42']['test'][label].append(broken['references'][label][0])
        path = self.root/'invalid_manifest.json'
        path.write_text(json.dumps(broken), encoding='utf-8')
        with self.assertRaises(ValueError):
            read_manifest(path)

    def test_config_has_same_source_and_no_additional_artifacts(self):
        path=DEFAULT_CLASSES.with_name('obrazy.json')
        for job in json.loads(path.read_text(encoding='utf-8')):
            self.assertEqual(job['aug_params']['crop_scale'], .8)
            self.assertEqual(job['aug_params']['rot_deg'], 15)
            self.assertEqual(job['aug_params']['jit_f'], .2)
            for source in job['data_sources']:
                self.assertNotIn('aug_noise', source)
                self.assertNotIn('aug_jpeg', source)
                if source['domain']=='Synthetic':
                    self.assertEqual(source['path'],'obrazy_syntetyczne')


if __name__ == '__main__':
    unittest.main()
