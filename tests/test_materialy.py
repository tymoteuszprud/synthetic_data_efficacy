"""Kontrole słownika oraz wyszukiwania materiałów bez uruchamiania interfejsu."""
import ast
from pathlib import Path
import unittest
import tempfile
import re

from generowanie.pobierz_slownik import validate

ROOT = Path(__file__).resolve().parents[1]


class MaterialsTests(unittest.TestCase):
    def test_image_names_do_not_reuse_deleted_indices(self):
        tree = ast.parse((ROOT / 'generowanie/obrazy_gui.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'next_image_path')
        scope = dict(Path=Path, re=re)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<next_image_path>', 'exec'), scope)
        parent = ROOT / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            directory = Path(folder)
            for name in ('synth_00000.png', 'synth_00002.png'):
                (directory / name).write_bytes(b'existing')
            self.assertEqual(scope['next_image_path'](directory).name, 'synth_00003.png')
            self.assertEqual((directory / 'synth_00002.png').read_bytes(), b'existing')

    def test_sms_instructions_are_loaded_from_files(self):
        path = ROOT / 'generowanie/sms_gui.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_instruction')
        scope = dict(Path=Path, __file__=str(path))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<load_instruction>', 'exec'), scope)
        for label in ('ham', 'spam'):
            text = scope['load_instruction'](label, 'REFERENCE_MESSAGE', 'SEED_WORDS')
            self.assertIn('REFERENCE_MESSAGE', text)
            self.assertIn('SEED_WORDS', text)
            self.assertIn(label.upper() + ' SMS', text)
            self.assertNotIn('{examples_str}', text)

    def test_dictionary_format(self):
        self.assertEqual(validate(b'{"word": 1, "message": 1}'), 2)
        for malformed in (b'[]', b'{}', b'{"":1}'):
            with self.assertRaises(ValueError):
                validate(malformed)

    def test_bundled_words(self):
        words = (ROOT / 'materialy/slownictwo_podstawowe.txt').read_text(encoding='utf-8').splitlines()
        self.assertGreaterEqual(len(words), 100)
        self.assertTrue(all(word.strip() for word in words))

    def test_generator_uses_material_directory(self):
        tree = ast.parse((ROOT / 'generowanie/sms_gui.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpamHamGeneratorApp')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'find_file')
        import os
        scope = dict(os=os)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<find_file>', 'exec'), scope)
        instance = type('Generator', (), {'script_dir': str(ROOT / 'generowanie')})()
        path = scope['find_file'](instance, 'slownictwo_podstawowe.txt')
        self.assertEqual(Path(path).resolve(), (ROOT / 'materialy/slownictwo_podstawowe.txt').resolve())

    def test_image_payload_supports_incomplete_final_batch(self):
        import json
        import random
        tree = ast.parse((ROOT / 'generowanie/obrazy_gui.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_payload')
        scope = dict(BATCH_SIZE=4, WIDTH=512, HEIGHT=512, STEPS=6, random=random)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<build_payload>', 'exec'), scope)
        workflow = json.loads((ROOT / 'konfiguracje/przeplyw_generowania_obrazow.json').read_text(encoding='utf-8'))
        before = json.dumps(workflow, sort_keys=True)
        for size in (1, 2, 3, 4):
            payload = scope['build_payload'](workflow, 'test scene', 'test_prefix', size)['prompt']
            latents = [n for n in payload.values() if n.get('class_type') == 'EmptySD3LatentImage']
            self.assertTrue(latents)
            self.assertTrue(all(n['inputs']['batch_size'] == size for n in latents))
            self.assertTrue(all(n['inputs']['width'] == n['inputs']['height'] == 512 for n in latents))
        self.assertEqual(json.dumps(workflow, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
