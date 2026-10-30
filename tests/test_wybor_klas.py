"""Wybór klas i potwierdzenie usuwania na izolowanych katalogach testowych."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ClassSelectionTests(unittest.TestCase):
    def test_deletion_requires_explicit_confirmation(self):
        parent = ROOT / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            folder = Path(directory)
            (folder/'archery').mkdir()
            (folder/'other').mkdir()
            (folder/'other/sample.jpg').write_bytes(b'test')
            selection = folder/'classes.txt'
            selection.write_text('archery\n')
            command = [sys.executable, '-m', 'obrazy.wybor_klas', '--root', str(folder),
                       '--class-file', str(selection)]
            for arguments, answer in ((command, ''), (command+['--remove-others'], 'NIE\n')):
                result = subprocess.run(arguments, input=answer.encode(), cwd=ROOT, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((folder/'other/sample.jpg').exists())
            result = subprocess.run(command+['--remove-others'], input=b'USUN 1\n', cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((folder/'archery').is_dir())
            self.assertFalse((folder/'other').exists())

    def test_counting_ignores_unselected_folders(self):
        parent = ROOT / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            for name in ('archery', 'other'):
                (root/name).mkdir()
                (root/name/'sample.jpg').touch()
            selection, output = root/'classes.txt', root/'counts.csv'
            selection.write_text('archery\n')
            result = subprocess.run([sys.executable, '-m', 'analizy.liczebnosci', '--images', str(root),
                                     '--class-file', str(selection), '--output', str(output)],
                                    cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['klasa'] for row in rows], ['archery'])
