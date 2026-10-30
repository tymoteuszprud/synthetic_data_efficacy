"""Kontrola wejść rankingu bez importowania modeli."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RankingTests(unittest.TestCase):
    def test_audit_does_not_load_model_or_create_results(self):
        root = Path(__file__).resolve().parents[1]
        parent = root / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            directory = Path(temp)
            model, images, output = directory/'model', directory/'images', directory/'ranking'
            model.mkdir()
            (images/'archery').mkdir(parents=True)
            (images/'archery/test.png').write_bytes(b'not a real image')
            (model/'best_model.pth').write_bytes(b'not a model')
            (directory/'classes.txt').write_text('archery\n')
            (model/'metadata.json').write_text(json.dumps(dict(classes=['archery'], params=dict(dropout=.5))))
            process = subprocess.run([sys.executable, '-m', 'analizy.ranking_straty',
                                      '--model-dir', str(model), '--synthetic', str(images),
                                      '--output', str(output), '--class-file', str(directory/'classes.txt')], cwd=root, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
