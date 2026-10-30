"""Kontrole wzorów i rozróżnienia wystąpień wyrazów od obecności w wiadomości."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from analizy.podobienstwo import intra
from analizy.sms import analyze, tokens
from wiadomosci.eksperymenty import f05


class AnalysisTests(unittest.TestCase):
    def test_cosine_average_excludes_diagonal(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(17, 512))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        matrix = vectors @ vectors.T
        self.assertAlmostEqual(intra(vectors), matrix[np.triu_indices(17, 1)].mean(), places=12)
        other = vectors[:7]
        self.assertAlmostEqual(float(vectors.mean(axis=0) @ other.mean(axis=0)),
                               float((vectors @ other.T).mean()), places=12)

    def test_f05_is_not_f1(self):
        result = f05(dict(tp=4, fp=1, tn=10, fn=5))
        self.assertAlmostEqual(result, 5 / 7.25)
        self.assertNotAlmostEqual(result, 8 / 14)

    def test_text_counts_all_occurrences_and_keeps_function_words(self):
        self.assertEqual(tokens("I, YOU! don't 4"), ['i', 'you', "don't", '4'])
        temporary_root = Path(__file__).resolve().parents[1] / 'wyniki/testy'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            path = Path(folder) / 'sms.csv'
            with path.open('w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['label', 'message'])
                writer.writerows([['HAM', 'A a b'], ['HAM', 'a A B'], ['SPAM', 'Win now'], ['SPAM', 'Reply win']])
            summaries, ranks = analyze(path, path)
            group = [r for r in ranks if r['zbior'] == 'Rzeczywiste' and r['klasa'] == 'HAM' and r['miara'] == 'slowa']
            self.assertEqual(group[0]['element'], 'a')
            self.assertEqual(group[0]['wystapienia'], 4)
            self.assertAlmostEqual(group[0]['skumulowany_udzial_proc'], 100 * 4/6)
            ham = next(r for r in summaries if r['zbior'] == 'Rzeczywiste' and r['klasa'] == 'HAM')
            self.assertEqual(ham['duplikaty_proc'], 50)


if __name__ == '__main__':
    unittest.main()
