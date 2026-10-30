"""Sterowanie serią i wznowienie sprawdzane na zastępczym module treningowym."""
from pathlib import Path
import ast
from contextlib import nullcontext
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from wiadomosci import eksperymenty as runner
from wiadomosci.dane import split_real_dataset, pretrain_cache_path


class RunnerTests(unittest.TestCase):
    def test_single_core_runner_routes_both_modes_and_stages(self):
        real = pd.DataFrame(dict(label=['HAM']*850+['SPAM']*150,
                                 message=[str(i) for i in range(1000)]))
        synth = real.copy()
        refs = pd.DataFrame(dict(df_index=[0, 1, 850, 851]))
        frames = real, synth, refs

        class Config:
            def __init__(self, **kw):
                self.__dict__.update(kw)
            def to_dict(self):
                return self.__dict__

        class Callback:
            def on_evaluate(self, *args, **kwargs):
                pass

        core = SimpleNamespace(split_real_dataset=split_real_dataset, SFTConfig=Config,
                               CustomEvalCallback=Callback)
        def metrics(*args, **kwargs):
            return dict(tp=2, fp=1, tn=4, fn=1)
        core.evaluate_classification = metrics
        calls = []
        def run(variant, seed, budget, real, split, synth, tracker):
            self.assertEqual((core.EPOCHS_STANDARD, core.EPOCHS_PRETRAIN, core.EPOCHS_FINETUNE), (1, 1, 1))
            cache = pretrain_cache_path(core._CACHE_ROOT, core._RUN_FINGERPRINT, seed, core.BALANCED)
            if variant == 'A' and not (cache / 'pretrain_history.json').exists():
                resolved = core.SFTConfig(num_train_epochs=core.EPOCHS_PRETRAIN,
                                          per_device_train_batch_size=4, gradient_accumulation_steps=4)
                self.assertTrue(resolved.output_dir.endswith('trainer_pretrain'))
                cache.mkdir(parents=True)
                (cache / 'pretrain_history.json').write_text('[]', encoding='utf-8')
            resolved = core.SFTConfig(num_train_epochs=1, per_device_train_batch_size=4,
                                      gradient_accumulation_steps=4)
            self.assertTrue(resolved.output_dir.endswith('trainer_finetune' if variant == 'A' else 'trainer_main'))
            self.assertEqual(core._EXPECTED_PLAN['variant'], variant)
            self.assertIn('_sample_id', real.columns)
            core.evaluate_classification()
            tracker.append(dict(variant=variant, seed=seed, percentage=budget, test_results=metrics()))
            calls.append((core.BALANCED, variant, budget))
        core.run_single_experiment = run
        parent = Path(__file__).resolve().parents[1] / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            for balanced in (False, True):
                output = Path(folder) / str(balanced)
                output.mkdir()
                config = dict(model='testowy-model', balanced=balanced, epochs=1, finetune_epochs=1,
                              pretrain_epochs=1, learning_rate=2e-5, finetune_learning_rate=5e-6,
                              pretrain_learning_rate=2e-5, max_seq_length=512)
                plans = runner.audit(frames, list('ABCDE'), [42], [3, 100], balanced)
                with patch.object(runner.importlib, 'import_module', return_value=core) as imported:
                    runner.train(frames, plans, config, output, dict(balanced=balanced), False)
                    imported.assert_called_once_with('wiadomosci.rdzen')
                self.assertIs(core.SFTConfig, Config)
                self.assertIs(core.evaluate_classification, metrics)
                self.assertIsNone(core._EXPECTED_PLAN)
                self.assertEqual(len(runner.read(output / 'results.json')), 10)
        self.assertEqual(len(calls), 20)

    def test_shared_evaluation_without_model_or_gpu(self):
        import numpy as np
        path = Path(__file__).resolve().parents[1] / 'wiadomosci/rdzen.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'evaluate_classification']
        ns = dict(torch=SimpleNamespace(no_grad=nullcontext), SYSTEM_PROMPT='instrukcja', MAX_SEQ_LENGTH=512)
        exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(path), 'exec'), ns)
        class Batch(dict):
            def to(self, device):
                return self
        class Tokenizer:
            padding_side = 'right'
            def encode(self, label, **kwargs):
                return [0 if label == 'HAM' else 1]
            def __call__(self, prompts, **kwargs):
                self.length = kwargs['max_length']
                return Batch(input_ids=[1]*len(prompts), attention_mask=[1]*len(prompts))
        class Model:
            training = True
            rope_deltas = 123
            def eval(self):
                self.training = False
            def train(self):
                self.training = True
            def __call__(self, input_ids, attention_mask):
                return SimpleNamespace(logits=np.array([[[1., 2.]], [[3., 1.]]]))
        data = [dict(instruction='x', output='SPAM'), dict(instruction='y', output='HAM')]
        for balanced in (False, True):
            ns['BALANCED'] = balanced
            model, tokenizer = Model(), Tokenizer()
            result = ns['evaluate_classification'](model, tokenizer, data)
            self.assertEqual(result['predictions'], ['SPAM', 'HAM'])
            self.assertEqual((result['tp'], result['tn'], result['fp'], result['fn']), (1, 1, 0, 0))
            self.assertTrue(model.training)
            self.assertIsNone(model.rope_deltas)
            self.assertEqual(tokenizer.padding_side, 'right')
            self.assertEqual(tokenizer.length, 512)

    def test_rebuilt_cache_uses_pretrain_configuration(self):
        parent = Path(__file__).resolve().parents[1] / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            cache = Path(folder)
            history = cache / 'pretrain_history.json'
            history.write_text('[]', encoding='utf-8')
            self.assertEqual(runner.training_stage('A', 0, cache), 'finetune')
            history.unlink()  # Rdzeń odrzucił niekompletny lub uszkodzony model.
            self.assertEqual(runner.training_stage('A', 0, cache), 'pretrain')
            self.assertEqual(runner.training_stage('A', 1, cache), 'finetune')
            self.assertEqual(runner.training_stage('B', 0, cache), 'main')

    def test_resume_and_effective_batch_without_gpu(self):
        real = pd.DataFrame(dict(label=['HAM']*850+['SPAM']*150,
                                  message=[f'tekst {i}' for i in range(1000)]))
        synth = real.copy()
        refs = pd.DataFrame(dict(df_index=[0, 1, 2, 3, 4, 850, 851, 852, 853, 854]))
        frames = (real, synth, refs)
        plans = runner.audit(frames, ['B'], [42, 43], [100], False)
        config = dict(model='testowy-model', balanced=False, epochs=4, finetune_epochs=4,
                      pretrain_epochs=1, learning_rate=2e-5, finetune_learning_rate=5e-6,
                      pretrain_learning_rate=2e-5, max_seq_length=512, microbatch_100=2)
        core = SimpleNamespace(split_real_dataset=split_real_dataset)

        class Config:
            def __init__(self, **kw):
                self.__dict__.update(kw)
            def to_dict(self):
                return self.__dict__

        class Callback:
            def __init__(self):
                self.epoch_logs = []
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                core.evaluate_classification()
                self.epoch_logs.append(dict(epoch=state.epoch))

        def metrics():
            return dict(tp=4, fp=1, tn=10, fn=5)

        calls = []
        def run(variant, seed, budget, real, split, synth, tracker):
            self.assertEqual((variant, budget), ('B', 100))
            self.assertIn(seed, (42, 43))
            resolved = core.SFTConfig(num_train_epochs=4, per_device_train_batch_size=4,
                                      gradient_accumulation_steps=4)
            self.assertEqual(resolved.per_device_train_batch_size, 2)
            self.assertEqual(resolved.gradient_accumulation_steps, 8)
            callback = core.CustomEvalCallback()
            for epoch in range(1, 5):
                callback.on_evaluate(resolved, SimpleNamespace(epoch=epoch), None)
            core.evaluate_classification()
            tracker.append(dict(variant=variant, seed=seed, percentage=budget,
                                test_results=metrics(), history=callback.epoch_logs))
            calls.append((variant, seed, budget))

        def fresh(*args):
            core.SFTConfig, core.CustomEvalCallback = Config, Callback
            core.evaluate_classification, core.run_single_experiment = metrics, run
            return core

        parent = Path(__file__).resolve().parents[1] / 'wyniki/testy'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            output = Path(folder)
            with patch.object(runner.importlib, 'import_module', side_effect=fresh):
                runner.train(frames, plans, config, output, dict(test=True), False)
                runner.train(frames, plans[:1], config, output, dict(test=True), False)
            self.assertEqual(len(calls), 2)
            saved = runner.read(output / 'results.json')
            self.assertEqual(len(saved), 2)
            self.assertAlmostEqual(saved[0]['test_results']['f0_5'], 5/7.25)
            self.assertEqual(len(saved[0]['history']), 4)


if __name__ == '__main__':
    unittest.main()
