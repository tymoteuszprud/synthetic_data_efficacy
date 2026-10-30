"""Dodatkowe metryki zapisanych klasyfikatorów obrazowych; bez treningu."""

from obrazy.dane import DEFAULT_CLASSES, selected_classes
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('wyniki/metryki_obrazy'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    selected = selected_classes(args.manifest, args.class_file)
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata = sorted((root / 'experiments_archive').glob('*/metadata.json'))
    metadata = [p for p in metadata if 'SmallBatch' not in p.parent.name]
    if args.run_id:
        metadata = [p for p in metadata if p.parent.name == args.run_id]
    if not metadata:
        raise ValueError('Brak modeli spełniających kryteria.')
    audit = []
    test_file_hashes = {}
    for path in metadata:
        m = json.loads(path.read_text(encoding='utf-8'))
        paths, labels, _ = m['last_test_paths']
        classes = m['classes']
        assert len(classes) == len(set(classes)) and set(classes) == set(selected), 'Klasy modelu nie odpowiadają wyborowi klas.'
        assert len(paths) == len(set(paths)) == len(labels) == len(m['test_preds']) == len(m['test_trues'])
        assert [Path(p).parent.name for p in paths] == labels
        assert all((root / p).is_file() for p in paths)
        for image_path in paths:
            if image_path not in test_file_hashes:
                test_file_hashes[image_path] = digest(root / image_path)
        assert (path.parent / 'best_model.pth').is_file()
        assert m['dataset_for_stats'] == 'ImageNet'
        saved_labels = [classes[int(x)] for x in m['test_trues']]
        assert Counter(saved_labels) == Counter(labels)
        audit.append({'run_id': m['id'], 'n_test': len(paths),
                      'saved_order_matches_paths': saved_labels == labels,
                      'metadata_sha256': digest(path)})
    write_json(out / 'audyt_plikow.json', audit)
    if not args.execute:
        print(f'Audit OK: {len(audit)} models; use --execute for inference.')
        return

    import cv2
    import numpy as np
    import torch
    import torchvision
    from torchvision import models
    from torchvision.transforms import v2
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, precision_recall_fscore_support)
    import sklearn
    versions = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'numpy': np.__version__, 'sklearn': sklearn.__version__, 'opencv': cv2.__version__}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transform = v2.Compose([v2.CenterCrop((224, 224)),
                           v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    cache, rows, per_class, controls = {}, [], [], []
    started = time.monotonic()
    for run_index, path in enumerate(metadata, 1):
        m = json.loads(path.read_text(encoding='utf-8'))
        name, classes = m['id'], m['classes']
        indices = {c: i for i, c in enumerate(classes)}
        paths, labels, _ = m['last_test_paths']
        truth = np.array([indices[label] for label in labels], dtype=np.int64)
        model_path = path.parent / 'best_model.pth'
        identity = {'metadata': digest(path), 'weights': digest(model_path),
                    'script': digest(Path(__file__)), 'versions': versions,
                    'test_images': {p: test_file_hashes[p] for p in paths},
                    'preprocessing': 'cv2 RGB; resize 256x256 INTER_LINEAR; /255; CenterCrop224; ImageNet normalization',
                    'device': str(device), 'batch_size': int(m['params']['batch_size'])}
        marker = out / (name + '.json')
        scores_path = out / (name + '.npz')
        previous = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else None
        if previous and previous.get('identity') == identity and scores_path.exists():
            with np.load(scores_path, allow_pickle=False) as data:
                assert np.array_equal(data['truth'], truth)
                assert data['paths'].tolist() == paths
                probabilities = data['probabilities']
            print(f'[{run_index}/{len(metadata)}] reused {name}', flush=True)
        else:

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Sequential(torch.nn.Dropout(m['params']['dropout']),
                                            torch.nn.Linear(model.fc.in_features, len(classes)))
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model = model.to(device).eval()
            batches = []
            batch_size = int(m['params']['batch_size'])
            with torch.inference_mode():
                for start in range(0, len(paths), batch_size):
                    batch = []
                    for raw_path in paths[start:start + batch_size]:
                        if raw_path not in cache:
                            image = cv2.imread(str(root / raw_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                            if image is None:
                                raise ValueError(f'Unreadable test image: {raw_path}')
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                            cache[raw_path] = torch.from_numpy(image).permute(2, 0, 1)
                        batch.append(cache[raw_path])
                    images = torch.stack(batch).to(device, dtype=torch.float32) / 255.0
                    logits = model(transform(images))
                    batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            probabilities = np.concatenate(batches)
            np.savez_compressed(scores_path, probabilities=probabilities, truth=truth,
                                paths=np.asarray(paths), classes=np.asarray(classes))
            del model, logits, images
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f'[{run_index}/{len(metadata)}] evaluated {name}', flush=True)
        pred = probabilities.argmax(axis=1)
        assert probabilities.shape == (len(truth), len(classes))
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        confusion = confusion_matrix(truth, pred, labels=range(len(classes)))
        saved_confusion = confusion_matrix(np.asarray(m['test_trues'], dtype=np.int64),
                                           np.asarray(m['test_preds'], dtype=np.int64), labels=range(len(classes)))
        control = {'run_id': name, 'same_confusion_as_saved': bool(np.array_equal(confusion, saved_confusion)),
                   'new_accuracy': float(accuracy_score(truth, pred)), 'saved_accuracy': m['test_acc']}
        controls.append(control)

        if not control['same_confusion_as_saved']:
            print(f'CONTROL: confusion matrix differs from saved predictions: {name}', flush=True)
        p, r, f1, support = precision_recall_fscore_support(truth, pred, labels=range(len(classes)), zero_division=0)
        ap = []
        for i, cls in enumerate(classes):
            assert support[i] > 0, (name, cls)
            value = float(average_precision_score((truth == i).astype(int), probabilities[:, i]))
            ap.append(value)
            per_class.append({'run_id': name, 'class': cls, 'support': int(support[i]),
                              'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f1[i]), 'AP': value})
        row = {'run_id': name, 'variant': name[0],
               'budget': int(re.search(r'(\d+)pct', name).group(1)),
               'run': int(re.search(r'Run(\d+)$', name).group(1)), 'n_test': len(truth),
               'accuracy': float(accuracy_score(truth, pred)), 'precision_macro': float(p.mean()),
               'recall_macro': float(r.mean()), 'f1_macro': float(f1.mean()), 'mAP': float(np.mean(ap))}
        rows.append(row)
        write_json(marker, {'identity': identity, 'metrics': row, 'control': control})
        write_csv(out / 'metryki_uruchomienia.csv', rows)
        write_csv(out / 'metryki_klasy.csv', per_class)
        write_json(out / 'kontrola_predykcji.json', controls)
    summary = []
    for variant in 'ABCDE':
        for budget in (1, 3, 10, 30, 100):
            group = [r for r in rows if r['variant'] == variant and r['budget'] == budget]
            if not group:
                continue
            row = {'variant': variant, 'budget': budget, 'runs': len(group)}
            for metric in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'mAP'):
                values = [100*r[metric] for r in group]
                row[metric + '_mean'] = statistics.mean(values)
                row[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(row)
    write_csv(out / 'metryki_srednie.csv', summary)
    write_json(out / 'status.json', {'runs': len(rows), 'elapsed_seconds': time.monotonic()-started,
                                   'training_executed': False, 'versions': versions,
                                   'same_confusion_count': sum(r['same_confusion_as_saved'] for r in controls)})
    print(f'Done: {len(rows)} runs, training never invoked.', flush=True)


if __name__ == '__main__':
    main()
