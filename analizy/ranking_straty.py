"""Porządkuje syntetyczne obrazy według straty klasyfikatora ResNet-18."""
from obrazy.dane import DEFAULT_CLASSES, selected_classes
import argparse
import csv
import json
from pathlib import Path

EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('wyniki/ranking_straty'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    selected = selected_classes(args.manifest, args.class_file)
    if args.batch_size < 1:
        parser.error('Rozmiar partii musi być dodatni.')
    metadata = json.loads((args.model_dir / 'metadata.json').read_text(encoding='utf-8'))
    weights = args.model_dir / 'best_model.pth'
    if not weights.is_file():
        parser.error('Brak best_model.pth.')
    classes = metadata['classes']
    rows = [(folder.name, path) for folder in (args.synthetic/name for name in selected)
            for path in sorted(folder.iterdir()) if path.suffix.lower() in EXTENSIONS and path.is_file()]
    if not rows or any(label not in classes for label, _ in rows):
        parser.error('Brak obrazów lub etykiety niezgodne z modelem.')
    print(f'Obrazy do oceny: {len(rows)}')
    if not args.execute:
        print('Kontrola zakończona. Ocena wymaga --execute.')
        return
    import cv2
    import numpy as np
    import torch
    from torchvision import models
    from torchvision.transforms import v2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(metadata['params']['dropout']),
                                  torch.nn.Linear(model.fc.in_features, len(classes)))
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.to(device).eval()
    transform = v2.Compose([v2.CenterCrop((224, 224)),
                           v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    results, errors = [], []
    for start in range(0, len(rows), args.batch_size):
        images, valid = [], []
        for label, path in rows[start:start + args.batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                errors.append(dict(path=str(path.resolve()), error='Nieodczytywalny obraz'))
                continue
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_LINEAR)
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            valid.append((label, path))
        if not images:
            continue
        with torch.inference_mode():
            logits = model(transform(torch.stack(images).to(device, dtype=torch.float32) / 255.0))
            targets = torch.tensor([classes.index(label) for label, _ in valid], device=device)
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none').cpu().tolist()
            probabilities = logits.softmax(dim=1).cpu().numpy()
        for i, (label, path) in enumerate(valid):
            results.append(dict(path=str(path.resolve()), label=label,
                                predicted=classes[int(np.argmax(probabilities[i]))],
                                label_probability=float(probabilities[i, classes.index(label)]), loss=losses[i]))
    results.sort(key=lambda row: (-row['loss'], row['path']))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'ranking.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['rank', 'path', 'label', 'predicted', 'label_probability', 'loss'])
        writer.writeheader()
        writer.writerows(dict(rank=i, **row) for i, row in enumerate(results, 1))
    (args.output / 'nieodczytane.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Zapisano ranking {len(results)} obrazów. Decyzje o jakości wymagają ręcznego przeglądu.')


if __name__ == '__main__':
    main()
