"""Wyznacza cechy ImageNet ResNet-18 do analizy podobieństwa, bez dostrajania."""
from obrazy.dane import DEFAULT_CLASSES, selected_classes
import argparse
from pathlib import Path

EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('wyniki/embeddingi'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    selected = selected_classes(args.manifest, args.class_file)
    rows = [(source, folder.name, path) for source, root in
            (('rzeczywiste', args.real), ('syntetyczne', args.synthetic))
            for folder in (root/name for name in selected)
            for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in EXTENSIONS]
    if not rows:
        parser.error('Brak obrazów.')
    print(f'Znaleziono {len(rows)} plików.')
    if not args.execute:
        print('Model nie został załadowany. Obliczenia wymagają --execute.')
        return
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import resnet18, ResNet18_Weights
    if args.batch_size < 1:
        parser.error('Rozmiar partii musi być dodatni.')
    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    class Images(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            try:
                with Image.open(rows[index][2]) as image:
                    return transform(image.convert('RGB')), index, True
            except (OSError, ValueError):
                return transform(Image.new('RGB', (224, 224))), index, False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    vectors = np.empty((len(rows), 512), dtype=np.float32)
    valid = np.zeros(len(rows), dtype=bool)
    with torch.inference_mode():
        for images, indices, readable in DataLoader(Images(), batch_size=args.batch_size, shuffle=False, num_workers=0):
            features = torch.nn.functional.normalize(model(images.to(device)), dim=1)
            vectors[indices.numpy()] = features.cpu().numpy()
            valid[indices.numpy()] = readable.numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    bad = [str(rows[i][2]) for i in np.flatnonzero(~valid)]
    selected = [row for row, keep in zip(rows, valid) if keep]
    np.savez_compressed(args.output / 'embeddingi_resnet18.npz', vectors=vectors[valid],
                        sets=np.array([r[0] for r in selected]), classes=np.array([r[1] for r in selected]),
                        paths=np.array([str(r[2]) for r in selected]))
    (args.output / 'nieodczytane_obrazy.txt').write_text('\n'.join(bad), encoding='utf-8')
    print(f'Zapisano cechy {len(selected)} obrazów; pominięto {len(bad)} plików.')


if __name__ == '__main__':
    main()
