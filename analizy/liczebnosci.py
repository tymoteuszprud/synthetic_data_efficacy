"""Liczy próbki w klasach obrazowych albo w kolumnie label wiadomości."""
from obrazy.dane import DEFAULT_CLASSES, selected_classes
import argparse
from collections import Counter
import csv
from pathlib import Path

EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--sms', type=Path)
    parser.add_argument('--output', type=Path, default=Path('wyniki/liczebnosci.csv'))
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if bool(args.images) == bool(args.sms):
        parser.error('Wskaż dokładnie jeden zbiór: --images albo --sms.')
    if args.images:
        selected = selected_classes(args.manifest, args.class_file)
        counts = {folder.name: sum(p.is_file() and p.suffix.lower() in EXTENSIONS for p in folder.iterdir())
                  for folder in (args.images/name for name in selected)}
    else:
        with args.sms.open(encoding='utf-8-sig', newline='') as file:
            counts = dict(Counter(row['label'].strip().upper() for row in csv.DictReader(file)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['klasa', 'liczba_probek'])
        writer.writerows(sorted(counts.items()))
    print(f'Łącznie {sum(counts.values())} próbek w {len(counts)} klasach.')


if __name__ == '__main__':
    main()
