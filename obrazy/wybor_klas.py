"""Pokazuje wybór klas i opcjonalnie usuwa pozostałe foldery po potwierdzeniu."""
import argparse
from pathlib import Path
import shutil
from .dane import DEFAULT_CLASSES, load_classes


def folders_to_remove(root, selected):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError('Wymagany katalog zbioru, nie korzeń dysku.')
    missing = [name for name in selected if not (root/name).is_dir()]
    if missing:
        raise ValueError(f'Brak wybranych klas: {missing}')
    others = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in selected]
    for path in others:
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)() or path.resolve().parent != root:
            raise ValueError(f'Odmowa usunięcia dowiązania lub folderu poza zbiorem: {path}')
    return root, others


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--class-file', type=Path, default=DEFAULT_CLASSES)
    parser.add_argument('--remove-others', action='store_true')
    args = parser.parse_args()
    selected = load_classes(args.class_file)
    root, others = folders_to_remove(args.root, selected)
    print(f'Zbiór: {root}\nWybrane klasy ({len(selected)}): {", ".join(selected)}')
    print('Foldery do usunięcia:')
    for path in others:
        print(path)
    if not args.remove_others or not others:
        print('Nie usunięto żadnych folderów.')
        return
    expected = f'USUN {len(others)}'
    if input(f'Usunięcie jest trwałe. Potwierdzenie: {expected}\n> ').strip() != expected:
        print('Anulowano.')
        return
    # Ponowna kontrola pełnej listy po potwierdzeniu.
    _, current = folders_to_remove(root, selected)
    if current != others:
        raise RuntimeError('Lista folderów zmieniła się; operacja przerwana.')
    for path in current:
        if path.resolve().parent != root or path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise RuntimeError('Zmieniła się ścieżka folderu; operacja przerwana.')
        shutil.rmtree(path)
    print(f'Usunięto {len(current)} folderów.')


if __name__ == '__main__':
    main()
