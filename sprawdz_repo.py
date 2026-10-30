"""Kontrola składni i plików publikowanych; bez importowania modułów treningowych."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'dane', 'wyniki', 'results',
            'experiments_archive', 'unsloth_compiled_cache', '.cache'}


def main():
    checked = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in EXCLUDED for part in path.relative_to(ROOT).parts):
            continue
        if path.name.startswith('slownik_angielski.'):
            continue
        if path.suffix in ('.py', '.json', '.md', '.tex', '.txt', '.csv'):
            text = path.read_text(encoding='utf-8-sig')
            # Kontrola obejmuje również dokumentację i konfiguracje publikacji.
            if re.search(r'[A-Za-z]:[\\/](?:Users|Stable)[\\/]', text):
                raise ValueError(f'Osobista ścieżka: {path.relative_to(ROOT)}')
            if re.search(r'\b(?:sk-[A-Za-z0-9]{24,}|hf_[A-Za-z0-9]{24,})\b', text):
                raise ValueError(f'Potencjalny token dostępu: {path.relative_to(ROOT)}')
            for legacy in ('folder' + '_filtered', 'Synthetic' + '_Sports_Supported'):
                if legacy in text:
                    raise ValueError(f'Nazwa robocza: {path.relative_to(ROOT)}')
            if re.search(r'\[(?:adres|identyfikator wersji) do uzupełnienia\]', text):
                raise ValueError(f'Nieuzupełnione pole: {path.relative_to(ROOT)}')
        if path.suffix == '.py':
            text = path.read_text(encoding='utf-8')
            ast.parse(text, filename=str(path))
            checked += 1
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    for name in ('README.md', '.gitignore', 'konfiguracje/tekst.json', 'konfiguracje/obrazy.json',
                 'materialy/slownictwo_podstawowe.txt'):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    print(f'Kontrola zakończona: {checked} plików Python, poprawna składnia JSON i wymagane materiały.')


if __name__ == '__main__':
    main()
