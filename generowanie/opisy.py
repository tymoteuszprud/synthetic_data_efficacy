"""Opisuje obrazy referencyjne albo rozwija opisy przez lokalne API modelu."""
import argparse
import ast
import base64
from collections import Counter
import json
import math
from pathlib import Path
import random
import time
from obrazy.dane import inventory, read_manifest

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CAPTION = 'You are a precise image captioner. Describe the image in detail, with all necessary information to recreate it from the text description. Use clear descriptive sentences, not just tags. Keep it factual and concise. Image class is in the text prompt.'


def request(url, model, messages, temperature, max_tokens):
    import requests
    response = requests.post(url, json=dict(model=model, messages=messages,
                                            temperature=temperature, max_tokens=max_tokens,
                                            stream=False), timeout=200)
    response.raise_for_status()
    content = response.json()['choices'][0]['message']['content'].strip()
    if not content:
        raise ValueError('Model zwrócił pustą odpowiedź.')
    return content


def load_lines(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        key, value = line.split(':', 1)
        result[key.replace('\\', '/')] = value.strip()
    return result


def save_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(f'{key}: {value}\n' for key, value in sorted(records.items())), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['opisz', 'rozwin'])
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Opisy referencyjne dla trybu rozwin.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:1234/v1/chat/completions')
    parser.add_argument('--model', default='auto')
    parser.add_argument('--fraction', type=float, default=.25, help='Docelowy udział opisów względem liczby obrazów klasy.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--manifest', type=Path, required=True, help='Manifest przygotowany przez obrazy.dane lub przygotuj_kolejke.')
    parser.add_argument('--classes', nargs='+', help='Jawny podzbiór nazw klas manifestu.')
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error('Udział musi należeć do (0, 1].')
    images = args.images.resolve()
    manifest = read_manifest(args.manifest)
    classes = args.classes or manifest['classes']
    if set(classes) - set(manifest['classes']):
        parser.error('Klasy muszą należeć do manifestu.')
    listed = inventory(images, classes)
    rows = [(name, images / p) for name, paths in listed.items() for p in paths]
    if not rows:
        parser.error('Brak obrazów w folderach klas.')
    counts = Counter(label for label, _ in rows)
    print(dict(counts))
    if not args.generate:
        print('Kontrola zakończona; wywołania API wymagają --generate.')
        return
    output = args.output.resolve()
    records = load_lines(output) if output.exists() else {}
    rng = random.Random(args.seed)
    if args.mode == 'opisz':
        for label in sorted(counts):
            selected = [images / path for path in manifest['references'][label]]
            for path in selected:
                key = f'{label}/{path.name}'
                if key in records:
                    continue
                mime = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.tif': 'tiff'}.get(path.suffix.lower(), path.suffix[1:].lower())
                image_bytes = path.read_bytes()
                if mime not in ('jpeg', 'png', 'webp'):
                    import io
                    from PIL import Image
                    stream = io.BytesIO()
                    with Image.open(path) as image:
                        image.convert('RGB').save(stream, format='PNG')
                    image_bytes, mime = stream.getvalue(), 'png'
                encoded = base64.b64encode(image_bytes).decode('ascii')
                messages = [dict(role='system', content=SYSTEM_CAPTION),
                            dict(role='user', content=[dict(type='text', text=label),
                                 dict(type='image_url', image_url=dict(url=f'data:image/{mime};base64,{encoded}'))])]
                records[key] = request(args.url, args.model, messages, .1, 1024).replace('\n', ' ')
                save_lines(output, records)
                print(key, flush=True)
    else:
        if not args.input:
            parser.error('Tryb rozwin wymaga --input z opisami referencyjnymi.')
        references = load_lines(args.input)
        template = (ROOT / 'instrukcje/opis_sceny.txt').read_text(encoding='utf-8')
        for label, count in sorted(counts.items()):
            pool = [value for key, value in references.items() if key.split('/')[0] == label]
            if not pool:
                raise ValueError(f'Brak opisów referencyjnych: {label}')
            for index in range(math.ceil(count * args.fraction)):
                key = f'{label}/opis_{index:05d}.txt'
                reference = rng.choice(pool)
                if key in records:
                    continue
                messages = [dict(role='system', content=template),
                            dict(role='user', content=f'Sport: {label}. Raw description: {reference}')]
                records[key] = request(args.url, args.model, messages, .7, 175).replace('\n', ' ').replace('"', '')
                save_lines(output, records)
                print(key, flush=True)
    output.with_suffix('.config.json').write_text(json.dumps(dict(mode=args.mode, model=args.model,
        url=args.url, seed=args.seed, manifest=str(args.manifest.resolve()), classes=classes, fraction=args.fraction), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
