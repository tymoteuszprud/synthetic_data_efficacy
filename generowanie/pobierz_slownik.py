"""Pobieranie słownika wyrazów pomocniczych do generowania wiadomości."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error

URL = 'https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json'
ROOT = Path(__file__).resolve().parents[1]


def validate(data):
    dictionary = json.loads(data.decode('utf-8-sig'))
    if not isinstance(dictionary, dict) or not dictionary:
        raise ValueError('Słownik powinien być niepustym obiektem JSON.')
    if any(not isinstance(word, str) or not word.strip() for word in dictionary):
        raise ValueError('Klucze słownika powinny być niepustymi wyrazami.')
    return len(dictionary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'materialy/slownik_angielski.json')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    print(f'Źródło: {URL}')
    print(f'Plik docelowy: {args.output}')
    if not args.download:
        print('Pobieranie wymaga argumentu --download.')
        return
    if args.output.exists():
        print(f'Słownik jest dostępny lokalnie: {validate(args.output.read_bytes())} wyrazów.')
        return
    request = urllib.request.Request(URL, headers={'User-Agent': 'synthetic-data-research'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        parser.exit(1, f'Nie udało się pobrać słownika: {error}\nAdres źródła: {URL}\n')
    count = validate(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_bytes(data)
    temporary.replace(args.output)
    metadata = dict(source=URL, words=count, sha256=hashlib.sha256(data).hexdigest())
    args.output.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Zapisano słownik zawierający {count} wyrazów.')


if __name__ == '__main__':
    main()
