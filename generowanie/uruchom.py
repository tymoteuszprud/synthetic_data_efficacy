"""Uruchamia generator obrazów albo SMS w katalogu materiałów wejściowych."""
import argparse
import os
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['sms', 'obrazy'])
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Docelowy plik wiadomości lub katalog obrazów.')
    parser.add_argument('--manifest', type=Path, help='Wspólny manifest obrazów; wymagany dla generatora obrazów.')
    parser.add_argument('--classes', nargs='+', help='Wybrane nazwy klas manifestu obrazów.')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        parser.error('Katalog wejściowy nie istnieje.')
    app = Path(__file__).with_name(args.kind + '_gui.py')
    if args.kind == 'obrazy':
        if not args.manifest or not args.manifest.is_file():
            parser.error('Generator obrazów wymaga --manifest z istniejącym manifestem.')
        os.environ['IMAGE_MANIFEST'] = str(args.manifest.resolve())
        if args.classes:
            import json
            os.environ['IMAGE_CLASSES'] = json.dumps(args.classes)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        variable = 'SMS_OUTPUT_FILE' if args.kind == 'sms' else 'SYNTHETIC_IMAGES_DIR'
        os.environ[variable] = str(output)
    os.chdir(workspace)
    runpy.run_path(str(app), run_name='__main__')


if __name__ == '__main__':
    main()
