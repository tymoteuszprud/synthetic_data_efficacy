"""Uruchamia interfejs obrazowy w osobnym katalogu danych i wyników."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    work = args.workspace.resolve()
    work.mkdir(parents=True, exist_ok=True)
    app = Path(__file__).with_name('trening.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(app),
                    '--server.port', str(args.port)], cwd=work, check=True)


if __name__ == '__main__':
    main()
