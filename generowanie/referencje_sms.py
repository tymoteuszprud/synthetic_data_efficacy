"""Wybiera zbalansowane referencje SMS bez zmiany oryginalnego CSV."""
import argparse
from pathlib import Path


def main():
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    labels = frame['label'].str.lower()
    size = round(len(frame) * .01)
    n_spam = int(np.ceil(.5 * size))
    rng = np.random.RandomState(args.seed)
    spam = rng.choice(frame.index[labels == 'spam'], n_spam, replace=False)
    ham = rng.choice(frame.index[labels == 'ham'], size - n_spam, replace=False)
    indices = np.sort(np.concatenate([spam, ham]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / 'wiadomosci_referencyjne.csv',
               args.output_dir / 'indeksy_referencji.csv']
    if any(path.exists() for path in outputs):
        raise FileExistsError('Pliki referencji już istnieją; wymagany jest pusty katalog wyjściowy.')
    selected = frame.iloc[indices]
    selected.to_csv(outputs[0], index=False)
    pd.DataFrame(dict(df_index=indices, line_number=indices+2,
                      label=selected['label'].values)).to_csv(outputs[1], index=False)
    print(f'Zapisano {len(indices)} referencji: {selected.label.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
