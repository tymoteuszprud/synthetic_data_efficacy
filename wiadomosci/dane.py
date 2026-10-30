"""Podział wiadomości i budowa zagnieżdżonych wycinków treningowych."""

import numpy as np

def split_real_dataset(df_real, df_extracted_indices, seed):

    extracted_df_indices = df_extracted_indices['df_index'].tolist()


    total_indices = set(range(len(df_real)))
    candidate_indices = list(total_indices - set(extracted_df_indices))

    n_total = len(df_real)
    n_val = int(round(0.15 * n_total))
    n_test = int(round(0.15 * n_total))


    rng = np.random.default_rng(seed)
    shuffled_candidates = candidate_indices.copy()
    rng.shuffle(shuffled_candidates)

    val_indices = shuffled_candidates[:n_val]
    test_indices = shuffled_candidates[n_val:n_val+n_test]


    candidate_train_pool = shuffled_candidates[n_val+n_test:]

    return {
        "extracted_indices": extracted_df_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "candidate_train_pool": candidate_train_pool
    }

def get_training_subset(df_real, split_info, percentage, seed):
    extracted_indices = split_info["extracted_indices"]
    candidate_train_pool = split_info["candidate_train_pool"]

    n_total = len(df_real)
    if percentage == 100:

        n_p = n_total - len(split_info["val_indices"]) - len(split_info["test_indices"])
    else:
        n_p = int(round(percentage / 100.0 * n_total))

    if percentage == 1:

        train_indices = extracted_indices
    else:

        n_extra = max(0, n_p - len(extracted_indices))

        n_extra = min(n_extra, len(candidate_train_pool))
        rng = np.random.default_rng(seed)
        # Ta sama permutacja dla wszystkich budżetów danego ziarna.
        extra_indices = list(rng.permutation(candidate_train_pool)[:n_extra])
        train_indices = extracted_indices + extra_indices

    return df_real.iloc[train_indices].reset_index(drop=True)


def get_synthetic_subset(synth, size, seed):
    """Wycinek będący początkiem stałej dla danego ziarna kolejności."""
    if size < 0 or size > len(synth):
        raise ValueError('Liczebność wycinka wykracza poza zbiór syntetyczny.')
    order = np.random.default_rng(seed).permutation(len(synth))
    return synth.iloc[order[:size]].reset_index(drop=True)


def training_data(real, synth, variant, seed):
    import pandas as pd
    if variant in ('A', 'B'):
        return real
    if variant in ('C', 'E'):
        subset = get_synthetic_subset(synth, len(real), seed)
        if variant == 'E':
            return subset
        return pd.concat([real, subset]).sample(frac=1, random_state=seed).reset_index(drop=True)
    if variant == 'D':
        return pd.concat([real, synth]).sample(frac=1, random_state=seed).reset_index(drop=True)
    raise ValueError(f'Nieznany wariant: {variant}')


def prepare_training_stages(real, synth, variant, seed, percentage, balanced=False):
    """Wspólna dla audytu i treningu kolejność próbek w każdym etapie."""
    from .balansowanie import balance_dataframe
    training = training_data(real, synth, variant, seed)
    sampling = None
    if balanced:
        stage = 'finetune' if variant == 'A' else 'standard'
        training, sampling = balance_dataframe(training, seed, variant, percentage, stage)
    pretrain, pretrain_sampling = (synth, None) if variant == 'A' else (None, None)
    if balanced and variant == 'A':
        pretrain, pretrain_sampling = balance_dataframe(synth, seed, variant, percentage, 'pretrain')
    return training, pretrain, sampling, pretrain_sampling


def pretrain_cache_path(root, fingerprint, seed, balanced):
    """Rozdzielenie modeli według konfiguracji, trybu i ziarna."""
    from pathlib import Path
    mode = 'balanced' if balanced else 'standard'
    return Path(root) / 'pretrain_cache' / f'{mode}_{fingerprint}_seed_{seed}'
