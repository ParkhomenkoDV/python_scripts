from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def psi(old_data, new_data, bins=10) -> float:
    if not np.issubdtype(old_data.dtype, np.number):
        old_fractions = old_data.value_counts(normalize=True)
        new_fractions = new_data.value_counts(normalize=True)
    else:
        _, bins = pd.qcut(old_data, q=bins, duplicates='drop', retbins=True)
        if len(bins) == 1:
            bins = np.append(bins, bins + 1e-10)
        else:
            bins[-1] += 1e-10
        bins = np.hstack([-np.inf, bins, np.inf])
        old_fractions = pd.cut(old_data, bins=bins, include_lowest=True, right=False).value_counts(
            normalize=True).sort_index()
        new_fractions = pd.cut(new_data, bins=bins, include_lowest=True, right=False).value_counts(
            normalize=True).sort_index()

    old_fractions, new_fractions = old_fractions.align(new_fractions, fill_value=0.0)
    old_fractions, new_fractions = old_fractions.values, new_fractions.values
    old_fractions, new_fractions = old_fractions + 0.000_01, new_fractions + 0.000_01
    old_fractions, new_fractions = old_fractions / old_fractions.sum(), new_fractions / old_fractions.sum()

    return np.sum((old_fractions - new_fractions) * np.log(old_fractions / new_fractions))


def population_stability_index(date_column, old, new, features=None):
    assert all(column in old.columns for column in new.columns)
    assert date_column in old.columns

    if features is None:
        features = old.columns
    elif isinstance(features, (list, tuple, set)):
        assert all(column in old.columns for column in features)
    else:
        raise

    df_psi = pd.DataFrame({'features': features})  # насильный перевод всего в pandas

    for name, df in zip(('train', 'test'), (old, new)):
        for date in tqdm(sorted(df[date_column].unique()), desc=name):
            df_temp = df[df[date_column] == date]
            df_psi[f'{date} {name}'] = [psi(old[feature], df_temp[feature]) for feature in features]

    return df_psi[sorted(df_psi.columns)].set_index('features')


if __name__ == '__main__':
    df = pd.read_csv('datas/processed/ag.csv', sep=';').drop(['epk_id'], axis=1)
    df.head()

    features = [c for c in df.columns if 'date' not in c and 'epk_id' not in c]
    df_psi = population_stability_index('date_month',
                                        old=df[df['date_month'] <= '2023-01'],
                                        new=df[~(df['date_month'] <= '2023-01')],
                                        features=features)

    from matplotlib.colors import LinearSegmentedColormap

    cm_psi = LinearSegmentedColormap.from_list('', ['#00c853', 'khaki', '#f44336'], N=10, gamma=0.6)

    fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(features)))
    plt.title(f"PSI")
    plt.tight_layout()

    sns.heatmap(
        df_psi,
        cmap=cm_psi,
        annot=True,
        linewidths=0.75,
        vmin=0,
        vmax=0.3,
        cbar=False,
        fmt='.2f',
        annot_kws={'fontsize': 12})

    df_psi.round(3).style.set_precision(2).background_gradient(cmap=cm_psi, axis=None, vmin=0, vmax=0.3)
    df_psi.round(3).style.set_precision(2).background_gradient(cmap=cm_psi, axis=None, vmin=0, vmax=0.3).to_excel(
        'psi.xlsx')
