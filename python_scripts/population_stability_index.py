from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def psi(train, test, bins=10) -> float:
    """Подсчет конкретного значения"""
    if not np.issubdtype(train.dtype, np.number):
        train_fractions = train.value_counts(normalize=True)
        test_fractions = test.value_counts(normalize=True)
    else:
        _, bins = pd.qcut(train, q=bins, duplicates='drop', retbins=True)
        if len(bins) == 1:
            bins = np.append(bins, bins + 1e-10)
        else:
            bins[-1] += 1e-10
        bins = np.hstack([-np.inf, bins, np.inf])
        train_fractions = pd.cut(train, bins=bins, include_lowest=True, right=False).value_counts(
            normalize=True).sort_index()
        test_fractions = pd.cut(test, bins=bins, include_lowest=True, right=False).value_counts(
            normalize=True).sort_index()

    train_fractions, test_fractions = train_fractions.align(test_fractions, fill_value=0.0)
    train_fractions, test_fractions = train_fractions.values, test_fractions.values
    train_fractions, test_fractions = train_fractions + 0.000_01, test_fractions + 0.000_01
    train_fractions, test_fractions = train_fractions / train_fractions.sum(), test_fractions / train_fractions.sum()

    return np.sum((train_fractions - test_fractions) * np.log(train_fractions / test_fractions))


def population_stability_index(date_column, train, test, features=None):
    """Подсчет всех значений в dataframe"""
    assert all(column in train.columns for column in test.columns)
    assert date_column in train.columns

    if features is None:
        features = train.columns
    elif isinstance(features, (list, tuple, set)):
        assert all(column in train.columns for column in features)
    else:
        raise

    df_psi = pd.DataFrame({'features': features})  # насильный перевод всего в pandas

    for name, df in zip(('train', 'test'), (train, test)):
        for date in tqdm(sorted(df[date_column].unique()), desc=name):
            df_temp = df[df[date_column] == date]
            df_psi[f'{date} {name}'] = [psi(train[feature], df_temp[feature]) for feature in features]

    return df_psi[sorted(df_psi.columns)].set_index('features')


if __name__ == '__main__':
    df = pd.read_csv('datas/processed/ag.csv', sep=';').drop(['epk_id'], axis=1)
    df.head()

    features = [c for c in df.columns if 'date' not in c and 'epk_id' not in c]
    df_psi = population_stability_index('date_month',
                                        train=df[(df['date_month'] <= '2023-01')],
                                        test=df[~(df['date_month'] <= '2023-01')],
                                        features=features)

    fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(features)))
    plt.title(f"PSI")
    plt.tight_layout()

    sns.heatmap(df_psi,
                cmap='RdYlGn',
                annot=True,
                linewidths=0.75,
                vmin=0,
                vmax=0.3,
                cbar=False,
                fmt='.2f',
                annot_kws={'fontsize': 12})

    df_psi.style.set_precision(2).background_gradient(cmap='RdYlGn', axis=None, vmin=0, vmax=0.2)
    df_psi.style.set_precision(2).background_gradient(cmap='RdYlGn', axis=None, vmin=0, vmax=0.2).to_excel('psi.xlsx')
