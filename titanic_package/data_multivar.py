from __future__ import annotations
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_count_pairs(
    df: pd.DataFrame,
    x: str,
    hue: str,
    *,
    order: Optional[Sequence[str]] = None,
    hue_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    palette: str | Sequence = "Set2",
    rotate_xticks: int = 0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Count plot of category `x`, grouped by `hue`.

    Parameters
    ----------
    df : DataFrame
        Data containing categorical columns x and hue.
    x : str
        Column name to show on the x-axis (e.g. "Age Interval").
    hue : str
        Column to group by (e.g. "Pclass", "Embarked", "Fare Interval").
    order : list[str], optional
        Explicit order of categories along x.
    hue_order : list[str], optional
        Explicit order of hue categories.
    title : str, optional
        Plot title.
    palette : str or sequence, default "Set2"
        Seaborn palette name or list of colors.
    rotate_xticks : int, default 0
        Degrees to rotate x tick labels.
    ax : matplotlib Axes, optional
        Axes to draw on; if None, creates a new one.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Basic validation
    missing = [c for c in (x, hue) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    sns.countplot(
        data=df,
        x=x,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )

    ax.set_title(title or f'Count of "{x}" grouped by "{hue}"')
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")
    ax.legend(title=hue, loc="best")
    ax.figure.tight_layout()
    return ax

def add_family_type(all_df, train_df):
    """
    Create a 'Family Type' column based on 'Family Size' for both datasets.

    Parameters
    ----------
    all_df : pandas.DataFrame
        Combined dataset (e.g. train + test).
    train_df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Modified all_df and train_df with the new 'Family Type' column.
    """
    for dataset in [all_df, train_df]:
        # Start with direct copy
        dataset["Family Type"] = dataset["Family Size"]

        # Categorise family type based on size
        dataset.loc[dataset["Family Size"] == 1, "Family Type"] = "Single"
        dataset.loc[(dataset["Family Size"] > 1) & (dataset["Family Size"] < 5), "Family Type"] = "Small"
        dataset.loc[dataset["Family Size"] >= 5, "Family Type"] = "Large"

    return all_df, train_df

def unify_titles(all_df, train_df):
    """
    Standardise passenger titles across datasets by unifying variants
    (e.g. 'Mlle.' and 'Ms.' → 'Miss.', 'Mme.' → 'Mrs.', rare titles → 'Rare').

    Parameters
    ----------
    all_df : pandas.DataFrame
        Combined dataset (e.g. train + test).
    train_df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Modified all_df and train_df with a unified 'Titles' column.
    """
    for dataset in [all_df, train_df]:
        # Create a copy of the original Title column
        dataset["Titles"] = dataset["Title"]

        # Unify 'Miss' variants
        dataset["Titles"] = dataset["Titles"].replace(["Mlle.", "Ms."], "Miss.")

        # Unify 'Mrs' variants
        dataset["Titles"] = dataset["Titles"].replace("Mme.", "Mrs.")

        # Replace rare titles with 'Rare'
        dataset["Titles"] = dataset["Titles"].replace(
            [
                "Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.",
                "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona."
            ],
            "Rare"
        )

    return all_df, train_df