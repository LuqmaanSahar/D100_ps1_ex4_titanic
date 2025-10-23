import pandas as pd

def parse_names(row):
    """
    Parse Titanic-style names like:
      'Braund, Mr. Owen Harris'
      'Sandstrom, Mrs. (Margaret T.)'
    Returns a Series: [family_name, title, given_name, maiden_name]
    """
    try:
        text = row["Name"]
        if not isinstance(text, str):
            return pd.Series([None, None, None, None],
                             index=["family_name","title","given_name","maiden_name"])

        parts = [p.strip() for p in text.split(",", 1)]
        if len(parts) < 2:
            return pd.Series([None, None, None, None],
                             index=["family_name","title","given_name","maiden_name"])

        family_name = parts[0]
        next_text = parts[1]

        parts = [p.strip() for p in next_text.split(".", 1)]
        if len(parts) < 2:
            return pd.Series([family_name, None, None, None],
                             index=["family_name","title","given_name","maiden_name"])

        title = parts[0] + "."
        next_text = parts[1].strip()

        # Handle maiden name in parentheses if present
        maiden_name = None
        if "(" in next_text and ")" in next_text:
            before, _, after = next_text.partition("(")
            given_name = before.strip()
            maiden_name = after.rstrip(")").strip()
        else:
            given_name = next_text.strip()

        return pd.Series([family_name, title, given_name, maiden_name],
                         index=["family_name","title","given_name","maiden_name"])

    except Exception:
        # Keep silent, return all Nones on any unexpected format
        return pd.Series([None, None, None, None],
                         index=["family_name","title","given_name","maiden_name"])
    
def add_parsed_names(df):
    """
    Apply parse_names() across the DataFrame and add parsed name components
    as new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'Name' column (Titanic-style names).

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with four new columns:
        ['Family Name', 'Title', 'Given Name', 'Maiden Name'].
    """
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(parse_names, axis=1)
    return df

def summarise_survival_by_title_and_sex(df):
    """
    Summarise average survival rates grouped by passenger title and sex.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'Titles', 'Sex', and 'Survived' columns.

    Returns
    -------
    pandas.DataFrame
        A summary DataFrame with mean survival rates per (Title, Sex) group.
    """
    summary = (
        df[["Titles", "Sex", "Survived"]]
        .groupby(["Titles", "Sex"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(by="Survived", ascending=False)
    )
    return summary

