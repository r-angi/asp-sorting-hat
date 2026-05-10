"""Year-specific raw-CSV cleaners.

These run **once** to convert vendor- or year-specific input formats into the
canonical ``data/clean/buddies_{year}.csv`` and ``data/clean/crews_{year}.csv``
files that production code consumes. They are intentionally not imported by
``main.py``.
"""

import os

import polars as pl


def get_full_name_lookup(df: pl.DataFrame) -> dict[str, str]:
    """Map last names (and ``Last, F`` initials) to full names from a buddy df."""
    lookup: dict[str, str] = {}

    for row in df.iter_rows(named=True):
        full_name = row['full_name']
        last_name = row['last_name'].strip()
        first_name = row['full_name'].split()[0]

        lookup[last_name] = full_name
        lookup[f'{last_name}, {first_name[0]}'] = full_name

    return lookup


def buddy_forms_get_youth_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Extract & normalize youth rows from a raw buddy-form export."""
    buddies = (
        df.filter(pl.col('Grade').is_not_null())
        .with_columns(
            full_name=pl.concat_str([pl.col('Name'), pl.col('Last')], separator=' '),
            history=pl.col('New/Vet').str.replace('\\*', ''),
        )
        .select(
            pl.col('full_name'),
            pl.col('Last').alias('last_name'),
            pl.col('Par/Sib').alias('par_sib'),
            pl.col('history'),
            pl.col('Gender').alias('gender'),
            pl.col('Grade').alias('year'),
            pl.col('1').alias('first_choice'),
            pl.col('2').alias('second_choice'),
            pl.col('3').alias('third_choice'),
        )
    )

    buddies_clean = buddies.with_columns(
        [
            pl.col(col).str.replace(r'\s+', ' ').str.to_titlecase().str.strip_chars().alias(col)
            for col in buddies.columns
            if buddies.schema[col] == pl.Utf8
        ]
    )

    name_lookup = get_full_name_lookup(buddies_clean)

    buddies_clean = buddies_clean.with_columns(
        [
            pl.col('first_choice')
            .map_elements(lambda x: name_lookup.get(x, None), return_dtype=pl.Utf8)
            .alias('first_choice'),
            pl.col('second_choice')
            .map_elements(lambda x: name_lookup.get(x, None), return_dtype=pl.Utf8)
            .alias('second_choice'),
            pl.col('third_choice')
            .map_elements(lambda x: name_lookup.get(x, None), return_dtype=pl.Utf8)
            .alias('third_choice'),
        ]
    )

    return buddies_clean


def get_siblings(youth_df: pl.DataFrame) -> pl.DataFrame:
    """Compute the per-youth sibling list by grouping rows where ``par_sib`` contains ``S``."""
    all_sibs = youth_df.filter(pl.col('par_sib').str.contains('S')).select(
        pl.col('full_name'), pl.col('last_name')
    )
    siblings_map = all_sibs.group_by('last_name').agg(pl.col('full_name').alias('siblings_all'))

    sibs_out = (
        all_sibs.join(siblings_map, on='last_name')
        .with_columns(
            siblings=pl.col('siblings_all').list.set_difference(pl.col('full_name').str.split('||'))
        )
        .select(pl.col('full_name'), pl.col('siblings'))
        .with_columns(pl.concat_str(pl.col('siblings').list.join('|')))
        .with_columns(pl.col('siblings').fill_null('None'))
    )
    return sibs_out


def get_parent_names(youth_df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Look up parent full names for youth (``par_sib`` contains ``P``) by last-name match."""
    crews_path = f'./data/clean/crews_{year}.csv'
    if not os.path.exists(crews_path):
        raise ValueError(f'Crews file {crews_path} does not exist')

    crews_df = pl.read_csv(crews_path)
    adults_df = crews_df.with_columns(
        [pl.col('name').str.split(' ').list.last().alias('last_name')]
    ).select(['name', 'last_name'])
    youth_with_parents = youth_df.filter(pl.col('par_sib').str.contains('P')).select(
        ['full_name', 'last_name']
    )

    adults_by_lastname = adults_df.group_by('last_name').agg(pl.col('name').alias('parent_names'))

    parents_out = (
        youth_with_parents.join(adults_by_lastname, on='last_name', how='left')
        .with_columns(pl.col('parent_names').list.join('|').alias('parent_name'))
        .with_columns(pl.col('parent_name').fill_null('None'))
        .select(['full_name', 'parent_name'])
    )

    return parents_out


def clean_asp_buddies(raw_path: str, year: int) -> None:
    """Convert raw buddy-form CSV → ``data/clean/buddies_{year}.csv``."""
    if not os.path.exists(raw_path) or not raw_path.endswith('.csv'):
        raise ValueError(f'File {raw_path} does not exist or is not a csv')

    raw_buddy_df = pl.read_csv(raw_path)
    youth_df = buddy_forms_get_youth_rows(raw_buddy_df)
    siblings_df = get_siblings(youth_df)
    parents_df = get_parent_names(youth_df, year)

    youth_df_out = (
        youth_df.join(siblings_df, on='full_name', how='left')
        .join(parents_df, on='full_name', how='left')
        .with_columns(pl.col('parent_name').fill_null(''))
        .rename({'full_name': 'name'})
        .drop('last_name', 'par_sib')
    )
    youth_df_out.write_csv(f'./data/clean/buddies_{year}.csv')


def clean_historical_crews(raw_path: str, year: int) -> None:
    """Convert vendor historical-crews CSV → ``data/clean/crews_{year}.csv``."""
    if not os.path.exists(raw_path) or not raw_path.endswith('.csv'):
        raise ValueError(f'File {raw_path} does not exist or is not a csv')
    df = pl.read_csv(raw_path)
    df_out = df.select(
        pl.col("Participant's Name").alias('name'),
        pl.col('Center'),
        pl.col('Crew'),
        pl.col('I am registering for this ASP trip as:').alias('role'),
    ).with_columns(pl.col('name').str.replace(r'\s+', ' ').str.to_titlecase().str.strip_chars())
    df_out.write_csv(f'./data/clean/crews_{year}.csv')


def clean_historical_crews_old(historical_crew_path: str, year: int) -> None:
    """Legacy historical-crews CSV cleaner (older vendor format)."""
    historical_crews_df = pl.read_csv(historical_crew_path)
    cleaned_historical_df = (
        historical_crews_df.rename(
            {"Participant's Name - Last Name": 'last_name', "Participant's Name - First Name": 'first_name'}
        )
        .with_columns(
            name=pl.concat_str([pl.col('first_name'), pl.col('last_name')], separator=' '),
            crew_year=pl.concat_str([pl.col('Crew'), pl.lit(year)], separator=' '),
        )
        .with_columns((pl.col('name') == pl.col('name').str.to_uppercase()).fill_null(False).alias('is_adult'))
        .with_columns(pl.col('name').str.replace(r'\s+', ' ').str.to_titlecase().str.strip_chars().alias('name'))
        .select(['name', 'crew_year', 'is_adult'])
    )
    cleaned_historical_df.write_csv(f'./data/clean/historical_crews_{year}.csv')


def clean_crews_2025_raw(raw_path: str, year: int) -> None:
    """Clean the 2025 raw crew data and format it to match the 2024 format."""
    if not os.path.exists(raw_path) or not raw_path.endswith('.csv'):
        raise ValueError(f'File {raw_path} does not exist or is not a csv')

    df = pl.read_csv(raw_path)

    center_mapping = {'F': 'Fayette', 'K': 'Kanawha', 'N': 'Nicholas', 'L': 'Leslie'}

    df_out = (
        df.with_columns(
            [
                pl.concat_str([pl.col('First Name'), pl.col('Last Name')], separator=' ').alias('name'),
                pl.col('Crew').str.slice(0, 1).replace(center_mapping).alias('Center'),
                pl.concat_str(
                    [pl.col('Crew').str.slice(0, 1), pl.col('Crew').str.slice(1).str.zfill(2)]
                ).alias('Crew'),
                pl.when(pl.col('Adult/YA') == 'YA')
                .then(pl.lit('Young Adult'))
                .otherwise(pl.col('Adult/YA'))
                .alias('role'),
            ]
        )
        .select(['name', 'Center', 'Crew', 'role'])
        .with_columns([pl.col('name').str.replace(r'\s+', ' ').str.to_titlecase().str.strip_chars()])
    )

    df_out.write_csv(f'./data/clean/crews_{year}.csv')
