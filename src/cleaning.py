import polars as pl
import os
from src.models import Center, Crew, Youth
from src.config import CenterConfig


def get_full_name_lookup(df: pl.DataFrame) -> dict[str, str]:
    """Create a lookup dictionary for last names (and last, first initial) to full names."""
    lookup = {}

    # Get all name combinations
    for row in df.iter_rows(named=True):
        full_name = row['full_name']
        last_name = row['last_name'].strip()
        first_name = row['full_name'].split()[0]

        # Add simple last name lookup
        lookup[last_name] = full_name

        # Add "Last, F" format lookup
        lookup[f'{last_name}, {first_name[0]}'] = full_name

    return lookup


def buddy_forms_get_youth_rows(df: pl.DataFrame) -> pl.DataFrame:
    buddies = (
        df.filter(pl.col('Grade').is_not_null())  # filters down to youth only
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

    # Clean spaces
    buddies_clean = buddies.with_columns(
        [
            pl.col(col).str.replace(r'\s+', ' ').str.to_titlecase().str.strip_chars().alias(col)
            for col in buddies.columns
            if buddies.schema[col] == pl.Utf8
        ]
    )

    # Create name lookup
    name_lookup = get_full_name_lookup(buddies_clean)

    # Convert friend choices to full names
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
    all_sibs = youth_df.filter(pl.col('par_sib').str.contains('S')).select(pl.col('full_name'), pl.col('last_name'))
    siblings_map = all_sibs.group_by('last_name').agg(pl.col('full_name').alias('siblings_all'))

    sibs_out = (
        all_sibs.join(siblings_map, on='last_name')
        .with_columns(siblings=pl.col('siblings_all').list.set_difference(pl.col('full_name').str.split('||')))
        .select(pl.col('full_name'), pl.col('siblings'))
        .with_columns(pl.concat_str(pl.col('siblings').list.join('|')))
        .with_columns(pl.col('siblings').fill_null('None'))
    )
    return sibs_out


def get_parent_names(youth_df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Get parent names for youth by matching last names with adult crew members.

    Args:
        youth_df: DataFrame with youth data including par_sib column
        year: Year of the crews data

    Returns:
        DataFrame with full_name and parent_name columns
    """
    crews_path = f'./data/clean/crews_{year}.csv'
    if not os.path.exists(crews_path):
        raise ValueError(f'Crews file {crews_path} does not exist')

    # Read crew data and filter for adults only
    crews_df = pl.read_csv(crews_path)
    # Not filtering for only adults because we want to include young adults because they lead crews too
    adults_df = crews_df.with_columns([pl.col('name').str.split(' ').list.last().alias('last_name')]).select(
        ['name', 'last_name']
    )
    # Get youth who have parents (par_sib contains 'P')
    youth_with_parents = youth_df.filter(pl.col('par_sib').str.contains('P')).select(['full_name', 'last_name'])

    # Group adults by last name to handle multiple adults with same last name
    adults_by_lastname = adults_df.group_by('last_name').agg(pl.col('name').alias('parent_names'))

    # Join youth with their potential parents based on last name
    parents_out = (
        youth_with_parents.join(adults_by_lastname, on='last_name', how='left')
        .with_columns(pl.col('parent_names').list.join('|').alias('parent_name'))
        .with_columns(pl.col('parent_name').fill_null('None'))
        .select(['full_name', 'parent_name'])
    )

    return parents_out


def clean_asp_buddies(raw_path: str, year: int) -> None:
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


def get_historical_youth_leaders(all_historical_crews: pl.DataFrame) -> dict[str, list[str]]:
    adult_df = all_historical_crews.filter(pl.col('is_adult')).drop('is_adult').rename({'name': 'adult_name'})
    youth_df = all_historical_crews.filter(~pl.col('is_adult')).drop('is_adult').rename({'name': 'youth_name'})
    youth_pairings_df = (
        youth_df.join(adult_df, on='crew_year', how='left')
        .group_by('youth_name')
        .agg(pl.col('adult_name').alias('adult_names'))
    )
    return {row['youth_name']: row['adult_names'] for row in youth_pairings_df.iter_rows(named=True)}


def _build_centers_from_df(adult_crews: pl.DataFrame) -> list[Center]:
    """Build centers from df without any configuration."""
    centers: list[Center] = []
    # Filter to only rows with non-null Crew assignments
    crew_assigned = adult_crews.filter(pl.col('Crew').is_not_null() & (pl.col('Crew') != ''))
    center_df = crew_assigned.group_by('Center').agg(pl.col('Crew'))
    for center_row in center_df.iter_rows(named=True):
        this_center = center_row['Center']
        crews: list[Crew] = []
        crew_df = crew_assigned.filter(pl.col('Center') == this_center).group_by('Crew').agg(pl.col('name'))
        for crew_row in crew_df.iter_rows(named=True):
            crew = Crew(name=crew_row['Crew'], adults=crew_row['name'])
            crews.append(crew)
        centers.append(Center(name=this_center, crews=crews))
    return centers


def _build_crews_from_df(adult_crews: pl.DataFrame, center_name: str) -> list[Crew]:
    """Build crews for a center by extracting from df."""
    crew_assigned = adult_crews.filter(
        (pl.col('Center') == center_name) & 
        pl.col('Crew').is_not_null() & 
        (pl.col('Crew') != '')
    )
    crews: list[Crew] = []
    if len(crew_assigned) > 0:
        crew_df = crew_assigned.group_by('Crew').agg(pl.col('name'))
        for crew_row in crew_df.iter_rows(named=True):
            crew = Crew(name=crew_row['Crew'], adults=crew_row['name'])
            crews.append(crew)
    return crews


def _build_crews_for_center(
    adult_crews: pl.DataFrame,
    center_name: str,
    crew_count: int,
) -> list[Crew]:
    """Build crews for a center with specified count.
    
    Creates crew_count crews (named C01, C02, etc. using center prefix).
    Populates adults from df where available.
    """
    prefix = center_name[0].upper()  # F for Fayette, K for Kanawha, etc.
    crews: list[Crew] = []
    
    for i in range(1, crew_count + 1):
        crew_name = f"{prefix}{i:02d}"
        # Get adults assigned to this crew from df (if any)
        crew_adults_df = adult_crews.filter(
            (pl.col('Center') == center_name) & 
            (pl.col('Crew') == crew_name)
        )
        crew_adults = crew_adults_df['name'].to_list() if len(crew_adults_df) > 0 else []
        crews.append(Crew(name=crew_name, adults=crew_adults))
    
    return crews


def get_centers_from_adults_df(
    adult_crews: pl.DataFrame,
    center_configs: list[CenterConfig] | None = None,
) -> tuple[list[Center], list[tuple[str, str]], list[str]]:
    """Extract centers from adults df with optional configuration.
    
    Args:
        adult_crews: DataFrame with adult crew assignments
        center_configs: Optional list of center configurations.
            - If None: extract all centers and crews from df
            - If provided: validate df centers are subset, use specified crew counts
              where provided, otherwise extract from df
    
    Returns:
        Tuple of (
            list of Center objects,
            list of center-only adults as (name, center),
            list of unassigned adults (no center or crew)
        )
    
    Raises:
        ValueError: If df contains centers not in center_configs
    """
    # Get center-only adults (those with Center but blank Crew column)
    center_only_adults = adult_crews.filter(
        (pl.col('Center').is_not_null() & (pl.col('Center') != '')) &
        (pl.col('Crew').is_null() | (pl.col('Crew') == ''))
    )
    center_only_list = [
        (row['name'], row['Center']) 
        for row in center_only_adults.iter_rows(named=True)
    ]
    
    # Get unassigned adults (those with no Center and no Crew)
    unassigned_adults = adult_crews.filter(
        (pl.col('Center').is_null() | (pl.col('Center') == '')) &
        (pl.col('Crew').is_null() | (pl.col('Crew') == ''))
    )
    unassigned_list = [
        row['name'] 
        for row in unassigned_adults.iter_rows(named=True)
    ]
    
    # Get unique centers from df (excluding null/empty)
    df_centers = set(
        c for c in adult_crews['Center'].unique().to_list() 
        if c is not None and c != ''
    )
    
    if center_configs is None:
        # Default behavior: extract everything from CSV
        return _build_centers_from_df(adult_crews), center_only_list, unassigned_list
    
    # Validate df centers are subset of configured centers
    config_names = {c.name for c in center_configs}
    invalid = df_centers - config_names
    if invalid:
        raise ValueError(
            f"Centers {invalid} in adults df not in configured list: {config_names}"
        )
    
    centers: list[Center] = []
    for cfg in center_configs:
        if cfg.crew_count is not None:
            # Use specified crew count, create empty crews if needed
            crews = _build_crews_for_center(adult_crews, cfg.name, cfg.crew_count)
        else:
            # Extract crew count from df
            crews = _build_crews_from_df(adult_crews, cfg.name)
        centers.append(Center(name=cfg.name, crews=crews))
    
    return centers, center_only_list, unassigned_list


def get_youth_from_buddy_form_df(youth: pl.DataFrame) -> list[Youth]:
    return [Youth(**row) for row in youth.iter_rows(named=True)]


def all_parents_are_valid(youth_df: pl.DataFrame, adult_df: pl.DataFrame) -> bool:
    adult_names = adult_df['name'].to_list()
    missing_parents = []
    for row in youth_df.iter_rows(named=True):
        if row['parent_name']:
            # Handle multiple parents separated by pipe
            parent_names = row['parent_name'].split('|') if row['parent_name'] else []
            for parent_name in parent_names:
                if parent_name and parent_name not in adult_names:
                    missing_parents.append(f"{row['name']}'s parent {parent_name}")
    if missing_parents:
        raise ValueError(f'Missing parents in adult crews: {", ".join(missing_parents)}')
    return True


def all_friends_are_valid(youth_list: list[Youth]) -> bool:
    """Check if all friend choices reference valid youth names."""
    valid_names = {youth.name for youth in youth_list}
    missing_friends = []

    for youth in youth_list:
        friend_choices = [youth.first_choice, youth.second_choice, youth.third_choice]
        for choice in friend_choices:
            if choice and choice not in valid_names:
                missing_friends.append(f"{youth.name}'s friend {choice}")

    if missing_friends:
        raise ValueError(f'Invalid friend choices found: {", ".join(missing_friends)}')
    return True


def clean_crews_2025_raw(raw_path: str, year: int) -> None:
    """Clean the 2025 raw crew data and format it to match the 2024 format."""
    if not os.path.exists(raw_path) or not raw_path.endswith('.csv'):
        raise ValueError(f'File {raw_path} does not exist or is not a csv')

    df = pl.read_csv(raw_path)

    center_mapping = {'F': 'Fayette', 'K': 'Kanawha', 'N': 'Nicholas', 'L': 'Leslie'}

    df_out = (
        df.with_columns(
            [
                # Combine first and last name
                pl.concat_str([pl.col('First Name'), pl.col('Last Name')], separator=' ').alias('name'),
                # Map crew first letter to center name
                pl.col('Crew').str.slice(0, 1).replace(center_mapping).alias('Center'),
                # Convert crew names to 2-digit format (F01, F02, etc.)
                pl.concat_str([pl.col('Crew').str.slice(0, 1), pl.col('Crew').str.slice(1).str.zfill(2)]).alias('Crew'),
                # Convert YA to Young Adult, keep Adult as is
                pl.when(pl.col('Adult/YA') == 'YA')
                .then(pl.lit('Young Adult'))
                .otherwise(pl.col('Adult/YA'))
                .alias('role'),
            ]
        )
        .select(['name', 'Center', 'Crew', 'role'])
        # Clean up name formatting - remove extra spaces and convert to title case
        .with_columns([pl.col('name').str.replace(r'\s+', ' ').str.to_titlecase().str.strip_chars()])
    )

    df_out.write_csv(f'./data/clean/crews_{year}.csv')


def convert_crews_to_historical(crews_path: str, year: int) -> None:
    """Convert crews data to historical_crews format and append to existing historical_crews.csv.

    Args:
        crews_path: Path to the crews CSV file (e.g., 'data/clean/crews_2024.csv')
        year: Year to append to crew names
    """
    if not os.path.exists(crews_path) or not crews_path.endswith('.csv'):
        raise ValueError(f'File {crews_path} does not exist or is not a csv')

    crews_df = pl.read_csv(crews_path)

    historical_df = crews_df.with_columns(
        [
            pl.concat_str([pl.col('Crew'), pl.lit(str(year))], separator=' ').alias('crew_year'),
            pl.when(pl.col('role') == 'Adult').then(pl.lit(True)).otherwise(pl.lit(False)).alias('is_adult'),
        ]
    ).select(['name', 'crew_year', 'is_adult'])

    historical_path = './data/clean/historical_crews.csv'
    if not os.path.exists(historical_path):
        # Create new file if it doesn't exist
        historical_df.write_csv(historical_path)
        return
    existing_df = pl.read_csv(historical_path)
    pl.concat([existing_df, historical_df]).unique().write_csv(historical_path)
