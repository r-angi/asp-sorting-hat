from ortools.sat.python import cp_model
from src.models import Center, Youth
import polars as pl


def calculate_friend_scores(
    solver: cp_model.CpSolver,
    person_center: dict[tuple[str, str], int],
    youth_list: list[Youth],
    centers: list[Center],
) -> tuple[dict[str, float], float]:
    center_scores = {center.name: 0.0 for center in centers}
    center_people_count = {center.name: 0 for center in centers}
    youth_dict = {youth.name: youth for youth in youth_list}
    # Calculate raw scores and count people in each center
    for youth in youth_list:
        for center in centers:
            if solver.Value(person_center[youth.name, center.name]) == 1:
                center_people_count[center.name] += 1

                # Calculate friendship scores
                friend_weights = {
                    youth.first_choice: 3,
                    youth.second_choice: 2,
                    youth.third_choice: 1,
                }
                for friend_name, weight in friend_weights.items():
                    if friend_name and friend_name in youth_dict:
                        if solver.Value(person_center[friend_name, center.name]) == 1:
                            center_scores[center.name] += weight
    # Normalize scores by number of people
    normalized_scores = {
        center.name: round(center_scores[center.name] / center_people_count[center.name], 2)
        if center_people_count[center.name] > 0
        else 0.0
        for center in centers
    }
    avg_score = round(sum(center_scores.values()) / len(youth_list), 2)

    return normalized_scores, avg_score


def calculate_historical_friend_scores(centers: list[Center], year: int) -> tuple[dict[str, float], float]:
    youth_df = pl.read_csv(f'./data/clean/crews_{year}.csv').filter(pl.col('role') != 'Adult')
    buddies_df = pl.read_csv(f'./data/clean/buddies_{year}.csv')
    youth_buddies_df = youth_df.join(buddies_df, on='name', how='left')
    center_scores = {center.name: 0.0 for center in centers}
    center_people_count = {center.name: 0 for center in centers}
    overall_score = 0
    youth_list = youth_buddies_df['name'].to_list()

    for youth in youth_buddies_df.iter_rows(named=True):
        for center in centers:
            if youth['Center'] == center.name:
                center_people_count[center.name] += 1

                # Calculate friendship scores
                friend_weights = {
                    youth['first_choice']: 3,
                    youth['second_choice']: 2,
                    youth['third_choice']: 1,
                }
                for friend_name, weight in friend_weights.items():
                    # Skip if friend_name is None or empty string
                    if not friend_name:
                        continue

                    if friend_name in youth_list:
                        friend_center = youth_buddies_df.filter(pl.col('name') == friend_name)
                        # Check if friend exists and has a center assignment
                        if len(friend_center) > 0 and friend_center[0, 'Center'] == center.name:
                            center_scores[center.name] += weight
                            overall_score += weight

    normalized_scores = {
        center.name: round(center_scores[center.name] / center_people_count[center.name], 2)
        if center_people_count[center.name] > 0
        else 0.0
        for center in centers
    }
    avg_score = round(overall_score / len(youth_list), 2)
    return normalized_scores, avg_score


def calculate_friend_choice_stats(solver, person_center, youth_list, centers):
    """Calculate statistics about friend choice fulfillment."""
    youth_dict = {youth.name: youth for youth in youth_list}
    stats = {
        'first_choice': 0,
        'second_choice': 0,
        'third_choice': 0,
        'multiple_friends': 0,
        'total_youth': len(youth_list),
    }

    for youth in youth_list:
        friends_with = 0
        choices = [
            (youth.first_choice, 'first_choice'),
            (youth.second_choice, 'second_choice'),
            (youth.third_choice, 'third_choice'),
        ]

        for friend_name, choice_type in choices:
            if friend_name and friend_name in youth_dict:
                # Check if they're in the same center
                for center in centers:
                    if (
                        solver.Value(person_center[youth.name, center.name]) == 1
                        and solver.Value(person_center[friend_name, center.name]) == 1
                    ):
                        stats[choice_type] += 1
                        friends_with += 1
                        break

        if friends_with > 1:
            stats['multiple_friends'] += 1

    # Convert to percentages
    total = stats['total_youth']
    return {
        'first_choice_pct': round(stats['first_choice'] / total * 100, 1),
        'second_choice_pct': round(stats['second_choice'] / total * 100, 1),
        'third_choice_pct': round(stats['third_choice'] / total * 100, 1),
        'multiple_friends_pct': round(stats['multiple_friends'] / total * 100, 1),
    }


def print_crew_assignments(solver, person_crew, person_center, youth_list, centers):
    youth_dict = {youth.name: youth for youth in youth_list}
    friend_scores, _ = calculate_friend_scores(solver, person_center, youth_list, centers)

    # Track center-level statistics
    center_stats = {}

    for center in centers:
        print(f'\nCenter {center.name}:')
        center_youth = []
        center_stats[center.name] = {
            'total_youth': 0,
            'total_adults': 0,
            'years': {'Fr': 0, 'So': 0, 'Jr': 0, 'Sr': 0},
            'gender': {'M': 0, 'F': 0},
            'history': {'V': 0, 'N': 0},
            'friend_score': friend_scores[center.name],
        }

        for crew in center.crews:
            crew_youth = [
                youth.name for youth in youth_list if solver.Value(person_crew[youth.name, center.name, crew.name]) == 1
            ]
            center_youth.extend(crew_youth)

            # Calculate crew diversity metrics
            years = [youth_dict[person].year for person in crew_youth]
            year_counts = {}
            for year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
                center_stats[center.name]['years'][year] += 1

            genders = [youth_dict[person].gender for person in crew_youth]
            gender_counts = {}
            for gender in genders:
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
                center_stats[center.name]['gender'][gender] += 1

            histories = [youth_dict[person].history for person in crew_youth]
            history_counts = {}
            for history in histories:
                history_counts[history] = history_counts.get(history, 0) + 1
                center_stats[center.name]['history'][history] += 1

            print(f'  {crew.name}:')
            print(f'    Youth: {crew_youth}')
            print(f'    Adults: {crew.adults}')
            print(f'    Years: {year_counts}')
            print(f'    Gender (M/F): {gender_counts}')
            print(f'    History (vet/new): {history_counts}')

            center_stats[center.name]['total_youth'] += len(crew_youth)
            center_stats[center.name]['total_adults'] += len(crew.adults)

    # Print summary statistics
    print('\n=== Summary Statistics ===')
    total_youth = sum(stats['total_youth'] for stats in center_stats.values())
    total_adults = sum(stats['total_adults'] for stats in center_stats.values())

    print('\nOverall Totals:')
    print(f'Total Youth: {total_youth}')
    print(f'Total Adults: {total_adults}')
    print(f'Total Participants: {total_youth + total_adults}')

    # Add friend choice statistics
    friend_stats = calculate_friend_choice_stats(solver, person_center, youth_list, centers)
    print('\nFriend Choice Statistics:')
    print(f'Youth with first choice friend: {friend_stats["first_choice_pct"]}%')
    print(f'Youth with second choice friend: {friend_stats["second_choice_pct"]}%')
    print(f'Youth with third choice friend: {friend_stats["third_choice_pct"]}%')
    print(f'Youth with multiple friends: {friend_stats["multiple_friends_pct"]}%')

    print('\nCenter-by-Center Statistics:')
    for center_name, stats in center_stats.items():
        print(f'\n{center_name}:')
        print(f'  Total Participants: {stats["total_youth"] + stats["total_adults"]}')
        print(f'  Youth: {stats["total_youth"]}, Adults: {stats["total_adults"]}')
        print(f'  Friend Score: {stats["friend_score"]:.2f}')
        print(f'  Years: {stats["years"]}')
        print(f'  Gender: {stats["gender"]}')
        print(f'  History: {stats["history"]}')


def status_to_string(status):
    if status == cp_model.OPTIMAL:
        return 'OPTIMAL'
    elif status == cp_model.FEASIBLE:
        return 'FEASIBLE'
    elif status == cp_model.INFEASIBLE:
        return 'INFEASIBLE'
    elif status == cp_model.MODEL_INVALID:
        return 'MODEL_INVALID'
    else:
        return 'UNKNOWN'
