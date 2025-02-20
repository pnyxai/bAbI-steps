import itertools

import numpy as np


# ----------------------------------------------
# Location/AntiLocation Matrix Pre-Allocation
# ----------------------------------------------
def generate_location_matrix(num_locations):
    """
    Given a number of locations, thi function will return ALL possible combinations.
    Considerations:

    """
    # Known-locations
    zeros_except = np.zeros((num_locations, num_locations), dtype=np.bool)
    ones_except = np.ones((num_locations, num_locations), dtype=np.bool)
    # for each known location, set the value to 1
    for i in range(num_locations):
        zeros_except[i, i] = 1
        ones_except[i, i] = 0
    known_locations = np.concatenate((zeros_except, ones_except), axis=1)

    def generate_combinations_matrix(num_locations):
        # Generate all possible non-empty combinations
        all_combinations = list(
            itertools.chain.from_iterable(
                itertools.combinations(range(num_locations), r)
                for r in range(1, num_locations + 1)
            )
        )

        # Create binary matrix using NumPy indexing
        matrix = np.zeros((len(all_combinations), num_locations), dtype=np.bool)
        for i, comb in enumerate(all_combinations):
            matrix[i, list(comb)] = 1

        # Filter out rows with exactly one zero
        valid_rows = [row for row in matrix if np.count_nonzero(row == 0) != 1]

        return np.array(valid_rows)

    # Unknown location with anti-locations
    anti_locations = generate_combinations_matrix(num_locations)
    zeros_anti_locations = np.zeros(
        (anti_locations.shape[0], num_locations), dtype=np.bool
    )
    anti_locations = np.concatenate((zeros_anti_locations, anti_locations), axis=1)
    location_matrix = np.concatenate((known_locations, anti_locations), axis=0)
    return location_matrix


# ----------------------------------------------
# Parent OR Combinations generation
# ----------------------------------------------


def generate_parent_combinations(x):
    # Thank you Gemini 2.0 Flash Thinking !
    n = len(x)
    combinations = []

    def generate_combinations_recursive(index, current_x1, current_x2):
        if index == n:
            if any(current_x1) and any(current_x2):  # Check for all-zero parents
                combinations.append((current_x1[:], current_x2[:]))
            return

        if x[index] == 0:
            current_x1[index] = 0
            current_x2[index] = 0
            generate_combinations_recursive(index + 1, current_x1, current_x2)
        else:  # x[index] == 1
            current_x1[index] = 0
            current_x2[index] = 1
            generate_combinations_recursive(index + 1, current_x1, current_x2)

            current_x1[index] = 1
            current_x2[index] = 0
            generate_combinations_recursive(index + 1, current_x1, current_x2)

            current_x1[index] = 1
            current_x2[index] = 1
            generate_combinations_recursive(index + 1, current_x1, current_x2)

    generate_combinations_recursive(0, [0] * n, [0] * n)
    return combinations


def is_valid_parent(parent, n):
    first_half = np.array(parent[:n])
    second_half = np.array(parent[n:])
    solution_to_compare = np.array([1] * n)
    # if first half is all zeros
    if not any(first_half):
        # then check that they are not n-1 ones (forbidden case)
        return np.sum(second_half) != n - 1
    # If we are here, means that the first half has only one 1
    assert sum(first_half) == 1
    # then now check that the sum of first half and second half is equal to [1, 1, 1, 1]
    sum_halfs = first_half + second_half
    return np.array_equal(sum_halfs, solution_to_compare)


def is_valid_parent_antilocations(parent, n):
    # functions to remove solutions where for the second half there is a n-1 zeros
    second_half = np.array(parent[n:])
    return np.sum(second_half) != n - 1


def filter_unique_parent_combinations(combinations, n, validate_func):
    unique_combinations = []
    seen_combinations = set()

    for x1, x2 in combinations:
        if validate_func(x1, n) and validate_func(x2, n):
            # Ensure consistent ordering using lexicographical comparison
            if tuple(x1) <= tuple(x2):  # Compare as tuples for lexicographical order
                pair = (tuple(x1), tuple(x2))  # Use tuple for hashability
            else:
                pair = (tuple(x2), tuple(x1))

            if pair not in seen_combinations:
                unique_combinations.append(
                    (list(x1), list(x2))
                )  # Back to list for output
                seen_combinations.add(pair)

    return unique_combinations


def generate_OR_parents(x: np.array):
    n = len(x) // 2
    first_half, second_half = np.array(x[:n]), np.array(x[n:])

    # do we have all zeros in the first half?
    if not any(first_half):
        # Case where x is THE NOWHERE.
        if all(second_half):
            return ([0] * n + [1] * n, [0] * n + [1] * n)

        assert np.sum(second_half) != n - 1, (
            "the second half correspond to N-1 ones. Error!"
        )
        solutions = filter_unique_parent_combinations(
            generate_parent_combinations(x), n, is_valid_parent
        )
        nowhere_parent = ([0] * n + [1] * n, x.copy().astype(int).tolist())
        solutions.append(nowhere_parent)
        return solutions

    # assert thar in the first half there is only one 1.
    assert sum(first_half) == 1, "Expected exactly one 1 in the first half."
    # assert that sum of first half and second half is equal to [1, 1, 1, 1]

    sum_halfs = first_half + second_half
    assert np.array_equal(sum_halfs, np.array([1] * n)), (
        "The result of first half + second half is NOT equal to all ones."
    )
    solutions = filter_unique_parent_combinations(
        generate_parent_combinations(x), n, is_valid_parent
    )
    # Create special case where child is combination of
    # the full nowhere parent and the child == x.
    # Add the nowhere_parent special case
    nowhere_parent = ([0] * n + [1] * n, x.copy().astype(int).tolist())
    solutions.append(nowhere_parent)
    # Edge case: switch the unique 1 in first_half to 0 and generate additional parents.
    index_of_one = int(np.where(first_half == 1)[0][0])
    x_copy = x.copy()
    x_copy[index_of_one] = 0
    additional = filter_unique_parent_combinations(
        generate_parent_combinations(x_copy), n, is_valid_parent_antilocations
    )

    return solutions + additional
