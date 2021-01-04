from helpers import tree_constructor

import main
import pytest


@pytest.mark.parametrize("s, true_false", [
    ("code", False),
    ("aab", True),
    ("carerac", True)])
def test_can_permute_palindrome(s, true_false):
    card = main.Jan01
    assert card.can_permute_palindrome(card, s) == true_false


@pytest.mark.parametrize("arr, pieces, true_false", [
    ([85], [[85]], True),
    ([15, 88], [[88], [15]], True),
    ([49, 18, 16], [[16, 18, 49]], False),
    ([91, 4, 64, 78], [[78], [4, 64], [91]], True),
    ([1, 3, 5, 7], [[2, 4, 6, 8]], False)])
def test_can_form_array(arr, pieces, true_false):
    card = main.Jan01
    assert card.can_form_array(card, arr, pieces) == true_false


# @pytest.mark.parametrize("nodes, target", [
#     ([7, 4, 3, None, None, 6, 19], 3),
#     ([7], 7),
#     ([8, None, 6, None, 5, None, 4, None, 3, None, 2, None, 1], 4),
#     ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5),
#     ([1, 2, None, 3], 2)])
# def test_get_target_copy(nodes, target):
#     card = main.Jan02
#     original = tree_constructor(nodes)
#     cloned = original
#     case_1 = card.get_target_copy(card, original, cloned, target)
#     case_2 = card.get_target_copy(card, cloned, original, target)
#     assert case_1 is case_2


@pytest.mark.parametrize("n, result", [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 8),
    (5, 10),
    (6, 36),
    (7, 41),
    (8, 132),
    (9, 250),
    (10, 700),
    (11, 750),
    (12, 4010),
    (13, 4237),
    (14, 10680),
    (15, 24679)])
def test_count_arrangement(n, result):
    card = main.Jan03
    assert card.count_arrangement(card, n) == result
