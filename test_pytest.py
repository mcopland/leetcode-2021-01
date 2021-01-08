# from helpers import tree_constructor

from helpers import linked_list_constructor
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


@pytest.mark.parametrize("nodes1, nodes2, nodes_sorted", [
    ([1, 2, 4], [1, 3, 4], [1, 1, 2, 3, 4, 4]),
    ([], [], []),
    ([], [0], [0]),
    ([1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]),
    ([5, 6, 7, 8], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8])])
def test_merge_two_lists(nodes1, nodes2, nodes_sorted):
    card = main.Jan04
    l1 = linked_list_constructor(nodes1)
    l2 = linked_list_constructor(nodes2)
    expected = linked_list_constructor(nodes_sorted)
    actual = card.merge_two_lists(card, l1, l2)
    while expected and actual:
        if expected.val != actual.val:
            assert False
        expected = expected.next
        actual = actual.next

    assert not expected and not actual


@pytest.mark.parametrize("head, expected", [
    ([1, 2, 3, 3, 4, 4, 5], [1, 2, 5]),
    ([1, 1, 1, 2, 3], [2, 3]),
    ([1, 1, 1, 1, 1], []),
    ([], []),
    ([], [])])
def test_delete_duplicates(head, expected):
    card = main.Jan05
    expected = linked_list_constructor(expected)
    actual_list = linked_list_constructor(head)
    actual = card.delete_duplicates(card, actual_list)
    while expected and actual:
        if expected.val != actual.val:
            assert False
        expected = expected.next
        actual = actual.next

    assert not expected and not actual


@pytest.mark.parametrize("arr, k, result", [
    ([2, 3, 4, 7, 11], 5, 9),
    ([1, 2, 3, 4], 2, 6),
    ([1, 2, 3, 4], 16, 20),
    ([1, 2, 3, 4], 100, 104),
    ([1, 5, 15, 100], 35, 38),
    ([1, 100], 16, 17),
    ([2], 1, 1),
    ([1, 2, 3, 4, 5, 15, 25], 14, 20)])
def test_find_kth_positive(arr, k, result):
    card = main.Jan06
    assert card.find_kth_positive(card, arr, k) == result


@pytest.mark.parametrize("s, result", [
    # The answer is "abc", with the length of 3.
    ("abcabcbb", 3),
    # The answer is "b", with the length of 1.
    ("bbbbb", 1),
    # The answer is "wke", with the length of 3. Notice that the answer must be
    # a substring, "pwke" is a subsequence and not a substring.
    ("pwwkew", 3),
    ("abababa", 2),
    ("aabbaabb", 2),
    ("abcdeefghij", 6),
    ("", 0)])
def test_length_of_longest_substring(s, result):
    card = main.Jan07
    assert card.length_of_longest_substring(card, s) == result
