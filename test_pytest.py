# from helpers import binary_tree_constructor

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
#     original = binary_tree_constructor(nodes)
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


@pytest.mark.parametrize("word1, word2, true_false", [
    # "ab" + "c" == "a" + "bc"
    (["ab", "c"], ["a", "bc"], True),
    (["a", "cb"], ["ab", "c"], False),
    (["abc", "d", "defg"], ["abcddefg"], True),
    (["a"], ["a"], True),
    (["a", "bc", "d"], ["a", "cb", "d"], False),
    (["abcde"], ["a", "b", "c", "d", "e"], True)])
def test_array_strings_are_equal(word1, word2, true_false):
    card = main.Jan08
    assert card.array_strings_are_equal(card, word1, word2) == true_false


@pytest.mark.parametrize("beginWord, endWord, wordList, result", [
    # "hit" -> "hot" -> "dot" -> "dog" -> "cog"
    ("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"], 5),
    # The endWord "cog" is not in wordList.
    ("hit", "cog", ["hot", "dot", "dog", "lot", "log"], 0)
    ])
def test_ladder_length(beginWord, endWord, wordList, result):
    card = main.Jan09
    assert card.ladder_length(card, beginWord, endWord, wordList) == result


@pytest.mark.parametrize("instructions, result", [
    ([1, 5, 6, 2], 1),
    ([1, 2, 3, 6, 5, 4], 3),
    ([1, 3, 3, 3, 2, 4, 2, 1, 2], 4),
    ([1, 3, 5, 7, 9, 2, 4, 6, 8], 7),
    ([9, 8, 7, 6, 5, 4, 3, 2, 1], 0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9], 0)
    ])
def test_create_sorted_array(instructions, result):
    card = main.Jan10
    assert card.create_sorted_array(card, instructions) == result


@pytest.mark.parametrize("nums1, m, nums2, n", [
    ([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3),
    ([1], 1, [], 0),
    ([1, 3, 5, 7, 9, 0, 0, 0, 0], 5, [2, 4, 6, 8], 4),
    ([5, 6, 7, 8, 9, 0, 0, 0, 0], 5, [1, 2, 3, 4], 4),
    ([1, 2, 3, 4, 0, 0], 4, [5, 6], 2),
    ([5, 6, 0, 0, 0, 0], 2, [1, 2, 3, 4], 4)
    ])
def test_merge(nums1, m, nums2, n):
    card = main.Jan11
    result = sorted(nums1[:m] + nums2)
    card.merge(card, nums1, m, nums2, n)
    assert nums1 == result


@pytest.mark.parametrize("nodes1, nodes2, result", [
    ([2, 4, 3], [5, 6, 4], [7, 0, 8]),
    ([0], [0], [0]),
    ([9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9], [8, 9, 9, 9, 0, 0, 0, 1]),
    ([1, 1, 2], [3, 4, 5], [4, 5, 7]),
    ([9], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]),
    ([7, 4, 5], [3, 5, 4, 9], [0, 0, 0, 0, 1])
    ])
def test_add_two_numbers(nodes1, nodes2, result):
    card = main.Jan12
    l1 = linked_list_constructor(nodes1)
    l2 = linked_list_constructor(nodes2)
    expected = linked_list_constructor(result)
    actual = card.add_two_numbers(card, l1, l2)
    while expected and actual:
        if expected.val != actual.val:
            assert False
        expected = expected.next
        actual = actual.next

    assert not expected and not actual


@pytest.mark.parametrize("people, limit, result", [
    ([1, 2], 3, 1),
    ([3, 2, 2, 1], 3, 3),
    ([3, 5, 3, 4], 5, 4),
    ([1, 5, 1, 5, 2, 4, 2, 4, 3, 3, 3, 3], 6, 6),
    ([1, 5, 1, 5, 2, 4, 2, 4, 3, 3, 3, 3], 5, 8),
    ])
def test_num_rescue_boats(people, limit, result):
    card = main.Jan13
    assert card.num_rescue_boats(card, people, limit) == result


@pytest.mark.parametrize("nums, x, result", [
    ([1, 1, 4, 2, 3], 5, 2),
    ([5, 6, 7, 8, 9], 4, -1),
    ([3, 2, 20, 1, 1, 3], 10, 5)
    ])
def test_min_operations(nums, x, result):
    card = main.Jan14
    assert card.min_operations(card, nums, x) == result


@pytest.mark.parametrize("nested_list, result", [
    # Four 1's at depth 2, one 2 at depth 1.
    # 1*1 + 1*1 + 2*2 + 1*1 + 1*1 = 10.
    ([[1, 1], 2, [1, 1]], 10),
    # One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3.
    # 1*1 + 4*2 + 6*3 = 27.
    ([1, [4, [6]]], 27),
    ([0], 0)
    ])
def test_depth_sum(nested_list, result):
    card = main.Jan15
    assert card.depth_sum(card, nested_list) == result


@pytest.mark.parametrize("n, result", [
    (7, 3),
    (2, 1),
    (3, 2),
    (0, 0),
    (100, 21)
    ])
def test_get_maximum_generated(n, result):
    card = main.Jan15
    assert card.get_maximum_generated(card, n) == result


@pytest.mark.parametrize("nums, k, result", [
    ([3, 2, 1, 5, 6, 4], 2, 5),
    ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4),
    ([1], 1, 1),
    ([1, 2, 3, 4, 5, 6], 1, 6),
    ([2, 2, 2, 2, 2, 5, 2, 2, 2], 1, 5)
    ])
def test_find_kth_largest(nums, k, result):
    card = main.Jan16
    assert card.find_kth_largest(card, nums, k) == result


@pytest.mark.parametrize("n, result", [
    # The 5 sorted strings that consist of vowels only are:
    # ["a", "e", "i", "o", "u"].
    (1, 5),
    # The 15 sorted strings that consist of vowels only are:
    # ["aa", "ae", "ai", "ao", "au",
    #  "ee", "ei", "eo", "eu", "ii",
    #  "io", "iu", "oo", "ou", "uu"].
    # Note that "ea" is not a valid string since 'e' comes after 'a' in the
    # alphabet.
    (2, 15),
    (25, 23751),
    (33, 66045),
    (50, 316251)
    ])
def test_count_vowel_strings(n, result):
    card = main.Jan17
    assert card.count_vowel_strings(card, n) == result


@pytest.mark.parametrize("nums, k, result", [
    ([1, 2, 3, 4], 5, 2),
    ([3, 1, 3, 4, 3], 6, 1),
    ([1], 1, 0),
    ([1, 1], 2, 1),
    ([1, 2, 3, 7, 7, 7, 4, 5, 6], 7, 3),
    ([3, 1, 5, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2], 1, 0)
    ])
def test_max_operations(nums, k, result):
    card = main.Jan18
    assert card.max_operations(card, nums, k) == result


@pytest.mark.parametrize("s, result", [
    ("babad", "bab"),
    ("cbbd", "bb"),
    ("a", "a"),
    ("ac", "a"),
    ("abc1c23", "c1c")
    ])
def test_longest_palindrome(s, result):
    card = main.Jan19
    assert card.longest_palindrome(card, s) == result
