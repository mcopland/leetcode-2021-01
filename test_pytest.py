import main
import pytest


@pytest.mark.parametrize("s, true_false", [
    ("code", False),
    ("aab", True),
    ("carerac", True)
])
def test_can_permute_palindrome(s, true_false):
    card = main.Jan01
    assert card.can_permute_palindrome(card, s) is true_false


@pytest.mark.parametrize("arr, pieces, true_false", [
    ([85], [[85]], True),
    ([15, 88], [[88], [15]], True),
    ([49, 18, 16], [[16, 18, 49]], False),
    ([91, 4, 64, 78], [[78], [4, 64], [91]], True),
    ([1, 3, 5, 7], [[2, 4, 6, 8]], False)
])
def test_can_form_array(arr, pieces, true_false):
    card = main.Jan01
    assert card.can_form_array(card, arr, pieces) is true_false
