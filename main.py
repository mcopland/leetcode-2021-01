from models import TreeNode
from typing import List


# canPermutePalindrome and canFormArray
class Jan01:
    def can_permute_palindrome(self, s: str) -> bool:
        """Palindrome Permutation

        Given a string, determine if a permutation of the string could form a
        palindrome.

        Hints:
        - Consider the palindromes of odd vs even length. What difference do
          you notice?
        - Count the frequency of each character.
        - If each character occurs even number of times, then it must be a
          palindrome.
        - How about character which occurs odd number of times?
        """
        count = set()

        # Put char in set if it occurs an odd number of times
        for c in s:
            if c in count:
                count.remove(c)
            else:
                count.add(c)

        # True for 0/1 char occurring an odd number of times
        return len(count) < 2

    def can_form_array(self, arr: List[int], pieces: List[List[int]]) -> bool:
        """Check Array Formation Through Concatenation

        You are given an array of distinct integers arr and an array of integer
        arrays pieces, where the integers in pieces are distinct. Your goal is
        to form arr by concatenating the arrays in pieces in any order.
        However, you are not allowed to reorder the integers in each array
        pieces[i].

        Return true if it is possible to form the array arr from pieces.
        Otherwise, return false.

        Constraints:
        - 1 <= pieces.length <= arr.length <= 100
        - sum(pieces[i].length) == arr.length
        - 1 <= pieces[i].length <= arr.length
        - 1 <= arr[i], pieces[i][j] <= 100
        - The integers in arr are distinct.
        - The integers in pieces are distinct (i.e., If we flatten pieces in a
          1D array, all the integers in this array are distinct).

        Hints:
        - Note that the distinct part means that every position in the array
        belongs to only one piece.
        - Note that you can get the piece every position belongs to naively.
        """
        # Key = head int of piece : val = entire piece
        dic = {p[0]: p for p in pieces}
        res = []

        # Add to result if found
        for i in arr:
            res += dic.get(i, [])

        return res == arr


# getTargetCopy
class Jan02:
    def get_target_copy(self, original: TreeNode, cloned: TreeNode,
                        target: TreeNode) -> TreeNode:
        """Find a Corresponding Node of a Binary Tree in a Clone of That Tree

        Given two binary trees original and cloned and given a reference to a
        node target in the original tree.

        The cloned tree is a copy of the original tree.

        Return a reference to the same node in the cloned tree.

        Note that you are not allowed to change any of the two trees or the
        target node and the answer must be a reference to a node in the cloned
        tree.

        Follow up: Solve the problem if repeated values on the tree are
        allowed.

        Constraints:
        - The number of nodes in the tree is in the range [1, 10^4].
        - The values of the nodes of the tree are unique.
        - target node is a node from the original tree and is not null.
        """
        if not original:
            return None

        if original is target:
            return cloned

        left = self.get_target_copy(self, original.left, cloned.left, target)
        if left:
            return left

        right = self.get_target_copy(self, original.right, cloned.right, target)
        if right:
            return right

        return None
