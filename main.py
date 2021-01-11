from models import ListNode, Node, TreeNode
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

        You are given an array of distinct integers `arr` and an array of
        integer arrays `pieces`, where the integers in `pieces` are distinct.
        Your goal is to form `arr` by concatenating the arrays in `pieces` in
        any order. However, you are not allowed to reorder the integers in each
        array `pieces[i]`.

        Return true if it is possible to form the array `arr` from `pieces`.
        Otherwise, return false.

        Constraints:
        - 1 <= `pieces.length` <= `arr.length` <= 100
        - `sum(pieces[i].length)` == `arr.length`
        - 1 <= `pieces[i].length` <= `arr.length`
        - 1 <= `arr[i]`, `pieces[i][j]` <= 100
        - The integers in `arr` are distinct.
        - The integers in `pieces` are distinct (i.e., If we flatten `pieces`
          in a 1D array, all the integers in this array are distinct).

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

        Given two binary trees, `original` and `cloned`, and given a reference
        to a node `target` in the `original` tree. The `cloned` tree is a copy
        of the `original` tree. Return a reference to the same node in the
        `cloned` tree.

        Note that you are not allowed to change any of the two trees or the
        `target` node and the answer must be a reference to a node in the
        `cloned` tree.

        Follow up: Solve the problem if repeated values on the tree are
        allowed.

        Constraints:
        - The number of nodes in the tree is in the range [1, 10^4].
        - The values of the nodes of the tree are unique.
        - `target` node is a node from the `original` tree and is not `null`.
        """
        if not original:
            return None

        if original is target:
            return cloned

        left = self.get_target_copy(self, original.left, cloned.left, target)
        if left:
            return left

        right = self.get_target_copy(self, original.right, cloned.right,
                                     target)
        if right:
            return right

        return None


# countArrangement
class Jan03:
    def count_arrangement(self, n: int) -> int:
        """Beautiful Arrangement
        Suppose you have n integers from 1 to `n`. We define a beautiful
        arrangement as an array that is constructed by these `n` numbers
        successfully if one of the following is true for the ith position
        (1 <= i <= `n`) in this array:
        - The number at the ith position is divisible by i.
        - i is divisible by the number at the ith position.

        Given an integer `n`, return the number of the beautiful arrangements
        that you can construct.

        Constraints:
        - 1 <= `n` <= 15
        """
        def backtrack(pos, visited):
            if pos < 1:
                return 1

            count = 0
            for i in range(n, 0, -1):
                if (not visited[i] and (pos % i == 0 or i % pos == 0)):
                    visited[i] = True
                    count += backtrack(pos-1, visited)
                    visited[i] = False
            return count

        if n < 4:
            return n
        visited = [False] * (n+1)
        return backtrack(n, visited)


# mergeTwoLists
class Jan04:
    def merge_two_lists(self, l1: ListNode, l2: ListNode) -> ListNode:
        """Merge Two Sorted Lists

        Merge two sorted linked lists and return it as a sorted list. The list
        should be made by splicing together the nodes of the first two lists.

        Constraints:
        - The number of nodes in both lists is in the range [0, 50].
        - -100 <= Node.val <= 100
        - Both `l1` and `l2` are sorted in non-decreasing order.

        """
        new_list = ListNode()
        curr = new_list

        while l1 and l2:
            if l1.val < l2.val:
                # curr.next = ListNode(l1.val)
                curr.next = l1
                l1 = l1.next
            else:
                # curr.next = ListNode(l2.val)
                curr.next = l2
                l2 = l2.next

            curr = curr.next

        # if l1:
        #     curr.next = l1
        # elif l2:
        #     curr.next = l2
        curr.next = l1 if l1 else l2

        return new_list.next


# deleteDuplicates
class Jan05:
    def delete_duplicates(self, head: ListNode) -> ListNode:
        """Remove Duplicates from Sorted List II

        Given the `head` of a sorted linked list, delete all nodes that have
        duplicate numbers, leaving only distinct numbers from the original
        list. Return the linked list sorted as well.

        Constraints:
        - The number of nodes in the list is in the range [0, 300].
        - -100 <= Node.val <= 100
        - The list is guaranteed to be sorted in ascending order.
        """
        sentinel = prev = ListNode(0, head)

        while head and head.next:
            if head.val == head.next.val:
                # Skip all duplicates.
                while head.next and head.val == head.next.val:
                    head = head.next
                head = head.next
                prev.next = head
            else:
                prev = prev.next
                head = head.next

        return sentinel.next


# findKthPositive
class Jan06:
    def find_kth_positive(self, arr: List[int], k: int) -> int:
        """Kth Missing Positive Number

        Given an array `arr` of positive integers sorted in a strictly
        increasing order, and an integer `k`.

        Find the `k`th positive integer that is missing from this array.

        Constraints:
        - 1 <= `arr.length` <= 1000
        - 1 <= `arr[i]` <= 1000
        - 1 <= `k` <= 1000
        - `arr[i]` < `arr[j]` for 1 <= i < j <= `arr.length`

        Hints:
        - Keep track of how many positive numbers are missing as you scan the
          array.
        """
        # Find the missing numbers between each consecutive integer pair.
        start = 0
        for i in range(len(arr)):
            # If difference is >= current k, substract from k.
            if k > (arr[i] - start - 1):
                k -= (arr[i] - start - 1)
            # The answer is between the current consecutive values.
            else:
                return start + k
            # Set starting point to the next index.
            start = arr[i]

        # Result is greater than the last value.
        return arr[len(arr) - 1] + k

        # Original attempt:
        # missing = arr[0] - 1

        # for i in range(len(arr) - 1):
        #     if (missing + (arr[i + 1] - arr[i] - 1)) > k:
        #         return arr[i] + (k - missing)
        #     else:
        #         missing += (arr[i + 1] - arr[i] - 1)

        # return len(arr) + missing + k


# lengthOfLongestSubstring
class Jan07:
    def length_of_longest_substring(self, s: str) -> int:
        """Longest Substring Without Repeating Characters

        Given a string `s`, find the length of the longest substring without
        repeating characters.
        """
        substr = set()
        slow = fast = longest = 0

        while slow < len(s) and fast < len(s):
            if s[fast] not in substr:
                # Add unseen char and move fast pointer forward.
                substr.add(s[fast])
                fast += 1
                # Check if new max length is found.
                longest = max(longest, fast - slow)
            else:
                # Move slow pointer forward.
                substr.remove(s[slow])
                slow += 1

        return longest


# findRoot and arrayStringsAreEqual
class Jan08:
    def find_root(self, tree: List['Node']) -> 'Node':
        """Find Root of N-Ary Tree

        You are given all the nodes of an N-ary tree as an array of Node
        objects, where each node has a unique value.

        Return the root of the N-ary `tree`.

        Constraints:
        - The total number of nodes is between [1, 5 * 104].
        - Each node has a unique value.

        Follow up:
        - Could you solve this problem in constant space complexity with a
          linear time algorithm?

        Hints:
        - Node with indegree 0 is the root.
        """

        '''
        # Time and Space complexity: O(n)
        # Faster algorithm but takes more space.

        # Contains all child nodes.
        seen = set()

        # Add all child nodes to set.
        for node in tree:
            for child in node.children:
                # Add each node's unique value.
                seen.add(child.val)

        # Find node that is not in set.
        for node in tree:
            if node.val not in seen:
                return node

        # Time complexity: O(n)
        # Space complexity: O(1)
        '''

        val_sum = 0

        for node in tree:
            # Parent node value is added.
            val_sum += node.val
            for child in node.children:
                # Child node value is deducted.
                val_sum -= child.val

        for node in tree:
            if node.val == val_sum:
                return node

    def array_strings_are_equal(self, word1: List[str],
                                word2: List[str]) -> bool:
        """Check If Two String Arrays are Equivalent
        Given two string arrays `word1` and `word2`, return true if the two
        arrays represent the same string, and false otherwise.

        A string is represented by an array if the array elements concatenated
        in order forms the string.

        Constraints:
        - 1 <= word1.length, word2.length <= 103
        - 1 <= word1[i].length, word2[i].length <= 103
        - 1 <= sum(word1[i].length), sum(word2[i].length) <= 103
        - word1[i] and word2[i] consist of lowercase letters.

        Hints:
        - Concatenate all strings in the first array into a single string in
          the given order, the same for the second array.
        - Both arrays represent the same string if and only if the generated
          strings are the same.
        """
        str1 = str2 = ""

        for piece in word1:
            str1 += piece
        for piece in word2:
            str2 += piece

        return str1 == str2


# ladderLength
class Jan09:
    def ladder_length(self, beginWord: str, endWord: str,
                      wordList: List[str]) -> int:
        """Word Ladder

        Given two words `beginWord` and `endWord`, and a dictionary
        `wordList`, return the length of the shortest transformation sequence
        from `beginWord` to `endWord`, such that:
        - Only one letter can be changed at a time.
        - Each transformed word must exist in the word list.

        Return 0 if there is no such transformation sequence.

        Constraints:
        - 1 <= `beginWord.length` <= 100
        - `endWord.length` == `beginWord.length`
        - 1 <= `wordList.length` <= 5000
        - `wordList[i].length` == `beginWord.length`
        - `beginWord`, `endWord`, and `wordList[i]` consist of lowercase
          English letters.
        - `beginWord` != `endWord`
        - All the strings in `wordList` are unique.
        """
        from collections import deque
        from typing import DefaultDict

        # Solution given by LeetCode discussion:
        if (endWord not in wordList or not endWord
                or not beginWord or not wordList):
            return 0
        length = len(beginWord)
        all_combo_dict = DefaultDict(list)
        for word in wordList:
            for i in range(length):
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
        queue = deque([(beginWord, 1)])
        visited = set()
        visited.add(beginWord)
        while queue:
            current_word, level = queue.popleft()
            for i in range(length):
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]
                for word in all_combo_dict[intermediate_word]:
                    if word == endWord:
                        return level + 1
                    if word not in visited:
                        visited.add(word)
                        queue.append((word, level + 1))
        return 0


# createSortedArray
class Jan10:
    def create_sorted_array(self, instructions: List[int]) -> int:
        """Create Sorted Array through Instructions

        Given an integer array `instructions`, you are asked to create a sorted
        array from the elements in `instructions`. You start with an empty
        container `nums`. For each element from left to right in
        `instructions`, insert it into `nums`. The cost of each insertion is
        the minimum of the following:

        - The number of elements currently in `nums` that are strictly less
          than `instructions[i]`.
        - The number of elements currently in `nums` that are strictly greater
          than `instructions[i]`.

        For example, if inserting element 3 into `nums` = [1,2,3,5], the cost
        of insertion is min(2, 1) (elements 1 and 2 are less than 3, element 5
        is greater than 3) and `nums` will become [1,2,3,3,5].

        Return the total cost to insert all elements from `instructions` into
        `nums`. Since the answer may be large, return it modulo 10^9 + 7
        """
        import bisect

        nums = []
        cost = 0

        for i, val in enumerate(instructions):
            left = bisect.bisect_left(nums, val)
            right = bisect.bisect(nums, val)
            cost += min(left, i - right)
            # nums.insert(r, val)
            nums[right:right] = [val]

        return cost % (10**9 + 7)
