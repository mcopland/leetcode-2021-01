from collections import Counter, defaultdict
from math import comb
from models import ListNode, NestedInteger, Node, TreeNode
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


# merge
class Jan11:
    def merge(self, nums1: List[int], m: int, nums2: List[int],
              n: int) -> None:
        """Merge Sorted Array

        Given two sorted integer arrays `nums1` and `nums2`, merge `nums2` into
        `nums1` as one sorted array.

        The number of elements initialized in `nums1` and `nums2` are `m` and
        `n` respectively. You may assume that `nums1` has enough space (size
        that is equal to `m` + `n`) to hold additional elements from `nums2`.

        Do not return anything, modify `nums1` in-place instead.

        Constraints:
        - 0 <= `n`, `m` <= 200
        - 1 <= `n` + `m` <= 200
        - `nums1.length` == `m` + `n`
        - `nums2.length` == `n`
        - -109 <= `nums1[i]`, `nums2[i]` <= 109

        Hints:
        - You can easily solve this problem if you simply think about two
        elements at a time rather than two arrays. We know that each of the
        individual arrays is sorted. What we don't know is how they will
        intertwine. Can we take a local decision and arrive at an optimal
        solution?
        - If you simply consider one element each at a time from the two arrays
        and make a decision and proceed accordingly, you will arrive at the
        optimal solution.
        """
        i = m + n - 1
        ptr1, ptr2 = m - 1, n - 1

        while ptr1 >= 0 and ptr2 >= 0:
            if nums1[ptr1] > nums2[ptr2]:
                nums1[i] = nums1[ptr1]
                ptr1 -= 1
            else:
                nums1[i] = nums2[ptr2]
                ptr2 -= 1
            i -= 1

        while ptr2 >= 0:
            nums1[i] = nums2[ptr2]
            i -= 1
            ptr2 -= 1


# addTwoNumbers
class Jan12:
    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """Add Two Numbers
        You are given two non-empty linked lists representing two non-negative
        integers. The digits are stored in reverse order, and each of their
        nodes contains a single digit. Add the two numbers and return the sum
        as a linked list.

        You may assume the two numbers do not contain any leading zero, except
        the number 0 itself.

        Constraints:
        - The number of nodes in each linked list is in the range [1, 100].
        - 0 <= Node.val <= 9
        - It is guaranteed that the list represents a number that does not have
        leading zeros.
        """
        l1_sum = l2_sum = 0

        factor = 1
        while l1:
            l1_sum += l1.val * factor
            factor *= 10
            l1 = l1.next

        factor = 1
        while l2:
            l2_sum += l2.val * factor
            factor *= 10
            l2 = l2.next

        res = ListNode()
        curr = res
        total = l1_sum + l2_sum
        while total > 0:
            curr.val = total % 10
            total //= 10
            if total > 0:
                curr.next = ListNode()
                curr = curr.next

        return res


# numRescueBoats
class Jan13:
    def num_rescue_boats(self, people: List[int], limit: int) -> int:
        """Boats to Save People

        The i-th person has weight `people[i]`, and each boat can carry a
        maximum weight of `limit`.

        Each boat carries at most 2 `people` at the same time, provided the sum
        of the weight of those `people` is at most `limit`.

        Return the minimum number of boats to carry every given person.  (It is
        guaranteed each person can be carried by a boat.)

        Constraints:
        - 1 <= `people.length` <= 50000
        - 1 <= `people[i]` <= `limit` <= 30000
        """
        count = 0
        ptr_min = 0
        ptr_max = len(people) - 1

        people.sort()

        while ptr_min <= ptr_max:
            if people[ptr_min] + people[ptr_max] <= limit:
                ptr_min += 1
            ptr_max -= 1
            count += 1

        return count


# minOperations
class Jan14:
    def min_operations(self, nums: List[int], x: int) -> int:
        """Minimum Operations to Reduce X to Zero

        You are given an integer array `nums` and an integer `x`. In one
        operation, you can either remove the leftmost or the rightmost element
        from the array `nums` and subtract its value from `x`. Note that this
        modifies the array for future operations.

        Return the minimum number of operations to reduce `x` to exactly 0 if
        it's possible, otherwise, return -1.

        Constraints:
        - 1 <= `nums.length` <= 10^5
        - 1 <= `nums[i]` <= 10^4
        - 1 <= `x` <= 10^9

        Hints:
        - Think in reverse; instead of finding the minimum prefix + suffix,
          find the maximum subarray.
        - Finding the maximum subarray is standard and can be done greedily.
        """
        # Remove the longest sequence so that what's left is equal to x.
        current_sum = 0
        left = 0
        longest = -1
        target = sum(nums) - x

        for right in range(len(nums)):
            current_sum += nums[right]
            while current_sum > target and left <= right:
                current_sum -= nums[left]
                left += 1

            if current_sum == target:
                longest = max(longest, right - left + 1)

        return len(nums) - longest if longest != -1 else -1


# depthSum and getMaximumGenerated
class Jan15:
    def depth_sum(self, nestedList: List[NestedInteger]) -> int:
        """Nested List Weight Sum

        You are given a nested list of integers `nestedList`. Each element is
        either an integer or a list whose elements may also be integers or
        other lists.

        The depth of an integer is the number of lists that it is inside of.
        For example, the nested list [1,[2,2],[[3],2],1] has each integer's
        value set to its depth.

        Return the sum of each integer in `nestedList` multiplied by its depth.

        Constraints:
        - 1 <= `nestedList.length` <= 50
        - The values of the integers in the nested list is in the range
          [-100, 100].
        - The maximum depth of any integer is less than or equal to 50.
        """
        def dfs(nested_list, depth):
            total = 0

            for i in nested_list:
                # # if isinstance(i, int):
                #     total += i * depth
                # else:
                #     total += dfs(i, depth + 1)
                if i.isInteger():
                    total += i.getInteger() * depth
                else:
                    total += dfs(i.getList(), depth + 1)

            return total

        return dfs(nestedList, 1)

    def get_maximum_generated(self, n: int) -> int:
        """Get Maximum in Generated Array

        You are given an integer `n`. An array nums of length `n` + 1 is
        generated in the following way:
        - `nums[0]` = 0
        - `nums[1]` = 1
        - `nums[2 * i]` = `nums[i]` when 2 <= 2 * i <= `n`
        - `nums[2 * i + 1]` = `nums[i]` + `nums[i + 1]` when 2 <= 2 * i + 1 <=
          `n`

        Return the maximum integer in the array nums​​​.

        Constraints:
        - 0 <= n <= 100

        Hints:
        - Try generating the array.
        - Make sure not to fall in the base case of 0.
        """
        if n == 0:
            return 0

        nums = [0, 1]

        for i in range(2, n+1):
            if i % 2 == 0:
                nums.append(nums[i//2])
            else:
                nums.append(nums[i//2] + nums[i//2 + 1])

        return max(nums)


# findKthLargest
class Jan16:
    def find_kth_largest(self, nums: List[int], k: int) -> int:
        """Kth Largest Element in an Array

        Find the `k`th largest element in an unsorted array. Note that it is
        the `k`th largest element in the sorted order, not the `k`th distinct
        element.

        Note:
        - You may assume k is always valid, 1 ≤ k ≤ array's length.
        """
        # nums.sort(reverse=True)
        # return nums[k-1]
        return sorted(nums, reverse=True)[k-1]


# countVowelStrings
class Jan17:
    def count_vowel_strings(self, n: int) -> int:
        """Count Sorted Vowel Strings

        Given an integer `n`, return the number of strings of length `n` that
        consist only of vowels (a, e, i, o, u) and are lexicographically
        sorted.

        A string `s` is lexicographically sorted if for all valid `i`, `s[i]`
        is the same as or comes before `s[i+1]` in the alphabet.

        Constraints:
        - 1 <= `n` <= 50

        Hints:
        - For each character, its possible values will depend on the value of
        its previous character, because it needs to be not smaller than it.
        - Think backtracking. Build a recursive function count(`n`,
        `last_character`) that counts the number of valid strings of length `n`
        and whose first characters are not less than last_character.
        - In this recursive function, iterate on the possible characters for
        the first character, which will be all the vowels not less than
        last_character, and for each possible value `c`, increase the answer by
        count(`n-1`, `c`).
        """
        # res = 0

        # for i in range(n + 1):
        #     sum = 0
        #     for j in range(i + 1):
        #         sum += j + 1
        #         res += sum

        # return res

        # return (n + 1) * (n + 2) * (n + 3) * (n + 4) // 24

        return comb(n + 4, 4)


# maxOperations
class Jan18:
    def max_operations(self, nums: List[int], k: int) -> int:
        """Max Number of K-Sum Pairs

        You are given an integer array `nums` and an integer `k`.

        In one operation, you can pick two numbers from the array whose sum
        equals `k` and remove them from the array.

        Return the maximum number of operations you can perform on the array.

        Constraints:
        - 1 <= `nums.length` <= 10^5
        - 1 <= `nums[i]` <= 10^9
        - 1 <= `k` <= 10^9

        Hints:
        - The abstract problem asks to count the number of disjoint pairs with
          a given sum `k`.
        - For each possible value x, it can be paired up with `k` - x.
        - The number of such pairs equals to min(count(x), count(`k` - x)),
          unless that x = `k` / 2, where the number of such pairs will be floor
          count(x) / 2).
        """
        count = 0
        tracker = defaultdict(int)
        # unpaired = set()

        # for i in nums:
        #     if abs(k - i) in unpaired:
        #         unpaired.remove(k-i)
        #         count += 1
        #     else:
        #         unpaired.add(i)

        for i in nums:
            if tracker[k - i] > 0:
                tracker[k - i] -= 1
                count += 1
            else:
                tracker[i] += 1

        return count


# longestPalindrome
class Jan19:
    def longest_palindrome(self, s: str) -> str:
        """Longest Palindromic Substring

        Given a string `s`, return the longest palindromic substring in `s`.

        Constraints:
        - 1 <= `s.length` <= 1000
        - `s` consist of only digits and English letters (lower-case and/or
          upper-case)

        Hints:
        - How can we reuse a previously computed palindrome to compute a larger
        palindrome?
        - If "aba" is a palindrome, is "xabax" a palindrome? Similarly is
          "xabay" a palindrome?

        Complexity based hint:
        - If we use brute-force and check whether for every start and end
          position a substring is a palindrome we have O(n^2) start - end pairs
          and O(n) palindromic checks. Can we reduce the time for palindromic
          checks to O(1) by reusing some previous computation.
        """
        # Source: Discussion board
        res = ""
        for i in range(len(s)):
            # Odd character count
            current = self.find_palindrome(self, s, i, i)
            if len(current) > len(res):
                res = current
            # Even character count
            current = self.find_palindrome(self, s, i, i+1)
            if len(current) > len(res):
                res = current
        return res

    # Search from middle to outer indices
    def find_palindrome(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]


# isValid
class Jan20:
    def is_valid(self, s: str) -> bool:
        """Valid Parentheses

        Given a string `s` containing just the characters '(', ')', '{', '}',
        '[' and ']', determine if the input string is valid.

        An input string is valid if:
        - Open brackets must be closed by the same type of brackets.
        - Open brackets must be closed in the correct order.

        Constraints:
        - 1 <= `s.length` <= 104
        - `s` consists of parentheses only '()[]{}'.

        Hints:
        - An interesting property about a valid parenthesis expression is that
          a sub-expression of a valid expression should also be a valid
          expression. Can we exploit this recursive structure somehow?
        - What if whenever we encounter a matching pair of parenthesis in the
          expression, we simply remove it from the expression? This would keep
          on shortening the expression.
        - The stack data structure can come in handy here in representing this
          recursive structure of the problem. We can't really process this from
          the inside out because we don't have an idea about the overall
          structure. But, the stack can help us process this recursively i.e.
          from outside to inwards.
        """
        stack = []
        pairs = {')': '(', '}': '{', ']': '['}

        # Add opening brackets to stack, pop if a match is found.
        for char in s:
            if char in pairs:
                stack_top = stack.pop() if stack else '!'
                if pairs[char] != stack_top:
                    return False
            else:
                stack.append(char)

        # String is valid if stack is empty.
        return not stack


# mostCompetitive
class Jan21:
    def most_competitive(self, nums: List[int], k: int) -> List[int]:
        """Find the Most Competitive Subsequence

        Given an integer array `nums` and a positive integer `k`, return the
        most competitive subsequence of nums of size `k`.

        An array's subsequence is a resulting sequence obtained by erasing some
        (possibly zero) elements from the array.

        We define that a subsequence a is more competitive than a subsequence b
        (of the same length) if in the first position where a and b differ,
        subsequence a has a number less than the corresponding number in b. For
        example, [1,3,4] is more competitive than [1,3,5] because the first
        position they differ is at the final number, and 4 is less than 5.

        Constraints:
        - 1 <= `nums.length` <= 10^5
        - 0 <= `nums[i]` <= 10^9
        - 1 <= `k` <= `nums.length`

        Hint:
        - In lexicographical order, the elements to the left have higher
          priority than those that come after. Can you think of a strategy that
          incrementally builds the answer from left to right?
        """
        stack = []
        for i, num in enumerate(nums):
            while stack and stack[-1] > num and len(stack) + len(nums) - i > k:
                stack.pop()
            if len(stack) < k:
                stack.append(num)
        return stack


# isOneEditDistance and closeStrings
class Jan22:
    def is_one_edit_distance(self, s: str, t: str) -> bool:
        """One Edit Distance

        Given two strings `s` and `t`, return true if they are both one edit
        distance apart, otherwise return false.

        A string `s` is said to be one distance apart from a string `t` if you
        can:
        - Insert exactly one character into `s` to get `t`.
        - Delete exactly one character from `s` to get `t`.
        - Replace exactly one character of `s` with a different character to
          get `t`.

        Constraints:
        - 0 <= `s.length` <= 10^4
        - 0 <= `t.length` <= 10^4
        - `s` and `t` consist of lower-case letters, upper-case letters and/or
          digits.
        """
        # Ensure that s is shorter than t.
        if len(s) > len(t):
            s, t = t, s

        # Length difference cannot be more than 1.
        if len(t) - len(s) > 1:
            return False

        for i in range(len(s)):
            if s[i] != t[i]:
                if len(s) == len(t):
                    return s[i + 1:] == t[i + 1:]
                else:
                    return s[i:] == t[i + 1:]

        return len(s) + 1 == len(t)

    def close_strings(self, word1: str, word2: str) -> bool:
        """Determine if Two Strings Are Close

        Two strings are considered close if you can attain one from the other
        using the following operations:
        - Operation 1: Swap any two existing characters.
          - For example, abcde -> aecdb
        - Operation 2: Transform every occurrence of one existing character
          into another existing character, and do the same with the other
          character.
          - For example, aacabb -> bbcbaa (all a's turn into b's, and all b's
            turn into a's)

        You can use the operations on either string as many times as necessary.

        Given two strings, `word1` and `word2`, return true if `word1` and
        `word2` are close, and false otherwise.

        Constraints:
        - 1 <= `word1.length`, `word2.length` <= 10^5
        - `word1` and `word2` contain only lowercase English letters.

        Hints:
        - Operation 1 allows you to freely reorder the string.
        - Operation 2 allows you to freely reassign the letters' frequencies.
        """
        if len(word1) != len(word2):
            return False

        count1 = Counter(word1)
        count2 = Counter(word2)

        return count1.keys() == count2.keys()\
            and Counter(count1.values()) == Counter(count2.values())


# diagonalSort
class Jan23:
    def diagonal_sort(self, mat: List[List[int]]) -> List[List[int]]:
        """Sort the Matrix Diagonally

        A matrix diagonal is a diagonal line of cells starting from some cell
        in either the topmost row or leftmost column and going in the
        bottom-right direction until reaching the matrix's end. For example,
        the matrix diagonal starting from `mat[2][0]`, where `mat` is a 6 x 3
        matrix, includes cells `mat[2][0]`, `mat[3][1]`, and `mat[4][2]`.

        Given an m x n matrix `mat` of integers, sort each matrix diagonal in
        ascending order and return the resulting matrix.

        Constraints:
        - m == `mat.length`
        - n == `mat[i].length`
        - 1 <= m, n <= 100
        - 1 <= `mat[i][j]` <= 100

        Hints:
        - Use a data structure to store all values of each diagonal.
        - How to index the data structure with the id of the diagonal?
        - All cells in the same diagonal (i,j) have the same difference so we
          can get the diagonal of a cell using the difference i-j.
        """
        m = len(mat)
        n = len(mat[0])
        d = defaultdict(list)

        # Collect diagonals.
        for i in range(m):
            for j in range(n):
                d[i - j].append(mat[i][j])

        # Sort each diagonal in reverse for pop().
        for i in d:
            d[i].sort(reverse=True)

        # Rewrite diagonals in ascending order.
        for i in range(m):
            for j in range(n):
                mat[i][j] = d[i - j].pop()

        return mat


# mergeKLists
class Jan24:
    def merge_k_lists(self, lists: List[ListNode]) -> ListNode:
        """Merge k Sorted Lists

        You are given an array of k linked-lists `lists`, each linked-list is
        sorted in ascending order.

        Merge all the linked-lists into one sorted linked-list and return it.

        Constraints:
        - k == `lists.length`
        - 0 <= k <= 10^4
        - 0 <= `lists[i].length` <= 500
        - -10^4 <= `lists[i][j]` <= 10^4
        - `lists[i]` is sorted in ascending order.
        - The sum of `lists[i].length` won't exceed 10^4.
        """
        nodes = []
        head = curr = ListNode(0)

        for k in lists:
            while k:
                nodes.append(k.val)
                k = k.next
        for x in sorted(nodes):
            curr.next = ListNode(x)
            curr = curr.next

        return head.next


# kLengthApart
class Jan25:
    def k_length_apart(self, nums: List[int], k: int) -> bool:
        """Check If All 1's Are at Least Length K Places Away

        Given an array `nums` of 0's and 1's and an integer `k`, return True if
        all 1's are at least `k` places away from each other, otherwise return
        False.

        Constraints:
        - 1 <= `nums.length` <= 10^5
        - 0 <= `k` <= `nums.length`
        - `nums[i]` is 0 or 1

        Hint:
        - Each time you find a number 1, check whether or not it is `k` or more
          places away from the next one. If it's not, return false.
        """
        distance = k

        for num in nums:
            if num == 1:
                if distance < k:
                    return False
                # Reset distance.
                distance = 0
            else:
                # Increase distance between 1's.
                distance += 1

        return True


# minimumEffortPath
class Jan26:
    def minimum_effort_path(self, heights: List[List[int]]) -> int:
        """Path With Minimum Effort

        You are a hiker preparing for an upcoming hike. You are given
        `heights`, a 2D array of size rows x columns, where `heights[row][col]`
        represents the height of cell (row, col). You are situated in the
        top-left cell, (0, 0), and you hope to travel to the bottom-right cell,
        (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or
        right, and you wish to find a route that requires the minimum effort.

        A route's effort is the maximum absolute difference in `heights`
        between two consecutive cells of the route.

        Return the minimum effort required to travel from the top-left cell to
        the bottom-right cell.

        Constraints:
        - rows == `heights.length`
        - columns == `heights[i].length`
        - 1 <= rows, columns <= 100
        - 1 <= `heights[i][j]` <= 10^6

        Hints:
        - Consider the grid as a graph, where adjacent cells have an edge with
          cost of the difference between the cells.
        - If you are given threshold k, check if it is possible to go from
          (0, 0) to (n-1, m-1) using only edges of ≤ k cost.
        - Binary search the k value.
        """
        # Binary Search using DFS
        # row = len(heights)
        # col = len(heights[0])

        # def canReachDestinaton(x, y, mid):
        #     if x == row-1 and y == col-1:
        #         return True
        #     visited[x][y] = True
        #     for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
        #         adjacent_x = x + dx
        #         adjacent_y = y + dy
        #         if (0 <= adjacent_x < row
        #                 and 0 <= adjacent_y < col
        #                 and not visited[adjacent_x][adjacent_y]):
        #             current_difference = abs(
        #                 heights[adjacent_x][adjacent_y]-heights[x][y])
        #             if current_difference <= mid:
        #                 visited[adjacent_x][adjacent_y] = True
        #                 if canReachDestinaton(adjacent_x, adjacent_y, mid):
        #                     return True

        # left = 0
        # right = 10000000
        # while left < right:
        #     mid = (left + right) // 2
        #     visited = [[False]*col for _ in range(row)]
        #     if canReachDestinaton(0, 0, mid):
        #         right = mid
        #     else:
        #         left = mid + 1

        # return left

        # Fastest submission (452 vs 1892 ms)
        import heapq

        m = len(heights)
        n = len(heights[0])
        boundary = [(0, 0, 0)]
        visited = set()
        ans = 0
        todo = []
        while True:
            if todo:
                a, i, j = todo.pop()
            else:
                a, i, j = heapq.heappop(boundary)
                ans = a
            if (i, j) == (m-1, n-1):
                return ans
            for x, y in ((i, j-1), (i-1, j), (i+1, j), (i, j+1)):
                if 0 <= x < m and 0 <= y < n:
                    if (x, y) not in visited:
                        d = abs(heights[x][y] - heights[i][j])
                        if d <= ans:
                            todo.append((d, x, y))
                        else:
                            heapq.heappush(boundary, (d, x, y))
            visited.add((i, j))


# concatenatedBinary
class Jan27:
    def concatenated_binary(self, n: int) -> int:
        """Concatenation of Consecutive Binary Numbers

        Given an integer `n`, return the decimal value of the binary string
        formed by concatenating the binary representations of 1 to `n` in
        order, modulo 10^9 + 7.

        Constraints:
        - 1 <= `n` <= 105

        Hint:-
        - Express the `nth` number value in a recursion formula and think about
        how we can do a fast evaluation.
        """
        length = 0
        result = 0
        modulo = 10**9 + 7

        # Iterate from 1 to n.
        for i in range(1, n+1):
            # Find length of i's binary representation. Increase if power of 2.
            # & bitwise AND
            if i & (i-1) == 0:
                length += 1
            # Update result to result << length | i.
            # << bitwise left shift, | bitwise OR
            result = ((result << length) | i) % modulo

        return result


# getSmallestString
class Jan28:
    def get_smallest_string(self, n: int, k: int) -> str:
        # Initialize result to 'a' * required length
        res = ['a'] * n
        # Current value is 'a' * length
        val = n

        # Iterate backwards
        for i in range(n-1, -1, -1):
            if val == k:
                # Result found
                break
            # Remove value of placeholder 'a'
            val -= 1
            # Replace with either (k - value) or 'z'
            res[i] = chr(96 + min(k - val, 26))
            # Add value of new char
            val += ord(res[i]) - 96

        return ''.join(res)


# countCornerRectangles and verticalTraversal
class Jan29:
    def count_corner_rectangles(self, grid: List[List[int]]) -> int:
        """Number Of Corner Rectangles

        Given a `grid` where each entry is only 0 or 1, find the number of
        corner rectangles.

        A corner rectangle is 4 distinct 1s on the `grid` that form an
        axis-aligned rectangle. Note that only the corners need to have the
        value 1. Also, all four 1s used must be distinct.

        Notes:
        - The number of rows and columns of `grid` will each be in the range
          [1, 200].
        - Each `grid[i][j]` will be either 0 or 1.
        - The number of 1s in the `grid` will be at most 6000.

        Hint:
        - For each pair of 1s in the new row (say at `new_row[i]` and
          `new_row[j]`), we could create more rectangles where that pair forms
          the base. The number of new rectangles is the number of times some
          previous row had `row[i] = row[j] = 1`.
        """
        ans = 0
        prevs = []

        for row in grid:
            ones = {idx for idx, val in enumerate(row) if val}
            for prev in prevs:
                matches = len(ones & prev)
                ans += matches * (matches-1) // 2
            prevs.append(ones)

        return ans

    def vertical_traversal(self, root: TreeNode) -> List[List[int]]:
        """Vertical Order Traversal of a Binary Tree

        Given the `root` of a binary tree, calculate the vertical order
        traversal of the binary tree.

        For each node at position (x, y), its left and right children will be
        at positions (x - 1, y - 1) and (x + 1, y - 1) respectively.

        The vertical order traversal of a binary tree is a list of non-empty
        reports for each unique x-coordinate from left to right. Each report is
        a list of all nodes at a given x-coordinate. The report should be
        primarily sorted by y-coordinate from highest y-coordinate to lowest.
        If any two nodes have the same y-coordinate in the report, the node
        with the smaller value should appear earlier.

        Return the vertical order traversal of the binary tree.

        Constraints:
        - The number of nodes in the tree is in the range [1, 1000].
        - 0 <= Node.val <= 1000
        """
        # Traverse graph and use dict to group by x.
        group = defaultdict(list)
        queue = [(root, 0, 0)]

        for node in queue:
            if node[0]:
                point, x, y = node
                group[x].append((point.val, y))
                if point.left:
                    queue.append((point.left, x-1, y+1))
                if point.right:
                    queue.append((point.right, x+1, y+1))

        # Group by vertical line.
        res = []
        for x in sorted(group):
            res.append(group[x])

        # Sort by value then depth.
        for i, item in enumerate(res):
            item.sort(key=lambda x: (x[1], x[0]))
            # item.sort(key=lambda x: x[0])
            # item.sort(key=lambda x: x[1])
            res[i] = [val for val, y in item]

        return res


# minimumDeviation
class Jan30:
    def minimum_deviation(self, nums: List[int]) -> int:
        """Minimize Deviation in Array

        You are given an array `nums` of n positive integers.

        You can perform two types of operations on any element of the array any
        number of times:
        - If the element is even, divide it by 2.
            - For example, if the array is [1, 2, 3, 4], then you can do this
              operation on the last element, and the array will be
              [1, 2, 3, 2].
        - If the element is odd, multiply it by 2.
            - For example, if the array is [1, 2, 3, 4], then you can do this
              operation on the first element, and the array will be
              [2, 2, 3, 4].

        The deviation of the array is the maximum difference between any two
        elements in the array.

        Return the minimum deviation the array can have after performing some
        number of operations.

        Constraints:
        - n == `nums.length`
        - 2 <= n <= 10^5
        - 1 <= `nums[i]` <= 10^9

        Hints:
        - Assume you start with the minimum possible value for each number so
          you can only multiply a number by 2 till it reaches its maximum
          possible value.
        - If there is a better solution than the current one, then it must have
          either its maximum value less than the current maximum value, or the
          minimum value larger than the current minimum value.
        - Since that we only increase numbers (multiply them by 2), we cannot
          decrease the current maximum value, so we must multiply the current
          minimum number by 2.
        """
        import heapq
        from math import inf

        evens = []          # Initialize max-heap.
        deviation = inf     # Deviation result.
        min_val = inf       # Track smallest element in evens.

        # Since heapq is a min-heap, use negative numbers to mimic a max-heap.
        for num in nums:
            # If even, put into heap.
            if num % 2 == 0:
                evens.append(-num)
                min_val = min(min_val, num)
            # If odd, multiply by 2 and put into heap.
            else:
                evens.append(-num*2)
                min_val = min(min_val, num*2)

        # Convert list into min-heap.
        heapq.heapify(evens)

        while evens:
            # Take out max number.
            current_val = -heapq.heappop(evens)
            # Update minimum deviation.
            deviation = min(deviation, current_val-min_val)
            # If max number is even, divide by 2 and push into evens.
            if current_val % 2 == 0:
                min_val = min(min_val, current_val//2)
                heapq.heappush(evens, -current_val//2)
            # Repeat until max number in evens is odd.
            else:
                break

        return deviation


# nextPermutation
class Jan31:
    def next_permutation(self, nums: List[int]) -> None:
        """Next Permutation

        Implement next permutation, which rearranges numbers into the
        lexicographically next greater permutation of numbers.

        If such an arrangement is not possible, it must rearrange it as the
        lowest possible order (i.e., sorted in ascending order).

        The replacement must be in place and use only constant extra memory.

        Constraints:
        - 1 <= `nums.length` <= 100
        - 0 <= `nums[i]` <= 100
        """
        i = j = len(nums)-1

        # Find longest non-increasing suffix.
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        # If nums is in descending order:
        if i == 0:
            nums.reverse()
            return

        # Find rightmost successor to pivot (i-1) in suffix.
        while nums[j] <= nums[i-1]:
            j -= 1

        # Swap with pivot.
        nums[i-1], nums[j] = nums[j], nums[i-1]

        # Reverse suffix.
        nums[i:] = nums[len(nums)-1: i-1: -1]

    #     # Sample 28 ms submission
    #     n = len(nums)
    #     left_idx = self.find_first_decreasing(self, nums)
    #     if left_idx == n:
    #         self.reverse(self, nums, 0, n - 1)
    #         return

    #     right_idx = self.find_first_greater_num(self, nums, left_idx)
    #     nums[left_idx], nums[right_idx] = nums[right_idx], nums[left_idx]
    #     self.reverse(self, nums, left_idx + 1, n - 1)

    # def find_first_decreasing(self, nums):
    #     n = len(nums)
    #     for i in range(n - 2, -1, -1):
    #         if nums[i] < nums[i + 1]:
    #             return i

    #     return n

    # def find_first_greater_num(self, nums, stop):
    #     n = len(nums)
    #     for i in range(n - 1, stop, -1):
    #         if nums[i] > nums[stop]:
    #             return i

    # def reverse(self, nums, i, j):
    #     while i < j:
    #         nums[i], nums[j] = nums[j], nums[i]
    #         i += 1
    #         j -= 1
