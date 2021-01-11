# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
