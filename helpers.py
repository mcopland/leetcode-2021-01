from models import TreeNode
from typing import List


def tree_constructor(node_list: List[int]) -> TreeNode:
    root = TreeNode(node_list[0])
    current = [root]
    ptr = 1

    while ptr < len(node_list):
        next_node = []
        for n in current:
            if node_list[ptr]:
                n.left = TreeNode(node_list[ptr])
                next_node.append(n.left)
            ptr += 1
            if node_list[ptr]:
                n.right = TreeNode(node_list[ptr])
                next_node.append(n.right)
            ptr += 1
        current = next_node
    return root
