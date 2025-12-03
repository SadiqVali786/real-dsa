from collections import deque
from dataclasses import dataclass


@dataclass
class TreeNode:
    data: int
    left: 'TreeNode' | None = None
    right: 'TreeNode' | None = None


@dataclass
class TreeNodeMetadata:
    min: float
    max: float
    size: int
    is_bst: bool


class BinarySearchTree:

    def __init__(self):
        self.root: TreeNode | None = None

    def build_bst(self, nodes: list[int]):
        """
            Builds a BST from a list of nodes.
            Args:
                nodes: The list of nodes to build the BST from.
            Returns:
                TreeNode: The root of the BST.
            Time Complexity: O(n log n)
            Space Complexity: O(n)
        """
        def insert(node: TreeNode | None, data: int):
            if node is None:
                return TreeNode(data)
            if data < node.data:
                node.left = insert(node.left, data)
            else:
                node.right = insert(node.right, data)
            return node
        
        for node in nodes:
            self.root = insert(self.root, node)
        
        return self.root

    def search(self, data: int):
        """
            Searches for a node in the BST.
            Args:
                data: The data of the node to search for.
            Returns:
                TreeNode: The node if found, None otherwise.
            Time Complexity: O(log n) in average case, O(n) in worst case
            Space Complexity: O(1)
        """
        def helper(node: TreeNode | None, data: int):
            if node is None:
                return False
            if node.data == data:
                return True
            if data < node.data:
                return helper(node.left, data)
            else:
                return helper(node.right, data)
        
        return helper(self.root, data)

    def min(self):
        """
            Finds the minimum node in the BST.
            Returns:
                TreeNode: The minimum node in the BST.
            Time Complexity: O(log n)
            Space Complexity: O(1)
        """
        temp = self.root
        while temp.left is not None:
            temp = temp.left
        return temp.data

    def max(self):
        """
            Finds the maximum node in the BST.
            Returns:
                TreeNode: The maximum node in the BST.
            Time Complexity: O(log n)
            Space Complexity: O(1)
        """
        temp = self.root
        while temp.right is not None:
            temp = temp.right
        return temp.data

    def delete(self, data: int):
        """
            Deletes a node from the BST.
            Args:
                data: The data of the node to delete.
            Returns:
                TreeNode: The root of the BST.
            Time Complexity: O(log n)
            Space Complexity: O(log n)
        """
        def find_successor(node: TreeNode | None):
            temp = node
            while temp.left is not None:
                temp = temp.left
            return temp.data

        def helper(node: TreeNode | None, data: int):
            # base case
            if node is None:
                return None
            # recursive case
            if data < node.data:
                node.left = helper(node.left, data)
            elif data > node.data:
                node.right = helper(node.right, data)
            else:
                # Handling leaf node
                if (node.left is None) and (node.right is None):
                    return None
                # Hanlding internal node with one child
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                # Handling internal node with two children
                successor = find_successor(node.right)
                node.data = successor
                node.right = helper(node.right, successor)
            return node

        self.root = helper(self.root, data)
        return self.root

    def is_bst(self):
        """
            Checks if the BST is valid.
            Returns:
                bool: True if the BST is valid, False otherwise.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def helper(node: TreeNode | None, min_val: int, max_val: int):
            if node is None:
                return True
            if node.data <= min_val or node.data >= max_val:
                return False
            left_valid = helper(node.left, min_val, node.data)
            right_valid = helper(node.right, node.data, max_val)
            return left_valid and right_valid
        
        return helper(self.root, float('-inf'), float('inf'))

    def lca(self, value1: int, value2: int): # lowest common ancestor
        """
            Finds the lowest common ancestor of two nodes.
            Returns:
                int: The lowest common ancestor of the two nodes, None if not found.
            Time Complexity: O(log n)
            Space Complexity: O(log n)
        """
        def helper(node: TreeNode | None, value1: int, value2: int):
            if node is None:
                return None
            if value1 < node.data and value2 < node.data:
                return helper(node.left, value1, value2)
            if value1 > node.data and value2 > node.data:
                return helper(node.right, value1, value2)
            return node
        
        return helper(self.root, value1, value2)

    def kth_smallest(self, k: int):
        """
            Finds the kth smallest node in the BST.
            Returns:
                int: The kth smallest node in the BST, None if not found.
            Time Complexity: O(log n)
            Space Complexity: O(log n)
        """
        def helper(node: TreeNode | None, k: int):
            if node is None:
                return None
            left_count = helper(node.left, k)
            if left_count is not None:
                return left_count
            k -= 1
            if k == 0:
                return node.data
            return helper(node.right, k)
        
        return helper(self.root, k)

    def kth_largest(self, k: int):
        """
            Finds the kth largest node in the BST.
            Returns:
                int: The kth largest node in the BST, None if not found.
            Time Complexity: O(log n)
            Space Complexity: O(log n)
        """
        def helper(node: TreeNode | None, k: int):
            if node is None:
                return None
            right_count = helper(node.right, k)
            if right_count is not None:
                return right_count
            k -= 1
            if k == 0:
                return node.data
            return helper(node.left, k)
        
        return helper(self.root, k)

    def build_balanced_bst_from_in_order(self, in_order: list[int]):
        """
            Builds a balanced BST from an in-order traversal.
            Returns:
                TreeNode: The root of the balanced BST, None if in_order is empty.
            Time Complexity: O(n)
            Space Complexity: O(log n)
        """
        def helper(start: int, end: int):
            if start > end:
                return None
            mid = (start + end) // 2
            parent = TreeNode(in_order[mid])
            parent.left = helper(start, mid - 1)
            parent.right = helper(mid + 1, end)
            return parent
        
        self.root = helper(0, len(in_order) - 1)
        return self.root

    # TODO: HW: convert BST into a balanced BST

    def inorder_traversal(self):
        """
            Performs In Order Traversal (DFS) traversal. Explores the tree in the order: left, root, right.
            Returns:
                list[int]: The list of nodes in in order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def helper(node: TreeNode | None):
            if node is None:
                return
            helper(node.left)
            result.append(node.data)
            helper(node.right)
        
        helper(self.root)
        return result

    def two_sum(self, target: int):
        """
            Finds two nodes in the BST that sum to the target.
            Returns:
                tuple[int, int]: The two nodes that sum to the target, None if not found.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = self.inorder_traversal()
        left = 0
        right = len(result) - 1
        while left < right:
            if result[left] + result[right] == target:
                return (result[left], result[right])
            elif result[left] + result[right] < target:
                left += 1
            else:
                right -= 1
        return None

    def convert_bst_into_sorted_dll(self):
        """
            Converts the BST into a DLL.
            Returns:
                TreeNode: The root of the DLL.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        head: TreeNode | None = None
        
        def print_dll():
            temp = head
            while temp is not None:
                print(temp.data, end=" -> ")
                temp = temp.right  # right => next
            print()
        
        def helper(node: TreeNode | None):
            nonlocal head
            if node is None:
                return
            helper(node.right)
            node.right = head  # right => next
            if head is not None:
                head.left = node  # left => prev
            head = node
            helper(node.left)
        
        helper(self.root)
        print_dll()
        return head

    def convert_sorted_dll_into_bst(self, head: TreeNode | None, size: int):
        """
            Converts the sorted DLL into a BST.
            Returns:
                TreeNode: The root of the BST.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def helper(n: int):
            nonlocal head
            if n <= 0 or head is None:
                return None
            # create the left subtree
            left_subtree = helper(n // 2)
            # create the root node
            root = head
            root.left = left_subtree
            head = head.right # move head to the next node
            # create the right subtree
            root.right = helper(n - n // 2 - 1)
            return root

        self.root = helper(size)
        return self.root

    def largest_bst_subtree_size_in_bt(self):
        """
            Finds the largest BST subtree.
            Returns:
                TreeNode: The root of the largest BST subtree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        size_of_largest_bst: int = 0
        
        def helper(node: TreeNode | None):
            nonlocal size_of_largest_bst
            if node is None:
                return TreeNodeMetadata(
                    min=float('inf'),
                    max=float('-inf'),
                    size=0,
                    is_bst=True
                )
            left_subtree_metadata = helper(node.left)
            right_subtree_metadata = helper(node.right)
            if (
                left_subtree_metadata.is_bst and
                right_subtree_metadata.is_bst and
                node.data > left_subtree_metadata.max and
                node.data < right_subtree_metadata.min
            ):
                size_of_largest_bst = max(
                    size_of_largest_bst, 
                    left_subtree_metadata.size + right_subtree_metadata.size + 1
                )
                return TreeNodeMetadata(
                    min=min(node.data, left_subtree_metadata.min),
                    max=max(node.data, right_subtree_metadata.max),
                    size=left_subtree_metadata.size + right_subtree_metadata.size + 1,
                    is_bst=True
                )
            return TreeNodeMetadata(
                min=float('inf'),
                max=float('-inf'),
                size=max(left_subtree_metadata.size, right_subtree_metadata.size),
                is_bst=False
            )
        
        helper(self.root)
        return size_of_largest_bst
