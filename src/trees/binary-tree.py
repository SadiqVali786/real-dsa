from collections import deque
from dataclasses import dataclass


@dataclass
class TreeNode:
    data: int
    left: 'TreeNode' | None = None
    right: 'TreeNode' | None = None


class BinaryTree:

    def __init__(self):
        self.root: TreeNode | None = None

    def insert(self, parent: TreeNode | None, data: int):
        """
            Inserts a new node into the tree.
            Args:
                parent: The parent node to insert the new node into.
                data: The data of the new node.
            Returns:
                TreeNode: The parent node.
            Time Complexity: O(1)
            Space Complexity: O(1)
        """
        if parent is None:
            return TreeNode(data)
        elif parent.left is None:
            parent.left = TreeNode(data)
        else:
            parent.right = TreeNode(data)
        return parent

    def build_tree(self, nodes: list[int]):
        """
            Builds a tree from a list of nodes.
            Args:
                nodes: The list of nodes to build the tree from.
            Returns:
                TreeNode: The root of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if not nodes:
            return None
        root = TreeNode(nodes[0])
        self.root = root
        for node in nodes[1:]:
            self.insert(root, node)
        return root

    def level_order_traversal(self):
        """
            Performs Level Order Traversal (BFS) traversal. Explores the tree level by level from left to right.
            Returns:
                list[int]: The list of nodes in level order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        queue = deque([self.root])
        result: list[int] = []
        while queue:
            current_node = queue.popleft()
            result.append(current_node.data)
            if current_node.left is not None:
                queue.append(current_node.left)
            if current_node.right is not None:
                queue.append(current_node.right)
        return result

    def level_order_traversal_v2(self):
        """
            Performs Level Order Traversal (BFS) traversal. Explores the tree level by level from left to right.
            Returns:
                list[int]: The list of nodes in level order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        queue = deque([self.root, None])
        result: list[int] = []
        while queue:
            current_node = queue.popleft()
            if current_node is None:
                if not queue:
                    queue.append(None)
            else:
                result.append(current_node.data)
                if current_node.left is not None:
                    queue.append(current_node.left)
                if current_node.right is not None:
                    queue.append(current_node.right)
        return result

    def in_order_traversal(self):  # LNR
        """
            Performs In Order Traversal (DFS) traversal. Explores the tree in the order: left, root, right.
            Returns:
                list[int]: The list of nodes in in order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def in_order_traversal_helper(node: TreeNode | None):
            if node is None:
                return
            in_order_traversal_helper(node.left)
            result.append(node.data)
            in_order_traversal_helper(node.right)

        in_order_traversal_helper(self.root)
        return result

    def pre_order_traversal(self):  # NLR
        """
            Performs Pre Order Traversal (DFS) traversal. Explores the tree in the order: root, left, right.
            Returns:
                list[int]: The list of nodes in pre order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def pre_order_traversal_helper(node: TreeNode | None):
            if node is None:
                return
            result.append(node.data)
            pre_order_traversal_helper(node.left)
            pre_order_traversal_helper(node.right)

        pre_order_traversal_helper(self.root)
        return result

    def post_order_traversal(self):  # LRN
        """
            Performs Post Order Traversal (DFS) traversal. Explores the tree in the order: left, right, root.
            Returns:
                list[int]: The list of nodes in post order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def post_order_traversal_helper(node: TreeNode | None):
            if node is None:
                return
            post_order_traversal_helper(node.left)
            post_order_traversal_helper(node.right)
            result.append(node.data)

        post_order_traversal_helper(self.root)
        return result

    def height_of_tree(self): # max depth of the tree
        """
            Calculates the height of the tree.
            Returns:
                int: The height of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def height_of_tree_helper(node: TreeNode | None):
            if node is None:
                return 0
            left_height = height_of_tree_helper(node.left)
            right_height = height_of_tree_helper(node.right)
            return max(left_height, right_height) + 1

        return height_of_tree_helper(self.root)

    def diameter_of_tree(self): 
        """
            Calculates the diameter of the tree.
            Returns:
                int: The diameter of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def diameter_of_tree_helper(node: TreeNode | None):
            if node is None:
                return 0
            # without current node
            left_diameter = diameter_of_tree_helper(node.left)
            right_diameter = diameter_of_tree_helper(node.right)
            # with current node
            left_height = self.height_of_tree_helper(node.left)
            right_height = self.height_of_tree_helper(node.right)
            current_diameter = left_height + right_height
            # return the maximum of the three
            return max(left_diameter, right_diameter, current_diameter)
        
        return diameter_of_tree_helper(self.root)

    def is_tree_balanced(self):
        """
            Checks if the tree is balanced.
            Returns:
                bool: True if the tree is balanced, False otherwise.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def is_tree_balanced_helper(node: TreeNode | None):
            if node is None:
                return True
            left_balanced = is_tree_balanced_helper(node.left)
            right_balanced = is_tree_balanced_helper(node.right)
            if not left_balanced or not right_balanced:
                return False
            left_height = self.height_of_tree_helper(node.left)
            right_height = self.height_of_tree_helper(node.right)
            return abs(left_height - right_height) <= 1
        
        return is_tree_balanced_helper(self.root)

    def convert_into_sum_tree(self):
        """
            Converts the tree into a sum tree.
            Returns:
                TreeNode: The root of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def convert_into_sum_tree_helper(node: TreeNode | None):
            if node is None:
                return 0
            left_sum = convert_into_sum_tree_helper(node.left)
            right_sum = convert_into_sum_tree_helper(node.right)
            node.data += left_sum + right_sum
            return node.data
        
        convert_into_sum_tree_helper(self.root)
        return self.root

    def lowest_common_ancestor(self, value1: int, value2: int):
        """
            Finds the lowest common ancestor of two nodes.
            Returns:
                TreeNode: The lowest common ancestor of the two nodes.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def lowest_common_ancestor_helper(node: TreeNode | None, value1: int, value2: int):
            if node is None:
                return None
            if node.data == value1 or node.data == value2:
                return node
            left_ancestor = lowest_common_ancestor_helper(node.left, value1, value2)
            right_ancestor = lowest_common_ancestor_helper(node.right, value1, value2)
            if left_ancestor is not None and right_ancestor is not None:
                return node
            return left_ancestor if left_ancestor is not None else right_ancestor
        
        return lowest_common_ancestor_helper(self.root, value1, value2)

    def kth_ancestor(self, k: int, value: int):
        """
            Finds the kth ancestor of a node.
            Returns:
                int: The kth ancestor of the node.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        ans: int | None = None
        def helper(k: int, value: int, node: TreeNode | None):
            nonlocal ans
            if node is None:
                return False
            if node.data == value:
                return True
            if helper(k, value, node.left) or helper(k, value, node.right):
                k -= 1
                if k == 0:
                    ans = node.data
                return True
            return False
        
        helper(k, value, self.root)
        return ans

    def path_sum(self, target_sum: int):
        """
            Checks if there is a path in the tree that sums to the target sum.
            Returns:
                bool: True if there is a path in the tree that sums to the target sum, False otherwise.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        import copy as cp
        curr_sum = 0
        curr_path: list[int] = []
        ans: list[list[int]] = []

        def helper(node: TreeNode | None, target_sum: int):
            nonlocal curr_sum
            if node is None:
                return
            curr_path.append(node.data)
            curr_sum += node.data
            if node.left is None and node.right is None:
                if curr_sum == target_sum:
                    ans.append(cp.deepcopy(curr_path))
                # backtrack
                curr_path.pop()
                curr_sum -= node.data
                return
            helper(node.left, target_sum)
            helper(node.right, target_sum)
            # backtrack
            curr_path.pop()
            curr_sum -= node.data

        helper(self.root, target_sum)
        return ans

    def build_tree_from_pre_order_and_in_order(self, pre_order: list[int], in_order: list[int]):
        """
            Constructs a tree from a pre-order and in-order traversal.
            Returns:
                TreeNode: The root of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        size = len(pre_order)
        pre_index = 0
        def helper(in_start: int, in_end: int):
            nonlocal pre_index
            # base case
            if in_start > in_end or pre_index > size - 1:
                return None
            #  step 1: get the root value
            root_value = pre_order[pre_index]
            root = TreeNode(root_value)
            pre_index += 1
            # step 2: get the root index in in_order
            root_index = in_order.index(root_value) # use hash map to optimize lookup
            # step 3: build the left subtree
            root.left = helper(in_start, root_index - 1)
            # step 4: build the right subtree
            root.right = helper(root_index + 1, in_end)
            return root
        
        return helper(0, size - 1)

    def build_tree_from_post_order_and_in_order(self, post_order: list[int], in_order: list[int]):
        """
            Constructs a tree from a post-order and in-order traversal.
            Returns:
                TreeNode: The root of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        size = len(post_order)
        post_index = size - 1
        def helper(in_start: int, in_end: int):
            nonlocal post_index
            # base case
            if in_start > in_end or post_index < 0:
                return None
            # step 1: get the root value
            root_value = post_order[post_index]
            root = TreeNode(root_value)
            post_index -= 1
            # step 2: get the root index in in_order
            root_index = in_order.index(root_value) # use hash map to optimize lookup
            # step 3: build the right subtree
            root.right = helper(root_index + 1, in_end)
            # step 4: build the left subtree
            root.left = helper(in_start, root_index - 1)
            return root
        
        return helper(0, size - 1) 

    def top_view(self):
        """
            Constructs the top view of the tree.
            Returns:
                list[int]: The top view of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        queue: deque[tuple[TreeNode, int]] = deque([(self.root, 0)])
        result: dict[int, int] = {}
        while queue:
            current_node, horizontal_distance = queue.popleft()
            if horizontal_distance not in result:
                result[horizontal_distance] = current_node.data
            if current_node.left is not None:
                queue.append((current_node.left, horizontal_distance - 1))
            if current_node.right is not None:
                queue.append((current_node.right, horizontal_distance + 1))
        return list(result.values())

    def bottom_view(self):
        """
            Constructs the bottom view of the tree.
            Returns:
                list[int]: The bottom view of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        queue: deque[tuple[TreeNode, int]] = deque([(self.root, 0)])
        result: dict[int, int] = {}
        while queue:
            current_node, horizontal_distance = queue.popleft()
            result[horizontal_distance] = current_node.data
            # if 5, 7 at same level => interchnage the if blocks below to get 5 instead of 7
            if current_node.left is not None:
                queue.append((current_node.left, horizontal_distance - 1))
            if current_node.right is not None:
                queue.append((current_node.right, horizontal_distance + 1))
        return list(result.values())

    def left_view(self):
        """
            Constructs the left view of the tree.
            Returns:
                list[int]: The left view of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        ans: list[int] = []
        def helper(node: TreeNode | None, level: int):
            if node is None:
                return
            if level == len(ans):
                ans.append(node.data)
            helper(node.left, level + 1)
            helper(node.right, level + 1)
        
        helper(self.root, 0)
        return ans

    def right_view(self):
        """
            Constructs the right view of the tree.
            Returns:
                list[int]: The right view of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        ans: list[int] = []
        def helper(node: TreeNode | None, level: int):
            if node is None:
                return
            if level == len(ans):
                ans.append(node.data)
            helper(node.right, level + 1)
            helper(node.left, level + 1)
        
        helper(self.root, 0)
        return ans

    def boundary_traversal(self):
        """
            Constructs the boundary traversal of the tree.
            Returns:
                list[int]: The boundary traversal of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        ans: list[int] = [self.root.data]
        
        def left_boundary(node: TreeNode | None):
            if node is None or (node.left is None and node.right is None):
                return
            ans.append(node.data)
            if node.left is not None:
                left_boundary(node.left)
            else:
                left_boundary(node.right)
        
        def leaf_boundary(node: TreeNode | None):
            if node is None:
                return
            if node.left is None and node.right is None:
                ans.append(node.data)
            leaf_boundary(node.left)
            leaf_boundary(node.right)
        
        def right_boundary(node: TreeNode | None):
            if node is None or (node.left is None and node.right is None):
                return
            if node.right is not None:
                right_boundary(node.right)
            else:
                right_boundary(node.left)
            ans.append(node.data)
        
        left_boundary(self.root.left)
        leaf_boundary(self.root)
        right_boundary(self.root.right)
        return ans
