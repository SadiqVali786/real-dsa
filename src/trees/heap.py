from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
import math


class HeapWithArray:

    def __init__(self):
        self.heap = []

    # ********* Primary Methods *********
    def push(self, data: int):
        self.heap.append(data)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if not self.empty():
            self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
            self._heapify_down(0, len(self.heap) - 1)
            return self.heap.pop()

    def top(self):
        if not self.empty():
            return self.heap[0]

    def build(self, data: list[int]):
        self.heap = data
        for i in range((len(self.heap) >> 1) - 1, -1, -1):  # range(len(self.heap) // 2 - 1, -1, -1)
            self._heapify_down(i, len(self.heap))

    def empty(self):
        return len(self.heap) == 0

    # ********* Secondary Methods *********
    def sort(self):
        """
            Sorts the heap in ascending order.
            Returns:
                list[int]: The sorted heap.
            Time Complexity: O(n log n)
            Space Complexity: O(1)
        """
        size = len(self.heap)
        while size > 1:
            self.heap[0], self.heap[size-1] = self.heap[size-1], self.heap[0]
            self._heapify_down(0, size)
            size -= 1
        return self.heap

    def __len__(self):
        return len(self.heap)


class MaxHeap(HeapWithArray):

    def __init__(self):
        super().__init__()

    # ********* Helper Methods *********
    def _heapify_up(self, child_index: int):
        parent_index = (child_index - 1) >> 1 # (child_index - 1) // 2
        if parent_index >= 0 or self.heap[child_index] > self.heap[parent_index]:
            self.heap[child_index], self.heap[parent_index] = self.heap[parent_index], self.heap[child_index]
            self._heapify_up(parent_index)

    def _heapify_down(self, parent_index: int, size: int):
        left_child_index = 2 * parent_index + 1
        right_child_index = 2 * parent_index + 2
        largest = parent_index
        if left_child_index < size and self.heap[left_child_index] > self.heap[largest]:
            largest = left_child_index
        if right_child_index < size and self.heap[right_child_index] > self.heap[largest]:
            largest = right_child_index
        if largest != parent_index:
            self.heap[parent_index], self.heap[largest] = self.heap[largest], self.heap[parent_index]
            self._heapify_down(largest, size)

    # ********* Problems *********
    def k_th_smallest(self, k: int, arr: list[int]):
        """
            Finds the kth smallest element in the array.
            Returns:
                int: The kth smallest element in the array.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        self.build(arr[:k])
        for i in range(k, len(arr)):
            if arr[i] < self.top():
                self.pop()
                self.push(arr[i])
        return self.top()


@dataclass
class ListNode:
    val: int
    next: 'ListNode' | None = None


class MinHeap(HeapWithArray):

    def __init__(self):
        super().__init__()

    # ********* Helper Methods *********
    def _heapify_up(self, child_index: int):
        parent_index = (child_index - 1) >> 1 # (child_index - 1) // 2
        if parent_index >= 0 or self.heap[child_index] < self.heap[parent_index]:
            self.heap[child_index], self.heap[parent_index] = self.heap[parent_index], self.heap[child_index]
            self._heapify_up(parent_index)

    def _heapify_down(self, parent_index: int, size: int):
        left_child_index = 2 * parent_index + 1
        right_child_index = 2 * parent_index + 2
        smallest = parent_index
        if left_child_index < size and self.heap[left_child_index] < self.heap[smallest]:
            smallest = left_child_index
        if right_child_index < size and self.heap[right_child_index] < self.heap[smallest]:
            smallest = right_child_index
        if smallest != parent_index:
            self.heap[parent_index], self.heap[smallest] = self.heap[smallest], self.heap[parent_index]
            self._heapify_down(smallest, size)

    # ********* Problems *********
    def k_th_largest(self, k: int, arr: list[int]):
        """
            Finds the kth largest element in the array.
            Returns:
                int: The kth largest element in the array.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        self.build(arr[:k])
        for i in range(k, len(arr)):
            if arr[i] > self.top():
                self.pop()
                self.push(arr[i])
        return self.top()

    def merge_k_sorted_arrays(self, arrays: list[list[int]]):
        """
            Merges k sorted arrays into a single sorted array.
            Returns:
                list[int]: The merged sorted array.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        # Initializing the heap
        k = len(arrays)
        heap: list[tuple[int, int, int]] = []
        for i in range(k):
            heap.append((arrays[i][0], i, 0))
        heapq.heapify(heap) # TC: O(k log k)

        # Merging the arrays
        result: list[int] = []
        while heap: # TC: O(n log k), where n is the total number of elements in all arrays
            val, i, j = heapq.heappop(heap) # TC: O(log k)
            result.append(val)
            if j < len(arrays[i]) - 1:
                # TC: O(log k) per push operation. 
                # This is MinHeap, to make it MaxHeap, we need to push -val, i, j
                heapq.heappush(heap, (arrays[i][j + 1], i, j + 1)) # TC: O(log k)
        
        return result

    def merge_k_sorted_sll(self, heads: list[ListNode]):
        """
            Merges k sorted singly linked lists into a single sorted linked list.
            Returns:
                ListNode: The merged sorted linked list.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        heap: list[tuple[int, ListNode]] = []
        k = len(heads)
        for i in range(k):
            heap.append((heads[i].val, heads[i]))
        heapq.heapify(heap)

        ans_head = ListNode()
        ans_tail = ans_head
        while heap:
            curr_val, curr_node = heapq.heappop(heap)
            ans_tail.next = curr_node
            ans_tail = curr_node.next
            if curr_node.next is not None:
                heapq.heappush(heap, (curr_node.next.val, curr_node.next))
        
        ans_head = ans_head.next
        return ans_head, ans_tail

    def smallest_range(self, arrays: list[list[int]]):
        """
            Finds the smallest range that includes at least one element from each array.
            Returns:
                tuple[int, int]: The smallest range.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        k = len(arrays)
        heap: list[tuple[int, int, int]] = []  # value, array index, element index
        for i in range(k):
            heap.append((arrays[i][0], i, 0))
        heapq.heapify(heap)

        min_range = math.inf
        max_val = max(heap, key=lambda x: x[0])[0]
        
        while True:  # len(heap) == k
            val, i, j = heapq.heappop(heap)
            if max_val - val < min_range:
                min_range = max_val - val
                min_range_start = val
                min_range_end = max_val
            if j < len(arrays[i]) - 1:
                heapq.heappush(heap, (arrays[i][j + 1], i, j + 1))
                max_val = max(max_val, arrays[i][j + 1])
            else:
                break
        return min_range_start, min_range_end


@dataclass
class HeapTreeNode:
    val: int
    parent: 'HeapTreeNode' | None = None
    left: 'HeapTreeNode' | None = None
    right: 'HeapTreeNode' | None = None


class MaxHeapWithBinaryTree:

    def __init__(self):
        self.root: HeapTreeNode | None = None

    # ********* Primary Methods *********
    def push(self, val: int):
        if self.root is None:
            self.root = HeapTreeNode(val)
        else:
            queue = deque([self.root])
            while queue:
                current_node = queue.popleft()
                # Try to insert left
                if current_node.left is None:
                    current_node.left = HeapTreeNode(val)
                    current_node.left.parent = current_node
                    self._heapify_up(current_node.left)
                    break
                
                # Try to insert right
                if current_node.right is None:
                    current_node.right = HeapTreeNode(val)
                    current_node.right.parent = current_node
                    self._heapify_up(current_node.right)
                    break
                
                # Continue to next level
                queue.append(current_node.left)
                queue.append(current_node.right)

    def _heapify_up(self, node: HeapTreeNode):
        # Bubble up while value is greater than parent's value
        while node.parent is not None and node.val > node.parent.val:
            node.val, node.parent.val = node.parent.val, node.val
            node = node.parent

    def pop(self):
        if self.root is not None:
            # 1. Find the last node in the tree (Level Order Traversal)
            queue = deque([self.root])
            while queue:
                current_node = queue.popleft()
                if current_node.left is not None:
                    queue.append(current_node.left)
                if current_node.right is not None:
                    queue.append(current_node.right)
            
            max_val = self.root.val
            
            # 2. Edge Case: Only one node (root only)
            if current_node == self.root:
                self.root = None
                return max_val
            
            # 3. Move last node's value to root
            self.root.val = current_node.val

            # 4. Remove the last node
            if current_node.parent.left == current_node:
                current_node.parent.left = None
            else:
                current_node.parent.right = None
            
            # 5. Restore heap property downwards
            self._heapify_down(self.root)
            return max_val

    def _heapify_down(self, node: HeapTreeNode | None):
        # Bubble down while node has left child
        while node.left is not None and node.right is not None:
            largest = node
            # Compare left
            if node.left.val > largest.val:
                largest = node.left
            # Compare right
            if node.right.val > largest.val:
                largest = node.right
            # Swap if child is larger
            if largest != node:
                node.val, largest.val = largest.val, node.val
                node = largest
            else:
                break

    def top(self):
        if self.root is not None:
            return self.root.val

    def build(self, vals: list[int]):
        for val in vals:
            self.push(val)

    def empty(self):
        return self.root is None

    # ********* Problems *********
    def is_heap(self):
        """
            Checks if the heap is a valid heap.
            Returns:
                bool: True if the heap is a valid heap, False otherwise.
            Time Complexity: O(n)
            Space Complexity: O(log n)
        """
        def helper(node: HeapTreeNode | None):
            if node is None or (node.left is None and node.right is None):
                return True
            if not helper(node.left):
                return False
            if not helper(node.right):
                return False
            return node.left.val <= node.val and node.right.val <= node.val
        
        return helper(self.root)

    def min_stone_sum(self, stones: list[int], k: int):
        """
            Finds the minimum sum of stones after k operations.
            Returns:
                int: The minimum sum of stones after k operations.
            Time Complexity: O(n + k log n)
            Space Complexity: O(k)
        """
        heap = stones[:]
        heapq.heapify(heap)
        while k > 0:
            max_val = heapq.heappop(heap)
            heapq.heappush(heap, max_val // 2)
            k -= 1
        return sum(heap)

    # https://leetcode.com/problems/reorganize-string/description/?ref=read.learnyard.com
    # Minimum cost to cut the ropes which uses Heaps not DP, also uses this approach
    def reorganize_string(self, s: str):
        """
            Reorganizes the string to avoid adjacent characters.
            Returns:
                str: The reorganized string.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        count: defaultdict[str, int] = defaultdict(int)
        for char in s:
            count[char] += 1
        max_heap: list[tuple[int, str]] = [(-count[char], char) for char in count]
        heapq.heapify(max_heap)
        
        ans: list[str] = []
        while len(max_heap) > 1:
            count1, char1 = heapq.heappop(max_heap)
            count2, char2 = heapq.heappop(max_heap)
            count1, count2 = -count1, -count2
            ans.append(char1)
            ans.append(char2)
            if count1 != 0:
                heapq.heappush(max_heap, (-(count1 - 1), char1))
            if count2 != 0:
                heapq.heappush(max_heap, (-(count2 - 1), char2))
        
        if len(max_heap) == 1:
            count1, char1 = heapq.heappop(max_heap)
            count1 = -count1
            if count1 > 1:
                return ''
            ans.append(char1)
        
        return ''.join(ans)

    # https://leetcode.com/problems/longest-happy-string/description/
    def longest_happy_string(self, a: int, b: int, c: int):
        """
            Finds the longest happy string.
            Returns:
                str: The longest happy string.
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        max_heap: list[tuple[int, str]] = [
            (-count, char) for count, char in [(a, 'a'), (b, 'b'), (c, 'c')] if count > 0
        ]
        heapq.heapify(max_heap)

        ans: list[str] = []
        while len(max_heap) > 1:
            count1, char1 = heapq.heappop(max_heap)
            count2, char2 = heapq.heappop(max_heap)
            count1, count2 = -count1, -count2
            
            if count1 >= 2:
                ans.append(char1 * 2)
                count1 -= 2
            else:
                ans.append(char1)
                count1 -= 1
            if count1 > 0:
                heapq.heappush(max_heap, (-count1, char1))
            
            if count2 >= 2 and count2 >= count1:  # MISTAKE: 2nd condition bhul jaaoge
                ans.append(char2 * 2)
                count2 -= 2
            else:
                ans.append(char2)
                count2 -= 1
            if count2 > 0:
                heapq.heappush(max_heap, (-count2, char2))
        
        if len(max_heap) == 1:
            count1, char1 = heapq.heappop(max_heap)
            count1 = -count1
            if count1 > 2 or ans[-1] == char1:
                return ''
            if count1 == 2:
                ans.append(char1 * 2)
            else:
                ans.append(char1)
        
        return ''.join(ans)

    def median_of_stream(self, nums: list[int]):
        """
            Finds the median of a stream of numbers.
            Returns:
                list[float]: The median of the stream.
            Time Complexity: O(n log n)
            Space Complexity: O(n)

            1. max_heap => median => min_heap ===> SORTED IN ASCENDING ORDER
            2. values < median will be in the left of median in max_heap
            3. values > median will be in the right of median in min_heap
            4. Ex: 3, 2, 1, 0, -1 => 4 => 5, 6, 7, 8, 9
        """
        min_heap: list[int] = []
        max_heap: list[int] = []
        medians: list[float] = [float('-inf')]
        
        for num in nums:
            if len(max_heap) == len(min_heap):
                if num > medians[-1]:
                    heapq.heappush(min_heap, num)
                    medians.append(min_heap[0])
                else:
                    heapq.heappush(max_heap, -num)
                    medians.append(-max_heap[0])
            elif len(max_heap) > len(min_heap):
                if num > medians[-1]:
                    heapq.heappush(min_heap, num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
                else:
                    heapq.heappush(min_heap, -heapq.heappop(max_heap))
                    heapq.heappush(max_heap, -num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
            elif len(max_heap) < len(min_heap):
                if num > medians[-1]:
                    heapq.heappush(max_heap, -heapq.heappop(min_heap))
                    heapq.heappush(min_heap, num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
                else:
                    heapq.heappush(max_heap, -num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
        
        return medians[1:]

    # TODO: Merge 2 Heaps
    # TODO: Is it CBT?
