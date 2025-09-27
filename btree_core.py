from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SensorRecord:
    timestamp: int
    object_id: int
    object_type: str
    position: Tuple[float, float, float]
    velocity: float
    distance_to_our_car: float
    lane_id: int = 1


class BTreeNode:
    def __init__(self, min_degree: int, leaf: bool) -> None:
        self.min_degree = min_degree
        self.leaf = leaf
        self.keys: List[int] = []
        self.values: List[SensorRecord] = []
        self.children: List["BTreeNode"] = []


class BTree:
    def __init__(self, min_degree: int = 2) -> None:
        if min_degree < 2:
            raise ValueError("min_degree must be >= 2")
        self.t = min_degree
        self.root = BTreeNode(min_degree=self.t, leaf=True)

    def split_child(self, parent: BTreeNode, index: int) -> None:
        t = self.t
        node_to_split = parent.children[index]
        new_node = BTreeNode(min_degree=t, leaf=node_to_split.leaf)
        mid_key = node_to_split.keys[t - 1]
        mid_val = node_to_split.values[t - 1]
        new_node.keys = node_to_split.keys[t:]
        new_node.values = node_to_split.values[t:]
        if not node_to_split.leaf:
            new_node.children = node_to_split.children[t:]
        node_to_split.keys = node_to_split.keys[: t - 1]
        node_to_split.values = node_to_split.values[: t - 1]
        if not node_to_split.leaf:
            node_to_split.children = node_to_split.children[:t]
        parent.keys.insert(index, mid_key)
        parent.values.insert(index, mid_val)
        parent.children.insert(index + 1, new_node)

    def insert_non_full(self, node: BTreeNode, key: int, value: SensorRecord) -> None:
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(0)
            node.values.append(value)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.t - 1):
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self.insert_non_full(node.children[i], key, value)

    def insert(self, key: int, value: SensorRecord) -> None:
        root = self.root
        if len(root.keys) == (2 * self.t - 1):
            new_root = BTreeNode(min_degree=self.t, leaf=False)
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.root = new_root
            self.insert_non_full(new_root, key, value)
        else:
            self.insert_non_full(root, key, value)

    def get_latest(self) -> Optional[Tuple[int, SensorRecord]]:
        node = self.root
        if not node.keys:
            return None
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]

    def search(self, key: int) -> Optional[SensorRecord]:
        """Search for a key in the B-tree and return the corresponding value."""
        def search_node(node: BTreeNode, k: int) -> Optional[SensorRecord]:
            i = 0
            while i < len(node.keys) and k > node.keys[i]:
                i += 1
            
            if i < len(node.keys) and k == node.keys[i]:
                return node.values[i]
            
            if node.leaf:
                return None
            
            return search_node(node.children[i], k)
        
        return search_node(self.root, key)

    def inorder_keys(self) -> List[int]:
        result: List[int] = []

        def walk(n: BTreeNode) -> None:
            if n.leaf:
                result.extend(n.keys)
                return
            for i, k in enumerate(n.keys):
                walk(n.children[i])
                result.append(k)
            walk(n.children[-1])

        if self.root.keys:
            walk(self.root)
        return result


class BTreeManager:
    """Facade for B-tree with simple history for plotting and queries."""

    def __init__(self) -> None:
        self._btree = BTree(min_degree=2)
        self._history: List[Tuple[int, float]] = []

    def insert_record(self, record: SensorRecord) -> None:
        self._btree.insert(record.timestamp, record)
        # maintain distance series
        self._history.append((record.timestamp, record.distance_to_our_car))

    def insert_with_key(self, key: int, record: SensorRecord) -> None:
        self._btree.insert(key, record)

    def get_latest(self) -> Optional[Tuple[int, SensorRecord]]:
        return self._btree.get_latest()

    def get_history(self) -> List[Tuple[int, float]]:
        return list(self._history)

    def search(self, key: int) -> Optional[SensorRecord]:
        """Search for a key and return the corresponding sensor record."""
        return self._btree.search(key)

    def inorder_keys(self) -> List[int]:
        return self._btree.inorder_keys()


