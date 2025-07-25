# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from collections import defaultdict
from typing import List


class TreeNode():
    def __init__(self):
        self.child = defaultdict(TreeNode)

class Trie:

    def __init__(self, bos, eos):
        self.root = TreeNode()
        self.bos = bos
        self.eos = eos

    def insert(self, word: List[int]):
        cur = self.root
        for c in word:
            cur = cur.child[c]

    def get_child_index(self, cur: TreeNode) -> List[int]:
        if len(cur.child)==0:
            return [self.eos]
        return list(cur.child.keys())
    
    def get_next_node(self, cur: TreeNode, w: int) -> TreeNode:
        if len(cur.child)==0:
            return cur
        return cur.child[w]