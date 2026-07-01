import re
from dataclasses import dataclass, field
from typing import List, Optional

# ==========================================
# LLVM Pipeline Parsing & Pruning Engine
# ==========================================


CONTAINER_BASE_NAMES = {"module", "function", "cgscc", "loop", "loop-mssa", "devirt"}


ALWAYS_KEEP_PASSES = {"require", "invalidate", "verify"}

@dataclass
class Node:
    raw_name: str
    base_name: str
    children: List["Node"] = field(default_factory=list)
    flag_id: Optional[str] = None

def split_top_level_commas(s: str) -> List[str]:
    items, buf = [], []
    paren, angle = 0, 0
    for c in s:
        if c == "(" and angle == 0: paren += 1; buf.append(c)
        elif c == ")" and angle == 0: paren -= 1; buf.append(c)
        elif c == "<": angle += 1; buf.append(c)
        elif c == ">": angle = max(0, angle - 1); buf.append(c)
        elif c == "," and paren == 0 and angle == 0:
            if item := "".join(buf).strip(): items.append(item)
            buf = []
        else: buf.append(c)
    if tail := "".join(buf).strip(): items.append(tail)
    return items

def parse_name_and_children(item: str) -> Node:
    item = item.strip()
    paren_pos = -1
    angle = 0
    for i, ch in enumerate(item):
        if ch == "<": angle += 1
        elif ch == ">": angle = max(0, angle - 1)
        elif ch == "(" and angle == 0: paren_pos = i; break
    
    if paren_pos == -1:
        raw_name, children_str = item, None
    else:
        raw_name = item[:paren_pos].strip()
        children_str = item[paren_pos+1:].rstrip(")")
    
    m = re.match(r"^([a-zA-Z0-9_.-]+)", raw_name)
    base_name = m.group(1) if m else raw_name
    node = Node(raw_name=raw_name, base_name=base_name)
    if children_str is not None:
        for child in split_top_level_commas(children_str):
            node.children.append(parse_name_and_children(child))
    return node

def build_ast(pipeline: str) -> Node:
    root = Node(raw_name="module", base_name="module")
    for item in split_top_level_commas(pipeline):
        root.children.append(parse_name_and_children(item))
    return root

def assign_flags(root: Node) -> List[str]:
    counters = {}
    flags = []
    
    def next_id(base):
        counters[base] = counters.get(base, 0) + 1
        return f"{base}__{counters[base]}"

    def dfs(node):
        if node.base_name in CONTAINER_BASE_NAMES:
            for ch in node.children: dfs(ch)
        elif node.base_name in ALWAYS_KEEP_PASSES:

            pass
        else:
            fid = next_id(node.base_name)
            node.flag_id = fid
            flags.append(fid)
            for ch in node.children: dfs(ch)

    for ch in root.children: dfs(ch)
    return flags

def prune_ast_in_place(node: Node, selected: set) -> bool:
    is_container = node.base_name in CONTAINER_BASE_NAMES
    node.children = [ch for ch in node.children if prune_ast_in_place(ch, selected)]
    
    if is_container:

        return len(node.children) > 0
        
    if node.base_name in ALWAYS_KEEP_PASSES:

        return True
        
    if node.flag_id is not None:

        return node.flag_id in selected
        
    return False

def serialize_pipeline(node: Node) -> str:
    if not node.children: return node.raw_name
    inner = ",".join(serialize_pipeline(ch) for ch in node.children)
    return f"{node.raw_name}({inner})"

def get_pipeline_string_from_root(root: Node) -> str:
    return ",".join(serialize_pipeline(ch) for ch in root.children)