"""
Dependency Graph — models agentic system state as a DAG.

Nodes represent events: messages, tool calls, and tool results.
Directed edges represent causal dependencies (child depends on parent).

The key operation is backward_slice(), which computes the transitive
closure of ancestors for a set of seed nodes. This slice is what the
reference monitor evaluates policy rules against.
"""

from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


class NodeType(Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class Node:
    type: NodeType
    entity: str       # "alice", "bob", "system", "agent"
    content: str      # human-readable payload
    metadata: dict = field(default_factory=dict)
    id: Optional[str] = None
    # Useful metadata keys:
    #   "doc_id"  — for TOOL_RESULT nodes that read a document
    #   "tool"    — for TOOL_CALL and TOOL_RESULT nodes


class DependencyGraph:
    """
    Monotonically-growing DAG of events in an agentic session.

    Nodes are added as actions are proposed and executed. Edges
    capture which prior events caused each new event (causal history).
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        # _deps[node_id] = set of parent node_ids (direct dependencies)
        self._deps: dict[str, set[str]] = {}
        self._counter = 0

    def add_node(self, node: Node, depends_on: Optional[list[str]] = None) -> str:
        """
        Add a node to the graph. Assigns an ID if node.id is None.
        Returns the final node ID.
        """
        if depends_on is None:
            depends_on = []

        if node.id is None:
            self._counter += 1
            node.id = f"n{self._counter}"

        self._nodes[node.id] = node
        self._deps[node.id] = set(depends_on)
        return node.id

    def backward_slice(self, seed_ids: list[str]) -> "DependencyGraph":
        """
        Return the subgraph of all transitive ancestors of seed_ids,
        including the seeds themselves.

        This is the causal history of the seeds — everything that
        could have influenced the events identified by seed_ids.
        """
        visited: set[str] = set()
        worklist: deque[str] = deque(seed_ids)

        while worklist:
            nid = worklist.popleft()
            if nid in visited or nid not in self._nodes:
                continue
            visited.add(nid)
            for parent in self._deps.get(nid, set()):
                if parent not in visited:
                    worklist.append(parent)

        result = DependencyGraph()
        result._counter = self._counter
        for nid in visited:
            if nid in self._nodes:
                result._nodes[nid] = self._nodes[nid]
                result._deps[nid] = {p for p in self._deps[nid] if p in visited}

        return result

    def to_datalog_facts(self) -> list[tuple]:
        """
        Export this graph as a list of EDB fact tuples for the Datalog engine.

        Emitted relations:
          GraphNode(node_id, type, entity)
          DirectDepends(child_id, parent_id)
          IsToolResult(node_id, tool_name, doc_id)   — only if doc_id set
          IsToolCall(node_id, tool_name, entity)
        """
        facts: list[tuple] = []

        for nid, node in self._nodes.items():
            facts.append(("GraphNode", nid, node.type.value, node.entity))

            if node.type == NodeType.TOOL_RESULT:
                tool_name = node.metadata.get("tool", "unknown")
                doc_id = node.metadata.get("doc_id")
                if doc_id:
                    facts.append(("IsToolResult", nid, tool_name, doc_id))

            if node.type == NodeType.TOOL_CALL:
                tool_name = node.metadata.get("tool", "unknown")
                facts.append(("IsToolCall", nid, tool_name, node.entity))

        for nid, parents in self._deps.items():
            if nid not in self._nodes:
                continue
            for parent in parents:
                if parent in self._nodes:
                    facts.append(("DirectDepends", nid, parent))

        return facts

    def get_all_node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def print_audit(self):
        """Print a human-readable summary of the graph for debugging."""
        SEP = "=" * 60
        print(f"\n{SEP}")
        print("  DEPENDENCY GRAPH AUDIT")
        print(SEP)
        for nid, node in self._nodes.items():
            parents = sorted(self._deps.get(nid, set()))
            parent_str = f" <- [{', '.join(parents)}]" if parents else ""
            meta_parts = []
            if node.metadata.get("doc_id"):
                meta_parts.append(f"doc:{node.metadata['doc_id']}")
            if node.metadata.get("tool"):
                meta_parts.append(f"tool:{node.metadata['tool']}")
            meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""
            snippet = node.content[:60].replace("\n", " ")
            print(f"  {nid:>4} ({node.type.value:12}, {node.entity:8}){meta_str}")
            print(f"       \"{snippet}\"")
            if parent_str:
                print(f"       depends on:{parent_str}")
        print(f"{SEP}\n")
