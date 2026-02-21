"""
Reference Monitor — the enforcement heart of PCAS.

For every proposed agent action, the monitor:
  1. Computes the backward slice of the dependency graph rooted at
     the action's causal dependencies.
  2. Converts the slice into Datalog EDB facts.
  3. Loads the policy rules into a fresh DatalogEngine.
  4. Evaluates the rules and queries for Allowed / Denied decisions.
  5. Returns (decision, feedback_message) — fail-closed (default deny).

This provides deterministic enforcement independent of LLM reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional

from .dependency_graph import DependencyGraph
from .datalog_engine import DatalogEngine, Rule


@dataclass
class Action:
    """A proposed agent action awaiting authorization."""
    action_id: str          # Node ID assigned in the dependency graph
    tool_name: str          # Name of the tool to call
    args: dict              # Keyword arguments for the tool
    requesting_entity: str  # Identity of the agent/principal making the request
    depends_on: list[str] = field(default_factory=list)
    # depends_on: all dependency graph node IDs the LLM had in its context
    # window when it generated this action (the causal history seeds).


class ReferenceMonitor:
    """
    Intercepts and authorizes agent tool calls via policy evaluation.

    The monitor is stateless — all state lives in the DependencyGraph
    and the policy rules passed per call.
    """

    def authorize(
        self,
        action: Action,
        graph: DependencyGraph,
        policy_rules: list[Rule],
        entity_roles: dict[str, str],
        verbose: bool = True,
    ) -> tuple[str, str]:
        """
        Evaluate whether the action is permitted under the policy.

        Parameters
        ----------
        action        : the proposed tool call
        graph         : the full shared dependency graph
        policy_rules  : pre-parsed Datalog rules (from the .dl policy file)
        entity_roles  : mapping of entity name -> role (e.g. {"alice": "vp"})
        verbose       : if True, print evaluation details to stdout

        Returns
        -------
        (decision, feedback)
          decision  : "allow" or "deny"
          feedback  : human-readable explanation (returned to the agent on deny)
        """
        if verbose:
            print(
                f"\n  [Monitor] Evaluating: {action.requesting_entity} -> "
                f"{action.tool_name}({action.args})"
            )

        # --- Step 1: Backward slice ---
        if action.depends_on:
            slice_graph = graph.backward_slice(action.depends_on)
        else:
            slice_graph = DependencyGraph()

        if verbose:
            slice_ids = slice_graph.get_all_node_ids()
            print(f"  [Monitor] Backward slice: {slice_ids}")

        # --- Step 2: Build fresh Datalog engine ---
        engine = DatalogEngine()

        # --- Step 3: Load entity roles ---
        for entity, role in entity_roles.items():
            engine.add_fact("EntityRole", entity, role)

        # --- Step 4: Load graph facts from slice ---
        for fact in slice_graph.to_datalog_facts():
            engine.add_fact(fact[0], *fact[1:])

        # --- Step 5: Load action facts ---
        engine.add_fact("PendingAction", action.action_id, action.tool_name,
                        action.requesting_entity)
        for k, v in action.args.items():
            engine.add_fact("ActionArg", action.action_id, k, str(v))

        # Add the action's direct dependency edges to the engine so the
        # transitive Depends rules can reach the slice's tainted nodes.
        for parent_id in action.depends_on:
            engine.add_fact("DirectDepends", action.action_id, parent_id)

        # --- Step 6: Load policy rules ---
        engine.rules.extend(policy_rules)

        # --- Step 7: Evaluate ---
        engine.evaluate()

        # --- Step 8: Query decision (deny overrides allow) ---
        denied = engine.query("Denied", action.requesting_entity, None, None)
        if denied:
            reason = denied[0][2].replace("_", " ") if len(denied[0]) > 2 else "policy violation"
            if verbose:
                print(f"  [Monitor] DENY — {reason}")
            return ("deny", f"Access denied: {reason}")

        allowed = engine.query("Allowed", action.requesting_entity, None, None)
        if allowed:
            if verbose:
                print(f"  [Monitor] ALLOW")
            return ("allow", "Action permitted by policy")

        # Default deny (fail-closed)
        if verbose:
            print(f"  [Monitor] DENY — no policy grants this action (default deny)")
        return ("deny", "No policy rule grants this action (default deny)")
