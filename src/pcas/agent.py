"""
InstrumentedAgent — an LLM agent whose tool calls are intercepted
by the PCAS reference monitor before execution.

Every tool call the LLM proposes is:
  1. Added to the dependency graph as a TOOL_CALL node (depends on
     all context nodes the LLM could have seen — the "causal history").
  2. Submitted to the reference monitor for authorization.
  3. Either executed (ALLOW) or returned as a policy feedback message
     to the LLM (DENY), so the model can reason about the refusal.

The accumulated context_node_ids list is the key mechanism: it grows
with every node the LLM observes, so any tool call inherits the full
causal history as its dependencies. This lets the backward slice catch
indirect information leakage (e.g. summarising a tainted document).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import openai
from dotenv import load_dotenv

from .dependency_graph import DependencyGraph, Node, NodeType
from .datalog_engine import Rule
from .reference_monitor import Action, ReferenceMonitor

load_dotenv()

MAX_ITERATIONS = int(os.getenv("PCAS_MAX_ITERATIONS", "10"))
DEFAULT_MODEL = os.getenv("PCAS_MODEL", "gemini-2.0-flash-preview")
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class AgentResult:
    """Return value from InstrumentedAgent.run()."""
    response: str
    context_node_ids: list[str]   # all node IDs the LLM observed
    messages: list[dict]          # full conversation history (sans system prompt)


@dataclass
class InstrumentedAgent:
    """
    An LLM agent instrumented with the PCAS reference monitor.

    Parameters
    ----------
    name            : identity of this agent's principal (e.g. "alice")
    role            : role string used in display (e.g. "VP")
    system_prompt   : initial system message for the LLM
    tool_schemas    : list of OpenAI-format tool definitions
    model           : OpenRouter model identifier
    """
    name: str
    role: str
    system_prompt: str
    tool_schemas: list[dict]
    model: str = field(default_factory=lambda: DEFAULT_MODEL)
    _client: Optional[openai.OpenAI] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self._client = openai.OpenAI(
            base_url=GOOGLE_API_BASE_URL,
            api_key=api_key,
        )

    def run(
        self,
        user_message: str,
        graph: DependencyGraph,
        monitor: ReferenceMonitor,
        policy_rules: list[Rule],
        entity_roles: dict[str, str],
        tool_impls: dict[str, Callable],
        initial_context_node_ids: Optional[list[str]] = None,
        initial_messages: Optional[list[dict]] = None,
    ) -> AgentResult:
        """
        Run the agentic loop for a single user request.

        Parameters
        ----------
        user_message             : the incoming request
        graph                    : shared dependency graph (mutated in place)
        monitor                  : reference monitor instance
        policy_rules             : pre-parsed Datalog rules
        entity_roles             : entity -> role mapping
        tool_impls               : tool_name -> callable implementations
        initial_context_node_ids : node IDs from prior turns to inherit
                                   (used in scenario 3 to inject alice's context)
        initial_messages         : prior conversation messages to prepend
                                   (must not include the system message)

        Returns
        -------
        AgentResult with the final text response, all accumulated node IDs,
        and the full message history for this run.
        """
        # --- Seed context from prior session (for shared-session scenarios) ---
        context_node_ids: list[str] = list(initial_context_node_ids or [])

        # Add the user's message as a node (no causal deps — it's fresh input)
        user_node = Node(
            type=NodeType.MESSAGE,
            entity=self.name,
            content=user_message,
        )
        user_node_id = graph.add_node(user_node, depends_on=[])
        context_node_ids.append(user_node_id)

        # Build the LLM message list
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if initial_messages:
            messages.extend(initial_messages)
        messages.append({"role": "user", "content": user_message})

        # Track only the non-system portion for the returned history
        non_system_messages: list[dict] = list(messages[1:])

        # --- Agentic loop ---
        for iteration in range(MAX_ITERATIONS):
            # Call the LLM
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas if self.tool_schemas else openai.NOT_GIVEN,
                tool_choice="auto" if self.tool_schemas else openai.NOT_GIVEN,
            )
            assistant_msg = response.choices[0].message

            # No tool calls → done
            if not assistant_msg.tool_calls:
                final_text = assistant_msg.content or ""
                # Record the final response as a node
                graph.add_node(
                    Node(type=NodeType.MESSAGE, entity=self.name,
                         content=final_text),
                    depends_on=context_node_ids,
                )
                non_system_messages.append(
                    {"role": "assistant", "content": final_text}
                )
                return AgentResult(
                    response=final_text,
                    context_node_ids=context_node_ids,
                    messages=non_system_messages,
                )

            # Add assistant message to LLM history
            messages.append(assistant_msg)
            non_system_messages.append(assistant_msg)

            # Add an assistant MESSAGE node to the graph
            assistant_node_id = graph.add_node(
                Node(
                    type=NodeType.MESSAGE,
                    entity=self.name,
                    content=assistant_msg.content or "(tool call)",
                ),
                depends_on=context_node_ids,
            )
            context_node_ids.append(assistant_node_id)

            # --- Process each tool call ---
            tool_results_for_llm: list[dict] = []

            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    raw_args = tc.function.arguments
                    tool_args = (
                        raw_args if isinstance(raw_args, dict)
                        else json.loads(raw_args)
                    )
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                # Add a TOOL_CALL node — depends on ALL current context.
                # This is the critical step: context_node_ids captures every
                # node the LLM has seen, so any prior tainted result is
                # included in the causal history of this call.
                call_node_id = graph.add_node(
                    Node(
                        type=NodeType.TOOL_CALL,
                        entity=self.name,
                        content=f"{tool_name}({tool_args})",
                        metadata={"tool": tool_name, "args": tool_args},
                    ),
                    depends_on=list(context_node_ids),  # snapshot
                )

                # Build the Action for the reference monitor
                action = Action(
                    action_id=call_node_id,
                    tool_name=tool_name,
                    args=tool_args,
                    requesting_entity=self.name,
                    depends_on=list(context_node_ids),  # same snapshot
                )

                decision, feedback = monitor.authorize(
                    action, graph, policy_rules, entity_roles
                )

                if decision == "allow":
                    # Execute the tool
                    try:
                        impl = tool_impls.get(tool_name)
                        if impl is None:
                            result_str = f"[Error: unknown tool '{tool_name}']"
                        else:
                            result_str = str(impl(**tool_args))
                    except Exception as exc:
                        result_str = f"[Tool error: {exc}]"

                    # Infer doc_id from args for taint tracking
                    result_meta = {"tool": tool_name}
                    if "doc_id" in tool_args:
                        result_meta["doc_id"] = tool_args["doc_id"]

                    # Add TOOL_RESULT node — depends on the TOOL_CALL node
                    result_node_id = graph.add_node(
                        Node(
                            type=NodeType.TOOL_RESULT,
                            entity="system",
                            content=result_str,
                            metadata=result_meta,
                        ),
                        depends_on=[call_node_id],
                    )
                    context_node_ids.append(call_node_id)
                    context_node_ids.append(result_node_id)

                    tool_results_for_llm.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "content": result_str,
                    })

                else:
                    # Denied — feed policy feedback back to the LLM
                    context_node_ids.append(call_node_id)
                    tool_results_for_llm.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "content": f"[POLICY DENIED] {feedback}",
                    })

            messages.extend(tool_results_for_llm)
            non_system_messages.extend(tool_results_for_llm)

        # Max iterations reached
        final_text = "[Agent stopped: maximum iterations reached]"
        return AgentResult(
            response=final_text,
            context_node_ids=context_node_ids,
            messages=non_system_messages,
        )
