"""
Datalog Engine — pure Python stratified bottom-up evaluator.

Implements a subset of stratified Datalog with negation-as-failure,
sufficient for PCAS policy enforcement. No external libraries required.

Supported syntax:
  Facts:   Relation(const1, const2).
  Rules:   Head(X, Y) :- Body1(X, Z), not Body2(Z, Y).

Variables start with an uppercase letter. Constants are lowercase (or
any string that does not start with uppercase). The special token _
is an anonymous variable (each occurrence is unique).

Evaluation is stratified: negated predicates must be fully computed
before the rules that negate them run. Within each stratum, a naive
bottom-up fixpoint is used until no new facts are derived.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# AST types
# ---------------------------------------------------------------------------

@dataclass
class Literal:
    """A predicate application: Relation(arg1, arg2, ...)"""
    relation: str
    args: list[str]


@dataclass
class BodyLiteral:
    """A possibly-negated literal in a rule body."""
    negated: bool
    literal: Literal


@dataclass
class Rule:
    """A Datalog rule or fact.  head :- body (body may be empty for facts)."""
    head: Literal
    body: list[BodyLiteral]

    @property
    def is_fact(self) -> bool:
        return len(self.body) == 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _is_variable(token: str) -> bool:
    """Variables start with uppercase or are anonymous (_anon_N)."""
    if not token:
        return False
    return token[0].isupper() or token.startswith("_anon_")


def _split_top_level(s: str, delim: str = ",") -> list[str]:
    """
    Split s on delim, but only when not inside parentheses.
    Used to split rule bodies and argument lists.
    """
    result: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
            current.append(c)
        elif c == ")":
            depth -= 1
            current.append(c)
        elif c == delim and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        result.append("".join(current).strip())
    return [r for r in result if r]


def _parse_literal_str(s: str, anon_counter: list[int]) -> Literal:
    """
    Parse a string like 'Relation(arg1, arg2)' into a Literal.
    Anonymous variables _ are rewritten to unique _anon_N names.
    """
    s = s.strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", s, re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse literal: {s!r}")

    relation = m.group(1)
    args_str = m.group(2)
    raw_args = _split_top_level(args_str)

    args: list[str] = []
    for arg in raw_args:
        arg = arg.strip()
        if arg == "_":
            anon_counter[0] += 1
            args.append(f"_anon_{anon_counter[0]}")
        else:
            args.append(arg)

    return Literal(relation=relation, args=args)


def _parse_single_rule(rule_str: str) -> Optional[Rule]:
    """
    Parse one rule or fact (without trailing period).
    Returns None for empty strings.
    """
    rule_str = rule_str.strip()
    if not rule_str:
        return None

    anon_counter = [0]

    if ":-" in rule_str:
        idx = rule_str.index(":-")
        head_str = rule_str[:idx].strip()
        body_str = rule_str[idx + 2:].strip()

        head = _parse_literal_str(head_str, anon_counter)

        body: list[BodyLiteral] = []
        for part in _split_top_level(body_str):
            part = part.strip()
            if not part:
                continue
            negated = False
            if part.startswith("not ") or part.startswith("not\t"):
                negated = True
                part = part[3:].strip()
            lit = _parse_literal_str(part, anon_counter)
            body.append(BodyLiteral(negated=negated, literal=lit))

        return Rule(head=head, body=body)
    else:
        head = _parse_literal_str(rule_str, anon_counter)
        return Rule(head=head, body=[])


def parse_rules(source: str) -> list[Rule]:
    """
    Parse all Datalog rules and facts from a source string.
    Strips % and // comments. Splits on '.' as the rule terminator.
    """
    cleaned_lines: list[str] = []
    for line in source.splitlines():
        # Strip % comments
        idx = line.find("%")
        if idx >= 0:
            line = line[:idx]
        # Strip // comments
        idx = line.find("//")
        if idx >= 0:
            line = line[:idx]
        cleaned_lines.append(line.rstrip())

    source = "\n".join(cleaned_lines)

    rules: list[Rule] = []
    for part in source.split("."):
        part = part.strip()
        if not part:
            continue
        try:
            rule = _parse_single_rule(part)
            if rule is not None:
                rules.append(rule)
        except Exception as exc:
            print(f"[Datalog] Warning: could not parse: {part!r} — {exc}")

    return rules


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def _compute_strata(rules: list[Rule]) -> dict[str, int]:
    """
    Assign a stratum number to every predicate.

    Rules:
      - Positive dependency src -> dst:  stratum[dst] >= stratum[src]
      - Negative dependency src -> dst:  stratum[dst] >  stratum[src]
                                          (i.e., >= stratum[src] + 1)

    EDB predicates (never appear as heads of derivation rules) get stratum 0.
    The algorithm iterates until stable, then checks for illegal negative cycles.
    """
    # IDB = predicates that are heads of at least one derivation rule
    idb_preds: set[str] = {r.head.relation for r in rules if not r.is_fact}

    # EDB = everything else (only appears in body, or only as fact heads)
    all_body_preds: set[str] = {
        bl.literal.relation for r in rules for bl in r.body
    }
    edb_preds = all_body_preds - idb_preds
    # Fact-only heads are also EDB
    for r in rules:
        if r.is_fact:
            edb_preds.add(r.head.relation)
            # FIX: A predicate can legitimately have BOTH seed facts and
            # derivation rules (i.e., it appears as the head of a fact and also
            # as the head of a non-fact rule). Such a predicate must remain IDB
            # for stratification purposes; the fact is just an initial tuple.
            # Discarding it from IDB would mis-classify the predicate as EDB
            # and can produce incorrect strata (especially with negation).

    stratum: dict[str, int] = {p: 0 for p in edb_preds | idb_preds}

    # Collect dependency edges from derivation rules
    deps: list[tuple[str, str, bool]] = []  # (src, dst, is_negative)
    for rule in rules:
        if rule.is_fact:
            continue
        for bl in rule.body:
            deps.append((bl.literal.relation, rule.head.relation, bl.negated))

    # Fixpoint: bump strata until stable
    changed = True
    while changed:
        changed = False
        for src, dst, is_neg in deps:
            src_s = stratum.get(src, 0)
            required = src_s + 1 if is_neg else src_s
            if stratum.get(dst, 0) < required:
                stratum[dst] = required
                changed = True

    # Safety check: no negative cycle allowed
    for src, dst, is_neg in deps:
        if is_neg and stratum.get(src, 0) >= stratum.get(dst, 0):
            raise ValueError(
                f"Program is not stratifiable: {dst!r} negates {src!r} but "
                f"stratum({src})={stratum.get(src,0)} >= stratum({dst})={stratum.get(dst,0)}"
            )

    return stratum


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

def _unify(
    pattern: list[str],
    values: tuple,
    binding: dict[str, str],
) -> Optional[dict[str, str]]:
    """
    Try to unify pattern (a list of variable/constant tokens) with a
    concrete tuple of values under an existing binding.

    Returns an extended binding on success, or None on failure.
    """
    if len(pattern) != len(values):
        return None
    new_binding = dict(binding)
    for pat, val in zip(pattern, values):
        if _is_variable(pat):
            if pat in new_binding:
                if new_binding[pat] != val:
                    return None  # Conflict with existing binding
            else:
                new_binding[pat] = val
        else:
            # Constant: must match exactly
            if pat != val:
                return None
    return new_binding


# ---------------------------------------------------------------------------
# Datalog Engine
# ---------------------------------------------------------------------------

class DatalogEngine:
    """
    Stratified Datalog engine using naive bottom-up fixpoint evaluation.

    Usage:
        engine = DatalogEngine()
        engine.add_fact("EntityRole", "alice", "vp")
        engine.add_rule("Allowed(E, T, D) :- EntityRole(E, vp), SensitiveDoc(D).")
        engine.evaluate()
        results = engine.query("Allowed", "alice", None, None)
    """

    def __init__(self):
        # EDB: ground facts provided via add_fact()
        self.edb: dict[str, set[tuple]] = defaultdict(set)
        # Rules: both facts (parsed from policy files) and derivation rules
        self.rules: list[Rule] = []
        # DB: working database (populated by evaluate())
        self.db: dict[str, set[tuple]] = defaultdict(set)

    def add_fact(self, relation: str, *args: str) -> None:
        """Add an EDB (ground) fact."""
        self.edb[relation].add(tuple(str(a) for a in args))

    def add_rule(self, rule_str: str) -> None:
        """Parse and add Datalog rule(s) from a string."""
        self.rules.extend(parse_rules(rule_str))

    def evaluate(self) -> None:
        """
        Run stratified bottom-up evaluation.

        After calling this, self.db contains all derivable facts.
        Can be called multiple times; resets and recomputes each time.
        """
        # 1. Initialise working DB from EDB
        self.db = defaultdict(set)
        for relation, tuples in self.edb.items():
            self.db[relation] = set(tuples)

        # 2. Process fact rules (rules with empty body)
        for rule in self.rules:
            if rule.is_fact:
                self.db[rule.head.relation].add(tuple(rule.head.args))

        # 3. Compute strata for derivation rules
        deriv_rules = [r for r in self.rules if not r.is_fact]
        if not deriv_rules:
            return

        strata = _compute_strata(self.rules)
        max_stratum = max(strata.get(r.head.relation, 0) for r in deriv_rules)

        # 4. Evaluate stratum by stratum
        for s in range(max_stratum + 1):
            s_rules = [r for r in deriv_rules if strata.get(r.head.relation, 0) == s]
            if not s_rules:
                continue

            # Naive fixpoint within this stratum
            changed = True
            while changed:
                changed = False
                for rule in s_rules:
                    for t in self._eval_rule(rule):
                        rel = rule.head.relation
                        if t not in self.db[rel]:
                            self.db[rel].add(t)
                            changed = True

    def _eval_rule(self, rule: Rule) -> set[tuple]:
        """
        Evaluate a rule against self.db.
        Returns the set of new head tuples that can be derived.

        Positive body literals are evaluated first (to maximally bind
        variables before negation checks).
        """
        bindings: list[dict[str, str]] = [{}]

        # --- Positive literals: extend bindings ---
        for bl in rule.body:
            if bl.negated:
                continue
            new_bindings: list[dict[str, str]] = []
            for b in bindings:
                for t in self.db.get(bl.literal.relation, set()):
                    extended = _unify(bl.literal.args, t, b)
                    if extended is not None:
                        new_bindings.append(extended)
            bindings = new_bindings
            if not bindings:
                return set()

        # --- Negative literals: filter bindings ---
        for bl in rule.body:
            if not bl.negated:
                continue
            filtered: list[dict[str, str]] = []
            for b in bindings:
                # Apply current binding to the negated literal's args
                applied_args = [b.get(a, a) for a in bl.literal.args]
                # Check whether any tuple in the relation matches
                matches = any(
                    _unify(applied_args, t, {}) is not None
                    for t in self.db.get(bl.literal.relation, set())
                )
                if not matches:
                    filtered.append(b)
            bindings = filtered
            if not bindings:
                return set()

        # --- Project bindings onto head ---
        results: set[tuple] = set()
        for b in bindings:
            head_tuple: list[str] = []
            ok = True
            for arg in rule.head.args:
                if _is_variable(arg):
                    if arg not in b:
                        ok = False
                        break
                    head_tuple.append(b[arg])
                else:
                    head_tuple.append(arg)
            if ok:
                results.add(tuple(head_tuple))

        return results

    def query(self, relation: str, *pattern) -> list[tuple]:
        """
        Query the database after evaluate() has been called.
        Use None as a wildcard that matches any value.

        Example:
            engine.query("Denied", "bob", None, None)
            # returns all ("bob", tool, reason) tuples
        """
        results: list[tuple] = []
        for t in self.db.get(relation, set()):
            if len(t) != len(pattern):
                continue
            if all(p is None or p == v for p, v in zip(pattern, t)):
                results.append(t)
        return results
