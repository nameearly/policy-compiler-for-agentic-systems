"""
MockDriveIAM — simulates Google Drive's resource-level "Share with" permissions.

This layer represents what Google Drive IAM (and similar storage-layer ACLs)
can enforce at the point of file access:
  - Read / write / comment access per user per file
  - Applied at the Drive API boundary before any bytes are returned

What it CANNOT enforce (the gap PCAS closes):
  - Tool calls that don't touch Drive directly (send_email, post_slack, etc.)
  - What the LLM does with data once it is loaded into the context window
  - Indirect leakage through shared multi-agent session context (taint)
  - Transitive information flow: data read by user A, then forwarded by user B

Analogy:
  Drive IAM = a locked filing cabinet. It controls who can open the drawer.
  PCAS      = a policy that controls what you can DO with the document
              after it leaves the cabinet — even if it's already in your hands.
"""

from dataclasses import dataclass, field

# Drive permission levels — mirrors the real Drive API
VIEWER    = "viewer"
COMMENTER = "commenter"
EDITOR    = "editor"


@dataclass
class MockDriveIAM:
    """
    Simulates Google Drive's "Share with" access control model.

    Permissions:
        viewer    — read-only (Drive: Viewer)
        commenter — read + comment (Drive: Commenter)
        editor    — read + write (Drive: Editor / Owner)

    Enforcement scope:
        Only Drive API file operations: read_document, list_documents.
        Non-Drive tool calls (send_email, write_to_slack, create_ticket, …)
        are entirely outside this layer's enforcement boundary — it simply
        never sees those calls. This is the structural gap.
    """

    # { entity -> { doc_id -> permission } }
    _acl: dict[str, dict[str, str]] = field(default_factory=dict)

    def share(self, doc_id: str, entity: str, permission: str = VIEWER) -> None:
        """
        Grant entity access to doc_id.
        Equivalent to clicking 'Share → [viewer/editor]' in the Drive UI.
        """
        self._acl.setdefault(entity, {})[doc_id] = permission

    def check_read(self, entity: str, doc_id: str) -> tuple[bool, str]:
        """
        Check whether entity may read doc_id via the Drive API.
        Returns (allowed: bool, log_message: str).

        In the real Drive API this check happens server-side, transparently,
        on every request — the caller gets HTTP 403 if denied.
        """
        perm = self._acl.get(entity, {}).get(doc_id)
        if perm is None:
            return (
                False,
                f"[Drive IAM] {entity!r} → read_document('{doc_id}')"
                f": DENY — not shared with this user (no ACL entry)",
            )
        return (
            True,
            f"[Drive IAM] {entity!r} → read_document('{doc_id}')"
            f": ALLOW — {perm} access",
        )

    def note_non_drive_call(self, entity: str, tool: str) -> str:
        """
        Return the standard note printed for non-Drive tool calls.
        Drive IAM has zero visibility into these — they bypass it entirely.
        """
        return (
            f"[Drive IAM] {entity!r} → {tool}(...)"
            f": OUT OF SCOPE — not a Drive API operation, no IAM check performed"
        )

    def print_acl_table(self) -> None:
        """Print the full ACL matrix for demo / audit output."""
        all_docs  = sorted({d for perms in self._acl.values() for d in perms})
        all_users = sorted(self._acl)
        col = 26
        header_width = col + col * len(all_docs)

        print(f"\n  Drive IAM Access Control List (simulated 'Share with')")
        print("  " + "─" * header_width)
        print(f"  {'Entity':<{col}}", end="")
        for doc in all_docs:
            print(f"{doc:<{col}}", end="")
        print()
        print("  " + "─" * header_width)
        for user in all_users:
            print(f"  {user:<{col}}", end="")
            for doc in all_docs:
                perm = self._acl[user].get(doc, "— (no access)")
                print(f"{perm:<{col}}", end="")
            print()
        print("  " + "─" * header_width)
        print(
            "\n  Note: Drive IAM only controls Drive API calls.\n"
            "        send_email / post_slack / etc. are NEVER checked here."
        )
