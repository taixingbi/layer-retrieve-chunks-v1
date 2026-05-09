"""Per-request access control for RAG retrieval.

Reads ``X-User-Id`` / ``X-User-Roles`` / ``X-User-Groups`` / ``X-User-Teams`` from the
inbound HTTP request, builds a Qdrant payload filter against
``payload.access.{roles,groups,teams}``, and applies it on the dense retrieval leg.

Semantics (see ``docs/access-control.md``):

- **ANY-OVERLAP** across dimensions: a chunk is visible if any one of the user's
  roles overlaps ``access.roles`` OR any group overlaps ``access.groups`` OR any team
  overlaps ``access.teams``. Empty user dimensions contribute zero ``should`` clauses.
- **Admin bypass**: a user whose ``roles`` contains ``"admin"`` (case-insensitive)
  bypasses filtering entirely (``build_qdrant_access_filter`` returns ``None``).
- **Anonymous default**: if ``X-User-Roles`` is missing or empty, the user's roles
  default to ``["anyuser"]`` so that chunks tagged with ``"anyuser"`` in
  ``access.roles`` form the public set. ``X-User-Id`` defaults to ``"-"``.
- **Deny-by-default on chunks**: chunks without an ``access`` payload field can
  never match a non-admin filter (``MatchAny`` against an absent field fails),
  so they are returned only to admins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qdrant_client.models import FieldCondition, Filter, MatchAny

if TYPE_CHECKING:
    from starlette.requests import Request


_ANONYMOUS_ROLE = "anyuser"
_ADMIN_ROLE_LOWER = "admin"


def _parse_csv(raw: str | None) -> list[str]:
    """Split on commas, strip each item, drop empties. Preserves case."""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class RagUser:
    """Caller identity for a single ``/v1/rag/query`` request.

    ``id`` is for logging / response echo; the three lists drive the Qdrant
    access filter. Construction is via :meth:`from_headers`; tests build instances
    directly. Frozen so the same instance can be threaded through
    ``complete_rag_answer`` → ``_rag_prepare`` → ``query_chunks`` without copy fear.
    """

    id: str = "-"
    roles: list[str] = field(default_factory=lambda: [_ANONYMOUS_ROLE])
    groups: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)

    @property
    def is_admin(self) -> bool:
        """``True`` when any of ``roles`` is ``"admin"`` (case-insensitive)."""
        return any(r.lower() == _ADMIN_ROLE_LOWER for r in self.roles)

    @property
    def is_anonymous(self) -> bool:
        """Convenience: roles is exactly the anonymous default."""
        return self.roles == [_ANONYMOUS_ROLE] and not self.groups and not self.teams

    @classmethod
    def from_headers(cls, request: Request) -> RagUser:
        """Build from Starlette ``Request`` headers (case-insensitive lookup).

        Missing / empty ``X-User-Roles`` defaults to ``["anyuser"]``. Missing
        ``X-User-Id`` becomes ``"-"`` (matches the contextvar default elsewhere).
        ``X-User-Groups`` / ``X-User-Teams`` default to empty lists when absent.
        """
        uid = (request.headers.get("x-user-id") or "").strip() or "-"
        roles = _parse_csv(request.headers.get("x-user-roles"))
        groups = _parse_csv(request.headers.get("x-user-groups"))
        teams = _parse_csv(request.headers.get("x-user-teams"))
        if not roles:
            roles = [_ANONYMOUS_ROLE]
        return cls(id=uid, roles=roles, groups=groups, teams=teams)


def build_qdrant_access_filter(user: RagUser | None) -> Filter | None:
    """Construct the payload filter for :func:`qdrant_client.AsyncQdrantClient.query_points`.

    Returns ``None`` when ``user`` is ``None`` (no access enforcement) or when
    ``user.is_admin`` is true (admin bypass). Otherwise returns a ``Filter(should=[...])``
    with one :class:`FieldCondition` per non-empty user dimension. ``MatchAny`` against
    a missing payload field fails to match, which is the deny-by-default mechanism for
    chunks with no ``payload.access``.

    If the user has no roles / groups / teams at all (defensive — the caller should
    apply the ``anyuser`` default), returns a sentinel "deny everything" filter so
    we never silently degrade to "no filter" for a populated identity.
    """
    if user is None or user.is_admin:
        return None

    should: list[FieldCondition] = []
    if user.roles:
        should.append(FieldCondition(key="access.roles", match=MatchAny(any=user.roles)))
    if user.groups:
        should.append(FieldCondition(key="access.groups", match=MatchAny(any=user.groups)))
    if user.teams:
        should.append(FieldCondition(key="access.teams", match=MatchAny(any=user.teams)))

    if not should:
        # Defensive deny-everything: a non-admin user with zero dimensions cannot
        # see anything. The route's RagUser.from_headers guarantees roles is at
        # least ["anyuser"], so this path should be unreachable in production.
        return Filter(
            must=[FieldCondition(key="__deny_all__", match=MatchAny(any=[]))]
        )

    return Filter(should=should)


def compact_for_log(user: RagUser | None) -> dict[str, object]:
    """Compact dict for one-line ``logger.info("... user=%s", compact_for_log(user))``.

    The full lists are emitted via ``extra={"user_roles": ..., "user_groups": ..., "user_teams": ...}``
    so dashboards can drill in; this helper keeps the human-readable line short.
    """
    if user is None:
        return {"id": "-", "n_roles": 0, "n_groups": 0, "n_teams": 0, "admin": False}
    return {
        "id": user.id,
        "n_roles": len(user.roles),
        "n_groups": len(user.groups),
        "n_teams": len(user.teams),
        "admin": user.is_admin,
    }
