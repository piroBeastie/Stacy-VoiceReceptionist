"""Loading tenants.

JSON files on disk today, Postgres later. The interface is the routing question
-- "a call came in on this number, who is it for?" -- so the swap is one class.
"""

from __future__ import annotations

import json
from pathlib import Path

from stacy.tenants.models import Tenant


class TenantStore:
    """Reads tenant records and indexes them by dialed number."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._by_id: dict[str, Tenant] = {}
        self._by_number: dict[str, Tenant] = {}
        self.reload()

    def reload(self) -> None:
        """Re-read every *.json in the tenant directory.

        Raises on a duplicate phone number: two tenants on one line is a
        misconfiguration that would otherwise surface as a caller reaching the
        wrong business, which is the worst possible place to find out.
        """
        by_id: dict[str, Tenant] = {}
        by_number: dict[str, Tenant] = {}
        for path in sorted(self._dir.glob("*.json")):
            tenant = Tenant.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if tenant.phone_number in by_number:
                other = by_number[tenant.phone_number].tenant_id
                raise ValueError(
                    f"{tenant.phone_number} is claimed by both {other!r} and {tenant.tenant_id!r}"
                )
            by_id[tenant.tenant_id] = tenant
            by_number[tenant.phone_number] = tenant
        self._by_id, self._by_number = by_id, by_number

    def for_number(self, dialed: str) -> Tenant:
        """Route an inbound call. Raises KeyError if the number is unclaimed."""
        try:
            return self._by_number[dialed]
        except KeyError:
            raise KeyError(f"no tenant configured for dialed number {dialed!r}") from None

    def get(self, tenant_id: str) -> Tenant:
        return self._by_id[tenant_id]

    def all(self) -> list[Tenant]:
        return list(self._by_id.values())
