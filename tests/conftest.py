"""Shared fixtures.

The tenant here is the real Millbrook record rather than a synthetic one, so a
change to the shipped tenant data that breaks booking or availability fails the
suite instead of being discovered on a call.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from stacy.tenants import Tenant, TenantStore

TENANT_DIR = "src/stacy/tenants/data"


@pytest.fixture(scope="session")
def store() -> TenantStore:
    return TenantStore(TENANT_DIR)


@pytest.fixture(scope="session")
def tenant(store: TenantStore) -> Tenant:
    return store.get("millbrook")


@pytest.fixture
def monday_9am(tenant: Tenant) -> datetime:
    """A fixed Monday morning, inside business hours, in the tenant's timezone.

    Availability depends on 'now' in three ways -- open day, elapsed window,
    weekday resolution -- so every test that touches it pins the clock.
    """
    return datetime(2026, 9, 14, 9, 0, tzinfo=ZoneInfo(tenant.timezone))
