"""Per-business configuration.

Everything that differs between clients lives here and nowhere else. The prompt
template, the pipeline, and the eval suite are all tenant-agnostic; they read
from a :class:`Tenant` and never hardcode a business fact.
"""

from stacy.tenants.models import BookingWindow, Reseller, Staff, Tenant
from stacy.tenants.store import TenantStore

__all__ = ["Tenant", "Staff", "Reseller", "BookingWindow", "TenantStore"]
