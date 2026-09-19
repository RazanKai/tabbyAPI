"""TabbyAPI resource-aware orchestration.

This package adds model-lifecycle governance to TabbyAPI *inside its existing
process*. It does not proxy, route, or reimplement inference: token generation,
sampling, templating, tool parsing and streaming stay exactly as upstream wrote
them. Only two things are added:

* a single lifecycle coordinator that owns load/unload/lease transitions, and
* a thin admission/lease integration at the request and management boundaries.

Modules
-------
``telemetry``
    Read-only NVML sampling plus process-identity resolution. Publishes immutable
    snapshots. Knows nothing about TabbyAPI.
``policy``
    Pure decision functions over snapshots. No I/O, no clock reads, no locks.
``lifecycle``
    The coordinator: leases, transitions, deadlines, idle timer, drain.
``api``
    FastAPI routes and response schemas for status/pause/resume.

Deliberately absent in V1: queues, durable state, schedulers, multi-model
support. See SPEC section 2 for the exclusions this package is held to.
"""
