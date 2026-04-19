"""Read-only web dashboard for the centralized Radar experiment DB.

Mounts under ``/dashboard`` inside ``database/server.py``. See
``database/dashboard/app.py::mount_dashboard`` for the entry point.
"""

from database.dashboard.app import mount_dashboard

__all__ = ["mount_dashboard"]
