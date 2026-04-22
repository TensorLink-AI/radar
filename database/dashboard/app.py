"""Dashboard app — cookie auth, router assembly, and mount helper.

Keeps all session state local to this module so the dashboard never
shares auth state with the Epistula middleware in ``database/server.py``.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from config import Config

logger = logging.getLogger(__name__)

COOKIE_NAME = "radar_dashboard_session"
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


class DashboardState:
    """Shared state for dashboard handlers (db store, pool, r2, serializer)."""

    def __init__(
        self,
        store,
        pool,
        r2,
        get_challenge: Optional[Callable[[], dict]] = None,
        get_frontier: Optional[Callable[[], list]] = None,
    ):
        self.store = store
        self.pool = pool
        self.r2 = r2
        self.get_challenge = get_challenge or (lambda: None)
        self.get_frontier = get_frontier or (lambda: None)
        # HMAC-signed cookie serializer. Key rotates per-process by default.
        key = Config.DASHBOARD_KEY or secrets.token_urlsafe(32)
        self.serializer = URLSafeTimedSerializer(key, salt="radar-dashboard-v1")

    def issue_token(self) -> str:
        return self.serializer.dumps({"v": 1})

    def verify_token(self, token: str) -> bool:
        try:
            self.serializer.loads(token, max_age=Config.DASHBOARD_SESSION_TTL)
            return True
        except (BadSignature, SignatureExpired):
            return False


# Module-level state set by mount_dashboard()
_state: Optional[DashboardState] = None


def get_state() -> DashboardState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    return _state


def get_templates() -> Jinja2Templates:
    return _templates


def require_session(request: Request) -> None:
    """FastAPI dependency: raise 401 if the cookie is missing/invalid.

    GET requests to non-API paths get a 302 to ``/dashboard/login`` so a
    browser hitting a bookmarked URL lands somewhere useful. Anything under
    ``/dashboard/api`` always gets a 401 JSON response.
    """
    if not Config.DASHBOARD_ENABLED or not Config.DASHBOARD_KEY:
        raise HTTPException(status_code=503, detail="Dashboard disabled")
    token = request.cookies.get(COOKIE_NAME, "")
    if not token or not get_state().verify_token(token):
        is_api = request.url.path.startswith("/dashboard/api")
        if request.method == "GET" and not is_api:
            raise _LoginRedirect()
        raise HTTPException(status_code=401, detail="Not authenticated")


class _LoginRedirect(Exception):
    """Internal signal to redirect to the login page."""


def _ensure_state(
    store,
    pool,
    r2,
    get_challenge: Optional[Callable[[], dict]] = None,
    get_frontier: Optional[Callable[[], list]] = None,
) -> DashboardState:
    """Install (or refresh) the module-level ``DashboardState`` singleton."""
    global _state
    _state = DashboardState(
        store=store,
        pool=pool,
        r2=r2,
        get_challenge=get_challenge,
        get_frontier=get_frontier,
    )
    return _state


def mount_public_api(
    app: FastAPI,
    store,
    pool,
    r2,
    get_challenge: Optional[Callable[[], dict]] = None,
    get_frontier: Optional[Callable[[], list]] = None,
) -> None:
    """Mount the **public** JSON API at ``/dashboard/api/*``.

    No auth. Safe to call in dashboard mode (no Jinja, no wallet) and in
    ``all`` mode (shares the app with validator routes). Idempotent.
    """
    _ensure_state(store, pool, r2, get_challenge, get_frontier)

    already_mounted = any(
        getattr(r, "path", "").startswith("/dashboard/api") for r in app.routes
    )
    if already_mounted:
        return

    from database.dashboard import api

    app.include_router(api.router)
    logger.info("Dashboard public JSON API mounted at /dashboard/api/*")


def mount_dashboard(
    app: FastAPI,
    store,
    pool,
    r2,
    get_challenge: Optional[Callable[[], dict]] = None,
    get_frontier: Optional[Callable[[], list]] = None,
) -> None:
    """Attach the dashboard router + static mount to an existing FastAPI app.

    Called from ``database/neuron.py`` after ``set_db(...)``. Mounts the
    cookie-gated Jinja HTML operator UI plus the public JSON API on the
    same app. No-op if the Jinja UI is disabled so operators can turn it
    off in production without losing the public JSON API — that is now
    the responsibility of ``mount_public_api``.
    """
    if not Config.DASHBOARD_ENABLED:
        logger.info("Dashboard disabled (RADAR_DASHBOARD_ENABLED != true)")
        return
    if not Config.DASHBOARD_KEY:
        logger.warning(
            "Dashboard enabled but RADAR_DASHBOARD_KEY is empty — refusing to mount",
        )
        return

    _ensure_state(store, pool, r2, get_challenge, get_frontier)

    # Static assets (HTMX, Chart.js glue, CSS)
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/dashboard/static",
        StaticFiles(directory=str(_STATIC_DIR)),
        name="dashboard_static",
    )

    # Login/logout + session-protected HTML + public JSON routes
    from database.dashboard import api, views

    app.include_router(_auth_router())
    app.include_router(views.router)
    # JSON router is public (no cookie dependency) — mount only if not
    # already wired in by ``mount_public_api``.
    if not any(getattr(r, "path", "").startswith("/dashboard/api") for r in app.routes):
        app.include_router(api.router)

    # Convert _LoginRedirect into a 302 to the login page
    @app.exception_handler(_LoginRedirect)
    async def _on_login_redirect(request: Request, exc: _LoginRedirect):
        next_url = request.url.path
        if request.url.query:
            next_url = f"{next_url}?{request.url.query}"
        return RedirectResponse(url=f"/dashboard/login?next={next_url}", status_code=302)

    logger.info("Dashboard mounted at /dashboard")


# ── Auth routes (login form + logout) ─────────────────────────

def _auth_router():
    from fastapi import APIRouter, Form

    router = APIRouter(prefix="/dashboard", tags=["dashboard"])

    @router.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/dashboard/", error: str = ""):
        return _templates.TemplateResponse(
            request,
            "login.html",
            {"next": next, "error": error},
        )

    @router.post("/login")
    def login_submit(
        request: Request,
        key: str = Form(...),
        next: str = Form("/dashboard/"),
    ):
        expected = Config.DASHBOARD_KEY
        if not expected or not secrets.compare_digest(key, expected):
            return _templates.TemplateResponse(
                request,
                "login.html",
                {"next": next, "error": "Invalid key"},
                status_code=401,
            )
        # Only accept relative paths for ?next= to prevent open-redirect
        if not next.startswith("/dashboard"):
            next = "/dashboard/"
        resp = RedirectResponse(url=next, status_code=302)
        resp.set_cookie(
            key=COOKIE_NAME,
            value=get_state().issue_token(),
            max_age=Config.DASHBOARD_SESSION_TTL,
            httponly=True,
            samesite="lax",
            path="/dashboard",
        )
        return resp

    @router.get("/logout")
    def logout():
        resp = RedirectResponse(url="/dashboard/login", status_code=302)
        resp.delete_cookie(COOKIE_NAME, path="/dashboard")
        return resp

    return router


__all__ = [
    "COOKIE_NAME",
    "DashboardState",
    "get_state",
    "get_templates",
    "mount_dashboard",
    "mount_public_api",
    "require_session",
]
