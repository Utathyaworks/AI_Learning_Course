"""Access control: only emails on the allowlist get past this gate.

With an [auth] section in secrets, Streamlit's built-in OpenID login is used
(Google by default) and the signed-in email must be on ALLOWED_EMAILS.
Without it, the app refuses to open unless DEV_MODE_NO_AUTH is set, which is
meant only for running on your own computer.
"""
from __future__ import annotations

import streamlit as st


def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def require_login(settings) -> str:
    """Stop the script unless the viewer is allowed. Returns the viewer's email."""
    if settings.dev_mode_no_auth and not _auth_configured():
        st.sidebar.warning("Developer mode: sign-in is off. Never deploy with DEV_MODE_NO_AUTH.")
        return "local-dev"

    if not _auth_configured():
        st.error(
            "Sign-in isn't set up, so the coach won't open. Add an [auth] section to the app's secrets "
            "(see README → Deploy)."
        )
        st.stop()

    if not settings.allowed_emails:
        st.error("ALLOWED_EMAILS is empty in the app's secrets, so nobody can open the coach.")
        st.stop()

    if not st.user.is_logged_in:
        st.title("DL Coach")
        st.write("Sign in with your approved Google account to continue.")
        st.button("Sign in with Google", on_click=st.login, type="primary")
        st.stop()

    email = (st.user.get("email") or "").strip().lower()
    verified = st.user.get("email_verified", True)
    if email not in settings.allowed_emails or verified is False:
        st.error(f"{email or 'This account'} isn't approved for this coach.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    return email
