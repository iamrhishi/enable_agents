"""
Shared outbound email sending, used by any feature that needs to send a
real email on behalf of a user (team invites, task reminders, campaign
sends, etc.) instead of duplicating this logic at each call site.

Tries the sender's connected Gmail account first (refreshing the OAuth
token if needed), then falls back to system SMTP (EMAIL_USER/EMAIL_PASS).
"""
from __future__ import annotations

import base64
import os
import smtplib
from email.message import EmailMessage
from typing import Optional, Tuple

GMAIL_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.send",
]


def send_platform_email(sender_user_id: str, to_email: str, subject: str, body: str) -> Tuple[bool, Optional[str]]:
    """
    Send a real email as sender_user_id. Returns (success, error) - never
    raises, so callers can always show the user an accurate status instead
    of silently pretending an email went out.

    Late-imports GoogleOAuthToken/db from app.py rather than core.models:
    that model is still defined in app.py (a larger, separate refactor to
    move it - see TODO below), and importing it at module load time here
    would create a circular import since app.py imports from core.* during
    startup.
    """
    # TODO(tech-debt): GoogleOAuthToken belongs in core/models.py alongside
    # Team/Project/etc. so callers don't need this late-import workaround.
    from app import GoogleOAuthToken, db

    token_record = GoogleOAuthToken.query.filter_by(username=sender_user_id).first()
    gmail_error_summary = None

    if token_record and token_record.token:
        try:
            success, error = _send_via_gmail(token_record, sender_user_id, to_email, subject, body, db)
            if success:
                return True, None
            gmail_error_summary = error
        except Exception as gmail_error:
            gmail_error_summary = str(gmail_error).split('.', 1)[0][:200]
            print(f"[send_platform_email] Gmail send failed for {sender_user_id}, trying SMTP: {gmail_error}")

    return _send_via_smtp(sender_user_id, to_email, subject, body, gmail_error_summary)


def _send_via_gmail(token_record, sender_user_id, to_email, subject, body, db) -> Tuple[bool, Optional[str]]:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    import googleapiclient.discovery

    creds = Credentials(
        token=token_record.token,
        refresh_token=token_record.refresh_token,
        token_uri=token_record.token_uri,
        client_id=token_record.client_id,
        client_secret=token_record.client_secret,
        scopes=token_record.scopes.split(',') if token_record.scopes else GMAIL_SCOPES,
    )
    if creds.refresh_token and (not creds.valid or creds.expired):
        creds.refresh(Request())
        token_record.token = creds.token
        db.session.commit()

    service = googleapiclient.discovery.build('gmail', 'v1', credentials=creds)
    message = EmailMessage()
    message.set_content(body)
    message['To'] = to_email
    message['From'] = sender_user_id
    message['Subject'] = subject
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={'raw': encoded_message}).execute()
    return True, None


def _send_via_smtp(sender_user_id, to_email, subject, body, gmail_error_summary) -> Tuple[bool, Optional[str]]:
    email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
    email_port = int(os.getenv('EMAIL_PORT', 587))
    email_user = os.getenv('EMAIL_USER')
    email_pass = os.getenv('EMAIL_PASS')
    if not email_user or not email_pass:
        if gmail_error_summary:
            return False, f'Gmail send failed ({gmail_error_summary}) and SMTP fallback is unavailable.'
        return False, 'No email account connected. Connect Google in Settings, or ask an admin to configure SMTP.'

    try:
        smtp_server = smtplib.SMTP(email_host, email_port, timeout=15)
        smtp_server.starttls()
        smtp_server.login(email_user, email_pass)
        message = EmailMessage()
        message.set_content(body)
        message['To'] = to_email
        message['From'] = email_user
        message['Reply-To'] = sender_user_id
        message['Subject'] = subject
        smtp_server.send_message(message)
        smtp_server.quit()
        return True, None
    except Exception as smtp_error:
        return False, f'Email send failed: {str(smtp_error).split(".", 1)[0][:200]}'
