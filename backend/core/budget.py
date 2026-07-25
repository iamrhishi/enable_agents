"""
Soft monthly spend alerting for projects with a budget configured.

This is intentionally NOT enforcement - no call is ever blocked because a
project is over budget. Blocking AI calls when a shared key is over budget
would be a confusing, hard-to-diagnose failure mode for whoever hits it
next, and this codebase doesn't have a way to explain that to an end user
mid-workflow. Instead: project owners/admins can set a monthly_budget_usd
on a project, and the first time actual spend crosses it in a given
calendar month, the project owner gets a single email. They're expected to
use the /usage dashboard for anything more granular than that.
"""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _current_month_spend_usd(project_id: str) -> float:
    from sqlalchemy import func
    from core.database import db
    from core.models import AIUsageLog

    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    total = (
        db.session.query(func.coalesce(func.sum(AIUsageLog.estimated_cost_usd), 0.0))
        .filter(AIUsageLog.project_id == project_id, AIUsageLog.created_at >= month_start)
        .scalar()
    )
    return float(total or 0.0)


def check_and_maybe_alert_budget(project_id: str) -> None:
    """Called after every usage log write. Cheap no-op unless the project
    has a budget configured; sends at most one email per calendar month."""
    from core.database import db
    from core.models import Project

    project = Project.query.filter_by(project_id=project_id).first()
    if not project or not project.monthly_budget_usd:
        return

    current_month = datetime.utcnow().strftime("%Y-%m")
    if project.budget_alert_month == current_month:
        return  # already alerted this month

    spend = _current_month_spend_usd(project_id)
    if spend < project.monthly_budget_usd:
        return

    try:
        from core.email_sender import send_platform_email

        subject = f'"{project.name}" has crossed its AI budget for {current_month}'
        body = (
            f'Project "{project.name}" has spent an estimated ${spend:.2f} on AI usage this month, '
            f'crossing the ${project.monthly_budget_usd:.2f} budget you set.\n\n'
            'This is informational only - AI actions in this project are not blocked. '
            'View the full breakdown by agent, model, and member in the Usage dashboard.'
        )
        sent, error = send_platform_email(project.owner_id, project.owner_id, subject, body)
        if not sent:
            logger.warning(f"Budget alert email failed for project {project_id}: {error}")
    except Exception as e:
        logger.warning(f"Budget alert email failed for project {project_id}: {e}")

    # Mark alerted for this month regardless of whether the email actually
    # sent - we don't want to retry-spam on every subsequent call this month
    # if e.g. the owner's mail account is disconnected.
    project.budget_alert_month = current_month
    db.session.commit()
