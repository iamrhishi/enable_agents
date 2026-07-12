"""Email Outreach Agent — service layer."""
import os
from datetime import datetime
from uuid import uuid4

from flask import jsonify, request

from core.database import db
from .models import EmailCampaign, EmailCampaignRecipient, EmailExtractionQuota


def create_campaign():
    data = request.get_json(silent=True) or {}
    required = ["name", "subject", "body_template"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    campaign_id = str(uuid4())
    campaign = EmailCampaign(
        id=campaign_id,
        name=data["name"],
        subject=data["subject"],
        body_template=data["body_template"],
        username=data.get("username", "system"),
    )
    db.session.add(campaign)

    for r in data.get("recipients", []):
        db.session.add(EmailCampaignRecipient(
            campaign_id=campaign_id,
            receiver_email=r.get("email"),
            receiver_name=r.get("name"),
            company=r.get("company"),
        ))

    db.session.commit()
    return jsonify({"success": True, "campaign_id": campaign_id}), 201


def list_campaigns():
    campaigns = EmailCampaign.query.order_by(EmailCampaign.created_at.desc()).all()
    return jsonify({"success": True, "campaigns": [
        {"campaign_id": c.id, "name": c.name, "status": c.status, "created_at": c.created_at.isoformat()}
        for c in campaigns
    ]})


def campaign_stats(campaign_id: str):
    campaign = EmailCampaign.query.filter_by(id=campaign_id).first()
    if not campaign:
        return jsonify({"error": "Campaign not found"}), 404
    recipients = EmailCampaignRecipient.query.filter_by(campaign_id=campaign_id).all()
    sent = sum(1 for r in recipients if r.status == "Sent")
    failed = sum(1 for r in recipients if r.status == "failed")
    return jsonify({
        "success": True,
        "campaign_id": campaign_id,
        "name": campaign.name,
        "total": len(recipients),
        "sent": sent,
        "failed": failed,
        "pending": len(recipients) - sent - failed,
        "status": campaign.status,
    })


def campaign_recipients(campaign_id: str):
    recipients = EmailCampaignRecipient.query.filter_by(campaign_id=campaign_id).all()
    return jsonify({"success": True, "recipients": [
        {"email": r.receiver_email, "name": r.receiver_name, "company": r.company, "status": r.status}
        for r in recipients
    ]})


def send_campaign(campaign_id: str):
    # Delegate to legacy handler during migration
    try:
        from app import send_bulk_emails  # noqa
        return send_bulk_emails()
    except (ImportError, AttributeError):
        return jsonify({"error": "Send not yet migrated"}), 501


def generate_email():
    try:
        from app import generate_email as legacy  # noqa
        return legacy()
    except (ImportError, AttributeError):
        return jsonify({"error": "Generate not yet migrated"}), 501


def send_bulk():
    try:
        from app import send_bulk_emails as legacy  # noqa
        return legacy()
    except (ImportError, AttributeError):
        return jsonify({"error": "Bulk send not yet migrated"}), 501


def usage():
    username = request.args.get("username", "default_user")
    quota = EmailExtractionQuota.query.filter_by(username=username).first()
    if not quota:
        return jsonify({"username": username, "emails_used": 0, "monthly_limit": 500})
    return jsonify({
        "username": username,
        "emails_used": quota.emails_used_this_month,
        "monthly_limit": quota.monthly_limit,
    })
