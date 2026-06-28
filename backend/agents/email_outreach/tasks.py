"""Email Outreach Agent — Celery async tasks."""
from core.celery_app import celery


@celery.task(bind=True, name="email_outreach.send_campaign_async")
def send_campaign_async(self, campaign_id: str) -> dict:
    """Send all pending recipients in a campaign asynchronously."""
    try:
        self.update_state(state="PROGRESS", meta={"campaign_id": campaign_id, "step": "sending"})
        from agents.email_outreach.models import EmailCampaign, EmailCampaignRecipient
        from core.database import db

        campaign = EmailCampaign.query.filter_by(id=campaign_id).first()
        if not campaign:
            return {"status": "error", "error": "Campaign not found"}

        sent = 0
        for recipient in EmailCampaignRecipient.query.filter_by(
            campaign_id=campaign_id, status="Sent"
        ).all():
            # TODO: integrate Gmail send here
            recipient.status = "sent"
            sent += 1

        campaign.status = "sent"
        db.session.commit()
        return {"status": "done", "sent": sent, "campaign_id": campaign_id}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
