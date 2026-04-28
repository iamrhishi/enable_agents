"""Email Outreach Agent — Flask Blueprint."""
from flask import Blueprint
from . import service

email_outreach_bp = Blueprint("email_outreach", __name__, url_prefix="/api/email")


@email_outreach_bp.post("/campaigns")
def create_campaign():
    return service.create_campaign()


@email_outreach_bp.get("/campaigns")
def list_campaigns():
    return service.list_campaigns()


@email_outreach_bp.get("/campaigns/<campaign_id>/stats")
def campaign_stats(campaign_id: str):
    return service.campaign_stats(campaign_id)


@email_outreach_bp.get("/campaigns/<campaign_id>/recipients")
def campaign_recipients(campaign_id: str):
    return service.campaign_recipients(campaign_id)


@email_outreach_bp.post("/campaigns/<campaign_id>/send")
def send_campaign(campaign_id: str):
    return service.send_campaign(campaign_id)


@email_outreach_bp.post("/generate")
def generate_email():
    return service.generate_email()


@email_outreach_bp.post("/send-bulk")
def send_bulk():
    return service.send_bulk()


@email_outreach_bp.get("/usage")
def usage():
    return service.usage()
