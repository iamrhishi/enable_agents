"""Email Outreach Agent — SQLAlchemy models (replaces raw sqlite3 tables)."""
from datetime import datetime
from core.database import db


class EmailCampaign(db.Model):
    __tablename__ = "email_campaigns"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(512))
    body_template = db.Column(db.Text)
    username = db.Column(db.String(120), index=True)
    sender_email = db.Column(db.String(255), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sent_at = db.Column(db.DateTime)
    status = db.Column(db.String(50), default="draft")  # draft | sent | failed

    recipients = db.relationship("EmailCampaignRecipient", back_populates="campaign", cascade="all, delete-orphan")


class EmailCampaignRecipient(db.Model):
    __tablename__ = "email_campaign_recipients"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    campaign_id = db.Column(db.String(36), db.ForeignKey("email_campaigns.id"), nullable=False, index=True)
    receiver_email = db.Column(db.String(255), nullable=False, index=True)
    receiver_name = db.Column(db.String(255))
    company = db.Column(db.String(255))
    personalised_body = db.Column(db.Text)
    status = db.Column(db.String(50), default="Sent")
    reply_status = db.Column(db.String(50), default="No Reply")
    message_id = db.Column(db.String(255), nullable=True)
    thread_id = db.Column(db.String(255), nullable=True)
    reply_subject = db.Column(db.String(512), nullable=True)
    reply_snippet = db.Column(db.Text, nullable=True)
    reply_body = db.Column(db.Text, nullable=True)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    replied_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text)

    campaign = db.relationship("EmailCampaign", back_populates="recipients")


class EmailExtractionQuota(db.Model):
    __tablename__ = "email_extraction_quotas"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False, index=True)
    monthly_limit = db.Column(db.Integer, default=500)
    emails_used_this_month = db.Column(db.Integer, default=0)
    reset_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
