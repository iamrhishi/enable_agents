"""Event Networking Agent — Flask Blueprint."""
from flask import Blueprint

from . import service

event_networking_bp = Blueprint(
    "event_networking",
    __name__,
    url_prefix="/api/event-networking",
)


@event_networking_bp.get("/events")
def list_events():
    return service.list_events()


@event_networking_bp.post("/events")
def create_event():
    return service.create_event()


@event_networking_bp.get("/events/<event_id>/attendees")
def list_attendees(event_id):
    return service.list_attendees(event_id)


@event_networking_bp.post("/events/<event_id>/attendees")
def upload_attendees(event_id):
    return service.upload_attendees(event_id)


@event_networking_bp.post("/events/<event_id>/recommendations")
def get_recommendations(event_id):
    return service.get_recommendations(event_id)


@event_networking_bp.post("/events/<event_id>/followup")
def send_followup(event_id):
    return service.send_followup(event_id)


@event_networking_bp.put("/attendees/<attendee_id>/notes")
def update_attendee_notes(attendee_id):
    return service.update_attendee_notes(attendee_id)


@event_networking_bp.put("/attendees/<attendee_id>/followup-date")
def update_attendee_followup_date(attendee_id):
    return service.update_attendee_followup_date(attendee_id)
