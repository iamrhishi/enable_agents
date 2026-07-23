#!/usr/bin/env python3
"""
Lead Nurturing Workflow Data Population Script

Fills in realistic stage `data` for a seeded lead-nurture instance whose
completed stages have status/timestamps but empty data (a gap left by
whatever seeded them - the instance-level `context` had summary numbers,
but they were never copied into each stage's own `data`).

Usage: python scripts/populate_lead_nurture_demo.py <instance_id>
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from datetime import datetime, timedelta
from core.database import db
from models.workflow import WorkflowInstance


def populate_workflow(workflow_id: str):
    print(f"\n Populating Lead Nurture Workflow: {workflow_id}")
    print("=" * 60)

    from app import app

    with app.app_context():
        instance = WorkflowInstance.query.filter_by(instance_id=workflow_id).first()
        if not instance:
            print(f"Workflow {workflow_id} not found!")
            return False

        print(f"Found workflow: {instance.name}")
        print(f"  Status: {instance.status}, current_stage_index: {instance.current_stage_index}")

        ctx = instance.context or {}
        states = instance.stage_states or {}

        total_leads = ctx.get("total_leads", 150)
        qualified_leads = ctx.get("qualified_leads", 85)
        emails_drafted = ctx.get("emails_drafted", 40)
        campaign_name = ctx.get("campaign_name", instance.name)

        # STAGE 1: Lead Qualification (agent: market_research / data_insights)
        # required_inputs: [lead_list], outputs: [qualified_leads]
        if states.get("qualify", {}).get("status") == "completed" and not states["qualify"].get("data"):
            stage1_data = {
                "lead_list": f"{total_leads} leads imported from {campaign_name} CRM export",
                "qualification_criteria": "Company size 50-500 employees, budget confirmed, active buying signal in last 30 days",
                "total_leads": total_leads,
                "qualified_leads": qualified_leads,
                "disqualified_leads": total_leads - qualified_leads,
                "top_segments": ["Mid-market SaaS", "Enterprise IT", "Financial Services"],
                "avg_fit_score": 78,
            }
            states["qualify"]["data"] = stage1_data
            print(f"\n Stage 1 (Lead Qualification): {qualified_leads}/{total_leads} leads qualified")

        # STAGE 2: Content Personalization (agent: content_marketing)
        # required_inputs: [qualified_leads], outputs: [personalized_content]
        if states.get("personalize", {}).get("status") == "completed" and not states["personalize"].get("data"):
            stage2_data = {
                "qualified_leads": qualified_leads,
                "channel": "email",
                "content_type": "email",
                "personalized_content": (
                    f"Subject: A faster way to scale {campaign_name.split(' ')[0]} operations\n\n"
                    "Hi {{first_name}},\n\n"
                    "I noticed your team has been growing quickly this quarter - congrats! "
                    "Companies at your stage often hit a wall automating repetitive ops work "
                    "across sales, support, and fulfillment.\n\n"
                    "We built Enable Agents to close that gap: AI agents that plug into your "
                    "existing stack and handle the busywork end-to-end.\n\n"
                    "Worth a quick call this week?"
                ),
                "variations": [
                    "Variation 1: Shorter, direct CTA version for VP-level contacts",
                    "Variation 2: Case-study-led version referencing a similar customer",
                    "Variation 3: Question-led version optimized for reply rate",
                ],
                "emails_drafted": emails_drafted,
            }
            states["personalize"]["data"] = stage2_data
            print(f"\n Stage 2 (Content Personalization): {emails_drafted} personalized emails drafted")

        instance.stage_states = states
        db.session.commit()

        print("\n" + "=" * 60)
        print(" Done - completed stages now have real data")
        print("=" * 60)
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python populate_lead_nurture_demo.py <instance_id>")
        sys.exit(1)
    success = populate_workflow(sys.argv[1])
    sys.exit(0 if success else 1)
