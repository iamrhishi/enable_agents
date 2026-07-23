#!/usr/bin/env python3
"""
Realistic Workflow Data Population Script

Populates workflow 065b4298-5b8e-4122-b325-b7cb798c7f41 with realistic data
for automotive electronic components supplier qualification.

Scenario: Finding and qualifying suppliers for automotive PCB assemblies
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from datetime import datetime, timedelta
from core.database import db
from models.workflow import WorkflowInstance

# Realistic supplier data for automotive electronics
REALISTIC_SUPPLIERS = [
    {
        "name": "Precision Circuits India Pvt Ltd",
        "location": "Bangalore, Karnataka, India",
        "email": "sales@precisioncircuits.in",
        "phone": "+91-80-2856-3421",
        "website": "https://precisioncircuits.in",
        "specialization": "Automotive PCB Assembly",
        "certifications": ["ISO 9001:2015", "IATF 16949:2016", "ISO 14001:2015"],
        "capacity": "500,000 units/month",
        "quote_price": "$2.45/unit",
        "lead_time": "6-8 weeks",
        "moq": "5,000 units"
    },
    {
        "name": "Shenzhen AutoTech Electronics Co., Ltd",
        "location": "Shenzhen, Guangdong, China",
        "email": "export@sz-autotech.com",
        "phone": "+86-755-8234-9876",
        "website": "https://sz-autotech.com",
        "specialization": "Automotive Control Units",
        "certifications": ["ISO 9001:2015", "IATF 16949:2016", "UL", "CE"],
        "capacity": "1,000,000 units/month",
        "quote_price": "$1.98/unit",
        "lead_time": "4-6 weeks",
        "moq": "10,000 units"
    },
    {
        "name": "Gujarat Electronic Manufacturing Services",
        "location": "Ahmedabad, Gujarat, India",
        "email": "business@gems-india.com",
        "phone": "+91-79-4002-8765",
        "website": "https://gems-india.com",
        "specialization": "EMS for Automotive Sector",
        "certifications": ["ISO 9001:2015", "IATF 16949:2016"],
        "capacity": "300,000 units/month",
        "quote_price": "$2.75/unit",
        "lead_time": "8-10 weeks",
        "moq": "3,000 units"
    },
    {
        "name": "Wuxi Precision Manufacturing Ltd",
        "location": "Wuxi, Jiangsu, China",
        "email": "sales@wuxiprecision.cn",
        "phone": "+86-510-8765-4321",
        "website": "https://wuxiprecision.cn",
        "specialization": "Automotive PCB & PCBA",
        "certifications": ["ISO 9001:2015", "IATF 16949:2016", "ISO 14001:2015", "RoHS"],
        "capacity": "800,000 units/month",
        "quote_price": "$2.15/unit",
        "lead_time": "5-7 weeks",
        "moq": "8,000 units"
    },
    {
        "name": "Chennai Automotive Components Pvt Ltd",
        "location": "Chennai, Tamil Nadu, India",
        "email": "sales@chennai-auto.in",
        "phone": "+91-44-4567-8901",
        "website": "https://chennai-auto.in",
        "specialization": "Automotive Electronics",
        "certifications": ["ISO 9001:2015", "IATF 16949:2016"],
        "capacity": "400,000 units/month",
        "quote_price": "$2.60/unit",
        "lead_time": "7-9 weeks",
        "moq": "4,000 units"
    },
]

def populate_workflow(workflow_id: str):
    """Populate workflow with realistic automotive electronics data"""

    print(f"\n🔧 Populating Workflow: {workflow_id}")
    print("=" * 60)

    # Initialize Flask app context
    from app import app

    with app.app_context():
        # Get workflow instance
        instance = WorkflowInstance.query.filter_by(instance_id=workflow_id).first()
        if not instance:
            print(f"❌ Workflow {workflow_id} not found!")
            return False

        print(f"✓ Found workflow: {instance.name}")
        print(f"  Template: {instance.template.name if instance.template else 'Unknown'}")
        print(f"  Status: {instance.status}")

        # Initialize stage states if needed
        if not instance.stage_states:
            instance.stage_states = {}

        # STAGE 1: Supplier Discovery (Requirements Gathering)
        print("\n📋 Stage 1: Supplier Discovery")
        print("-" * 60)

        stage1_data = {
            "client_name": "AutoMotive Solutions Inc.",
            "component_type": "Automotive PCB Assemblies for Engine Control Units",
            "search_query": "Automotive PCB assembly manufacturers ISO IATF certified India China",
            "location": "India, China",
            "industry": "Automotive Electronics Manufacturing",
            "businesses_found": len(REALISTIC_SUPPLIERS),
            "top_businesses": [s["name"] for s in REALISTIC_SUPPLIERS],
            "requirements": {
                "certifications_required": ["ISO 9001", "IATF 16949"],
                "target_price": "$2.50/unit",
                "target_volume": "50,000 units/month",
                "quality_standards": "Automotive Grade",
                "delivery_terms": "FOB"
            }
        }

        instance.stage_states["supplier_discovery"] = {
            "status": "completed",
            "data": stage1_data,
            "completedAt": (datetime.utcnow() - timedelta(days=5)).isoformat(),
            "updatedAt": (datetime.utcnow() - timedelta(days=5)).isoformat()
        }

        print(f"  ✓ Found {len(REALISTIC_SUPPLIERS)} suppliers")
        print(f"  ✓ Component: {stage1_data['component_type']}")
        print(f"  ✓ Locations: {stage1_data['location']}")

        # STAGE 2: Document Analysis
        print("\n📄 Stage 2: Document Analysis")
        print("-" * 60)

        stage2_data = {
            "document_analyzed": "Supplier Certifications & Datasheets",
            "documents_analyzed": 12,
            "key_findings": [
                "All 5 suppliers hold ISO 9001:2015 and IATF 16949:2016 certifications",
                "Shenzhen AutoTech has highest capacity at 1M units/month with best price ($1.98/unit)",
                "Indian suppliers offer competitive pricing ($2.45-$2.75/unit) with moderate capacity",
                "Lead times range from 4-10 weeks, with Chinese suppliers offering faster delivery",
                "All suppliers meet automotive grade quality standards and RoHS compliance"
            ],
            "recommendations": [
                "Shortlist Shenzhen AutoTech and Precision Circuits for detailed RFQ",
                "Request detailed technical datasheets for PCB assembly capabilities",
                "Verify production capacity with factory audit visits",
                "Negotiate pricing for volume commitments above 50k units/month"
            ],
            "sources": [
                "ISO 9001:2015 Certificates (5 suppliers)",
                "IATF 16949:2016 Certificates (5 suppliers)",
                "Technical capability statements",
                "Production capacity documents"
            ],
            "confidence_score": 0.94,
            "analysis_date": (datetime.utcnow() - timedelta(days=4)).isoformat()
        }

        instance.stage_states["document_analysis"] = {
            "status": "completed",
            "data": stage2_data,
            "completedAt": (datetime.utcnow() - timedelta(days=4)).isoformat(),
            "updatedAt": (datetime.utcnow() - timedelta(days=4)).isoformat()
        }

        print(f"  ✓ Analyzed {stage2_data['documents_analyzed']} documents")
        print(f"  ✓ {len(stage2_data['key_findings'])} key findings identified")
        print(f"  ✓ Confidence: {stage2_data['confidence_score']*100:.0f}%")

        # STAGE 3: RFQ Outreach
        print("\n📧 Stage 3: RFQ Outreach")
        print("-" * 60)

        stage3_data = {
            "emails_sent": 5,
            "total_recipients": 5,
            "template_used": "RFQ - Automotive PCB Assembly",
            "email_subject": "RFQ: Automotive PCB Assembly - 50K Units/Month",
            "email_body": "Dear Supplier,\n\nWe are seeking quotations for automotive-grade PCB assemblies for Engine Control Units. Please provide your best pricing for 50,000 units/month...",
            "recipients": [s["email"] for s in REALISTIC_SUPPLIERS],
            "recipient_names": [s["name"] for s in REALISTIC_SUPPLIERS],
            "sent_date": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "responses_expected": 5
        }

        instance.stage_states["rfq_outreach"] = {
            "status": "completed",
            "data": stage3_data,
            "completedAt": (datetime.utcnow() - timedelta(days=3)).isoformat(),
            "updatedAt": (datetime.utcnow() - timedelta(days=3)).isoformat()
        }

        print(f"  ✓ Sent {stage3_data['emails_sent']} RFQ emails")
        print(f"  ✓ Subject: {stage3_data['email_subject']}")
        print(f"  ✓ Recipients: {', '.join([s['name'] for s in REALISTIC_SUPPLIERS[:3]])}...")

        # STAGE 4: Response Analysis
        print("\n📊 Stage 4: Response Analysis")
        print("-" * 60)

        stage4_data = {
            "leads_analyzed": 5,
            "matched_prospects": 4,
            "top_match": "Shenzhen AutoTech Electronics",
            "project_name": "Automotive PCB Supplier Qualification",
            "match_scores": {
                REALISTIC_SUPPLIERS[1]["name"]: 95,  # Shenzhen AutoTech
                REALISTIC_SUPPLIERS[0]["name"]: 88,  # Precision Circuits India
                REALISTIC_SUPPLIERS[3]["name"]: 85,  # Wuxi Precision
                REALISTIC_SUPPLIERS[4]["name"]: 78,  # Chennai Automotive
            },
            "responses_received": 4,
            "analysis": {
                "best_price": f"{REALISTIC_SUPPLIERS[1]['name']} @ ${REALISTIC_SUPPLIERS[1]['quote_price']}",
                "best_capacity": f"{REALISTIC_SUPPLIERS[1]['name']} - {REALISTIC_SUPPLIERS[1]['capacity']}",
                "fastest_delivery": f"{REALISTIC_SUPPLIERS[1]['name']} - {REALISTIC_SUPPLIERS[1]['lead_time']}"
            }
        }

        instance.stage_states["response_analysis"] = {
            "status": "completed",
            "data": stage4_data,
            "completedAt": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "updatedAt": (datetime.utcnow() - timedelta(days=2)).isoformat()
        }

        print(f"  ✓ Analyzed {stage4_data['leads_analyzed']} responses")
        print(f"  ✓ Qualified {stage4_data['matched_prospects']} suppliers")
        print(f"  ✓ Top match: {stage4_data['top_match']} (95% score)")

        # STAGE 5: Qualification Audit
        print("\n🔍 Stage 5: Qualification Audit")
        print("-" * 60)

        stage5_data = {
            "supplier_audited": "Shenzhen AutoTech Electronics",
            "audit_score": 92,
            "audit_result": "PASSED - Qualified Supplier",
            "total_audited": 3,
            "passed_count": 2,
            "top_scorer": "Shenzhen AutoTech Electronics (92/100)",
            "audit_criteria": {
                "Facility & Equipment": 94,
                "Quality Systems": 95,
                "Production Capacity": 98,
                "Financial Stability": 88,
                "Compliance & Certifications": 85
            },
            "audit_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "auditor": "Quality Assurance Team",
            "findings": [
                "Excellent manufacturing facility with modern SMT lines",
                "Robust quality management system per IATF 16949",
                "Production capacity verified at 1M units/month",
                "Strong financial position with steady growth",
                "All required certifications valid and up to date"
            ]
        }

        instance.stage_states["qualification_audit"] = {
            "status": "completed",
            "data": stage5_data,
            "completedAt": (datetime.utcnow() - timedelta(hours=12)).isoformat(),
            "updatedAt": (datetime.utcnow() - timedelta(hours=12)).isoformat()
        }

        print(f"  ✓ Audited {stage5_data['total_audited']} suppliers")
        print(f"  ✓ Passed: {stage5_data['passed_count']} suppliers")
        print(f"  ✓ Top scorer: {stage5_data['top_scorer']}")
        print(f"  ✓ Status: {stage5_data['audit_result']}")

        # STAGE 6: Selection Tasks (Completed)
        print("\n✅ Stage 6: Selection Tasks")
        print("-" * 60)

        stage6_data = {
            "total_tasks": 8,
            "completed_tasks": 8,
            "pending_tasks": 0,
            "completion_rate": 1.0,
            "completed_items": [
                "Review audit reports",
                "Compare supplier quotes",
                "Prepare recommendation document",
                "Schedule contract negotiation meeting",
                "Draft Master Supply Agreement",
                "Obtain legal approval",
                "Finalize payment terms",
                "Setup supplier in ERP system"
            ],
            "selected_supplier": "Shenzhen AutoTech Electronics",
            "contract_value": "$99,000 (50,000 units @ $1.98/unit)",
            "contract_term": "24 months",
            "payment_terms": "Net 30",
            "delivery_schedule": "Monthly shipments of 4,166 units",
            "team_members": [
                "Procurement Manager",
                "Quality Engineer",
                "Legal Counsel",
                "Supply Chain Director"
            ],
            "decision_rationale": [
                "Best pricing at $1.98/unit vs target $2.50/unit",
                "Highest production capacity (1M units/month)",
                "Shortest lead time (4-6 weeks)",
                "Excellent audit score (92/100)",
                "All required certifications valid"
            ]
        }

        instance.stage_states["selection_tasks"] = {
            "status": "completed",
            "data": stage6_data,
            "startedAt": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
            "completedAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }

        print(f"  ✓ All tasks completed: {stage6_data['completed_tasks']}/{stage6_data['total_tasks']}")
        print(f"  ✓ Selected Supplier: {stage6_data['selected_supplier']}")
        print(f"  ✓ Contract Value: {stage6_data['contract_value']}")
        print(f"  👥 Team: {len(stage6_data['team_members'])} members")

        # Update workflow context with all data
        instance.context = {
            **stage1_data,
            **stage2_data,
            **stage3_data,
            **stage4_data,
            **stage5_data,
            **stage6_data
        }

        # Update workflow status - ALL STAGES COMPLETED
        instance.status = "completed"
        instance.current_stage_index = 5  # Completed stage 6 (0-indexed)
        instance.completed_at = datetime.utcnow()

        # Commit to database
        db.session.commit()

        print("\n" + "=" * 60)
        print("✅ SUCCESS! Workflow populated with realistic data")
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"  • 5 Qualified Suppliers Identified")
        print(f"  • 12 Documents Analyzed")
        print(f"  • 5 RFQ Emails Sent")
        print(f"  • 4 Responses Received & Analyzed")
        print(f"  • 3 Suppliers Audited")
        print(f"  • Selected Supplier: {stage6_data['selected_supplier']}")
        print(f"  • Contract Value: {stage6_data['contract_value']}")
        print(f"  • Status: ✅ COMPLETED - All 6 stages finished")
        print(f"\n🌐 View workflow at:")
        print(f"   http://localhost:3000/workflows/{workflow_id}")
        print()

        return True

if __name__ == "__main__":
    workflow_id = "065b4298-5b8e-4122-b325-b7cb798c7f41"
    success = populate_workflow(workflow_id)
    sys.exit(0 if success else 1)
