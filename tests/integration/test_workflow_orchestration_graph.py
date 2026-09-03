"""Integration tests — the LangGraph StateGraph itself (state.py/graph.py).

Requires Python >=3.10 (langgraph's floor) - run with a separate venv from
the rest of tests/integration/, e.g.:
    /path/to/py311venv/bin/python3 -m pytest tests/integration/test_workflow_orchestration_graph.py

Exercises supplier_discovery and qualification_audit/selection_tasks end
to end (real DB writes via their already-tested _core functions) since
those three don't lazily import the app.py monolith - document_analysis
and rfq_outreach do (see agents/email_outreach/service.py's docstring)
and are covered at the _core level in test_workflow_orchestration_core.py
instead. Uses an in-memory checkpointer here (interrupt/resume semantics
already verified against a real PostgresSaver+ConnectionPool by hand -
see graph.py's get_compiled_graph() docstring) to keep this fast and
independent of Postgres checkpoint-table setup.
"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from core.database import db
from core.models import Project, Team
from models.workflow import WorkflowInstance, WorkflowTemplate


@pytest.fixture
def project(flask_app):
    with flask_app.app_context():
        team = Team(team_id="team-graph-1", owner_id="user_graph", name="Graph Test Team")
        proj = Project(project_id="proj-graph-1", team_id="team-graph-1",
                        owner_id="user_graph", name="Graph Test Project")
        db.session.add(team)
        db.session.add(proj)
        db.session.commit()
        yield proj.project_id
        Project.query.filter_by(project_id="proj-graph-1").delete()
        Team.query.filter_by(team_id="team-graph-1").delete()
        db.session.commit()


@pytest.fixture
def template(flask_app):
    with flask_app.app_context():
        tpl = WorkflowTemplate.query.filter_by(template_id="supplier-qualification").first()
        if not tpl:
            tpl = WorkflowTemplate(template_id="supplier-qualification", name="Supplier Qualification Pipeline",
                                    is_system=True, is_active=True)
            tpl.stages = [
                {"id": "supplier_discovery", "agent": "market_research"},
                {"id": "document_analysis", "agent": "data_insights"},
                {"id": "rfq_outreach", "agent": "email_outreach"},
                {"id": "response_analysis", "agent": "sales_helper"},
                {"id": "qualification_audit", "agent": "supply_chain"},
                {"id": "selection_tasks", "agent": "executive_assistant"},
            ]
            db.session.add(tpl)
            db.session.commit()
        yield tpl.template_id


@pytest.fixture
def instance(flask_app, project, template):
    with flask_app.app_context():
        inst = WorkflowInstance(
            instance_id="wf-instance-graph-1",
            template_id=template,
            user_id="user_graph",
            project_id=project,
            name="Graph test run",
            status="pending",
            current_stage_index=0,
            autonomy_mode="autopilot",
        )
        db.session.add(inst)
        db.session.commit()
        yield inst.instance_id
        WorkflowInstance.query.filter_by(instance_id="wf-instance-graph-1").delete()
        db.session.commit()


def test_build_graph_structure():
    """Pure LangGraph wiring check - no DB, no backend deps beyond state.py."""
    from agents.workflow_orchestration.graph import STAGE_ORDER, build_graph

    compiled = build_graph().compile(checkpointer=InMemorySaver())
    graph_repr = compiled.get_graph()
    node_names = set(graph_repr.nodes.keys())
    for stage_id in STAGE_ORDER:
        assert stage_id in node_names


def test_qualification_audit_and_selection_tasks_autopilot(flask_app, instance, project):
    """Runs just the last two stages directly (not via the full 6-stage
    graph, to avoid needing supplier_discovery's real Google Places call)
    to prove run_stage()'s autopilot dispatch + _sync_legacy_state's
    dual-write both work against a real WorkflowInstance row."""
    from agents.workflow_orchestration.graph import (
        qualification_audit_node,
        selection_tasks_node,
    )

    with flask_app.app_context():
        from agents.supply_chain.models import SCSupplier

        supplier = SCSupplier(
            supplier_id="supplier-graph-1", project_id=project,
            user_id="user_graph", name="Acme Graph Supplier",
        )
        db.session.add(supplier)
        db.session.commit()

        state = {
            "instance_id": instance,
            "user_id": "user_graph",
            "project_id": project,
            "current_stage_id": "",
            "initial_inputs": {
                "qualification_audit": {"audits": [{"supplier_id": "supplier-graph-1", "score": 85}]},
            },
            "stage_outputs": {},
            "autonomy_mode": "autopilot",
            "errors": [],
        }

        audit_update = qualification_audit_node(state)
        assert audit_update["errors"] == []
        audited = audit_update["stage_outputs"]["qualification_audit"]["audited"]
        assert audited[0]["result"]["auditStatus"] == "passed"

        state["stage_outputs"].update(audit_update["stage_outputs"])

        tasks_update = selection_tasks_node(state)
        assert tasks_update["errors"] == []
        created = tasks_update["stage_outputs"]["selection_tasks"]["created"]
        assert len(created) == 1
        assert created[0]["result"]["title"] == "Follow up with supplier Acme Graph Supplier"

        # Dual-write into the legacy flat columns must be keyed by the
        # actual stage_id each node ran, not by call order - these two
        # stages are indices 4 and 5 in STAGE_ORDER, so current_stage_index
        # (the furthest stage reached) ends at 6 even though only 2 of the
        # 6 nodes actually ran in this test.
        refreshed = WorkflowInstance.query.filter_by(instance_id=instance).first()
        assert refreshed.current_stage_index == 6
        assert "qualification_audit" in refreshed.stage_states
        assert "selection_tasks" in refreshed.stage_states
        assert "supplier_discovery" not in refreshed.stage_states

        SCSupplier.query.filter_by(supplier_id="supplier-graph-1").delete()
        db.session.commit()


def test_qualification_audit_interrupt_and_resume(flask_app, instance, project):
    """Co-pilot mode: the node must pause via interrupt() before auditing,
    then apply the human's edited score on resume."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from agents.workflow_orchestration.graph import qualification_audit_node

    with flask_app.app_context():
        from agents.supply_chain.models import SCSupplier

        supplier = SCSupplier(
            supplier_id="supplier-graph-2", project_id=project,
            user_id="user_graph", name="Acme Graph Supplier 2",
        )
        db.session.add(supplier)
        db.session.commit()

        class S(TypedDict):
            instance_id: str
            user_id: str
            project_id: str
            current_stage_id: str
            initial_inputs: dict
            stage_outputs: dict
            autonomy_mode: str
            errors: list

        g = StateGraph(S)
        g.add_node("qualification_audit", qualification_audit_node)
        g.set_entry_point("qualification_audit")
        g.add_edge("qualification_audit", END)
        compiled = g.compile(checkpointer=InMemorySaver())

        config = {"configurable": {"thread_id": f"{instance}-copilot"}}
        initial_state = {
            "instance_id": instance,
            "user_id": "user_graph",
            "project_id": project,
            "current_stage_id": "",
            "initial_inputs": {
                "qualification_audit": {"audits": [{"supplier_id": "supplier-graph-2", "score": 40}]},
            },
            "stage_outputs": {},
            "autonomy_mode": "co-pilot",
            "errors": [],
        }

        result = compiled.invoke(initial_state, config=config)
        assert "__interrupt__" in result
        proposed = result["__interrupt__"][0].value["proposed_input"]
        assert proposed["audits"][0]["score"] == 40

        resumed = compiled.invoke(
            Command(resume={"action": "edit", "data": {"audits": [{"supplier_id": "supplier-graph-2", "score": 95}]}}),
            config=config,
        )
        audited = resumed["stage_outputs"]["qualification_audit"]["audited"]
        assert audited[0]["result"]["score"] == 95
        assert audited[0]["result"]["auditStatus"] == "passed"

        SCSupplier.query.filter_by(supplier_id="supplier-graph-2").delete()
        db.session.commit()
