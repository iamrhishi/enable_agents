"""
Projects API Routes - Platform-wide project management.

Projects are shared workspaces that can be accessed by multiple agents and team members.
Uses SQLAlchemy models for persistence.
"""

from flask import Blueprint, request, jsonify, g
from datetime import datetime
import uuid

from core.auth import require_auth, user_can_access_project
from core.database import db
from core.models import Project, Team, TeamMember

projects_bp = Blueprint('projects', __name__)


def get_user_team(user_email: str) -> Team:
    """Get user's team, creating one if needed."""
    member = TeamMember.query.filter_by(user_id=user_email).first()
    if member:
        return member.team

    # Create new team with user as owner
    team_id = str(uuid.uuid4())
    team = Team(
        team_id=team_id,
        owner_id=user_email,
        name=f"{user_email.split('@')[0]}'s Team"
    )
    db.session.add(team)

    member = TeamMember(
        member_id=str(uuid.uuid4()),
        team_id=team_id,
        user_id=user_email,
        name=user_email.split('@')[0],
        role='owner'
    )
    db.session.add(member)
    db.session.commit()
    return team


def get_user_projects(user_email: str):
    """Get all projects for a user (via their team)."""
    team = get_user_team(user_email)
    return Project.query.filter_by(team_id=team.team_id).all()


@projects_bp.route('/api/projects', methods=['GET'])
@require_auth
def list_projects():
    """List all projects for the current user."""
    user_email = g.user_id
    projects = get_user_projects(user_email)
    return jsonify({'success': True, 'projects': [p.to_dict() for p in projects]})


@projects_bp.route('/api/projects', methods=['POST'])
@require_auth
def create_project():
    """Create a new project."""
    user_email = g.user_id
    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    # Default to all agents if none specified
    agents = data.get('agents') or [
        'marketResearch', 'salesHelper', 'contentMarketing',
        'communityNetwork', 'eventNetworking', 'executiveAssistant', 'dataInsights'
    ]

    if not name:
        return jsonify({'error': 'Project name is required'}), 400

    team = get_user_team(user_email)
    project = Project(
        project_id=str(uuid.uuid4()),
        team_id=team.team_id,
        owner_id=user_email,
        name=name,
        description=description,
    )
    project.agents = agents
    project.data = {}

    db.session.add(project)
    db.session.commit()
    return jsonify({'success': True, 'project': project.to_dict()})


@projects_bp.route('/api/projects/<project_id>', methods=['GET'])
@require_auth
def get_project(project_id):
    """Get a specific project by ID."""
    if not user_can_access_project(g.user_id, project_id):
        return jsonify({'error': 'Project not found'}), 404

    project = Project.query.filter_by(project_id=project_id).first()
    return jsonify({'success': True, 'project': project.to_dict()})


@projects_bp.route('/api/projects/<project_id>', methods=['PUT'])
@require_auth
def update_project(project_id):
    """Update a project."""
    if not user_can_access_project(g.user_id, project_id):
        return jsonify({'error': 'Project not found'}), 404

    project = Project.query.filter_by(project_id=project_id).first()

    data = request.get_json()
    if 'name' in data:
        project.name = data['name'].strip()
    if 'description' in data:
        project.description = data['description'].strip()
    if 'agents' in data:
        project.agents = data['agents']
    if 'status' in data and data['status'] in ['active', 'archived', 'completed']:
        project.status = data['status']
    if 'data' in data:
        current = project.data
        current.update(data['data'])
        project.data = current

    project.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'project': project.to_dict()})


@projects_bp.route('/api/projects/<project_id>', methods=['DELETE'])
@require_auth
def delete_project(project_id):
    """Delete a project (owner only)."""
    project = Project.query.filter_by(project_id=project_id).first()
    if not project or not user_can_access_project(g.user_id, project_id):
        return jsonify({'error': 'Project not found'}), 404
    if project.owner_id != g.user_id:
        return jsonify({'error': 'Only project owner can delete'}), 403

    db.session.delete(project)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Project deleted'})


@projects_bp.route('/api/projects/<project_id>/data', methods=['PUT'])
@require_auth
def update_project_data(project_id):
    """Update agent-specific data within a project."""
    if not user_can_access_project(g.user_id, project_id):
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json()
    agent_key = data.get('agent')
    agent_data = data.get('data', {})

    if not agent_key:
        return jsonify({'error': 'Agent key is required'}), 400

    project = Project.query.filter_by(project_id=project_id).first()
    if agent_key not in project.agents:
        return jsonify({'error': f'Agent {agent_key} not enabled for this project'}), 403

    project.set_agent_data(agent_key, agent_data)
    project.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'project': project.to_dict()})
