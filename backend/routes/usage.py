"""
AI Usage & Cost Rollup API - individual, project, and team level views over
core/models.py's AIUsageLog, which core/ai_client.py writes to on every LLM
call (see that module for the token/cost estimation and key-resolution
logic this data is a byproduct of).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from flask import Blueprint, g, jsonify, request
from sqlalchemy import func

from core.auth import require_auth, user_can_access_project
from core.database import db
from core.models import AIUsageLog, TeamMember

usage_bp = Blueprint('usage', __name__)


def _since(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=max(1, min(days, 365)))


def _summarize(base_query, group_by_user: bool = False) -> dict:
    """Aggregates an AIUsageLog query into totals + breakdowns by agent,
    model, day, and (optionally) user - all via SQL GROUP BY rather than
    pulling every row into Python."""
    totals = base_query.with_entities(
        func.coalesce(func.sum(AIUsageLog.prompt_tokens), 0),
        func.coalesce(func.sum(AIUsageLog.completion_tokens), 0),
        func.coalesce(func.sum(AIUsageLog.estimated_cost_usd), 0.0),
        func.count(AIUsageLog.id),
    ).first()
    prompt_tokens, completion_tokens, total_cost, request_count = totals

    by_agent = base_query.with_entities(
        AIUsageLog.agent,
        func.sum(AIUsageLog.total_tokens),
        func.sum(AIUsageLog.estimated_cost_usd),
        func.count(AIUsageLog.id),
    ).group_by(AIUsageLog.agent).order_by(func.sum(AIUsageLog.estimated_cost_usd).desc()).all()

    by_model = base_query.with_entities(
        AIUsageLog.model,
        func.sum(AIUsageLog.total_tokens),
        func.sum(AIUsageLog.estimated_cost_usd),
        func.count(AIUsageLog.id),
    ).group_by(AIUsageLog.model).order_by(func.sum(AIUsageLog.estimated_cost_usd).desc()).all()

    by_day = base_query.with_entities(
        func.date(AIUsageLog.created_at),
        func.sum(AIUsageLog.total_tokens),
        func.sum(AIUsageLog.estimated_cost_usd),
    ).group_by(func.date(AIUsageLog.created_at)).order_by(func.date(AIUsageLog.created_at)).all()

    result = {
        'totalTokens': int(prompt_tokens + completion_tokens),
        'promptTokens': int(prompt_tokens),
        'completionTokens': int(completion_tokens),
        'totalCostUsd': round(float(total_cost), 6),
        'requestCount': int(request_count),
        'byAgent': [
            {'agent': agent, 'tokens': int(tokens), 'costUsd': round(float(cost), 6), 'requestCount': int(count)}
            for agent, tokens, cost, count in by_agent
        ],
        'byModel': [
            {'model': model, 'tokens': int(tokens), 'costUsd': round(float(cost), 6), 'requestCount': int(count)}
            for model, tokens, cost, count in by_model
        ],
        'byDay': [
            {'date': str(day), 'tokens': int(tokens), 'costUsd': round(float(cost), 6)}
            for day, tokens, cost in by_day
        ],
    }

    if group_by_user:
        by_user = base_query.with_entities(
            AIUsageLog.user_id,
            func.sum(AIUsageLog.total_tokens),
            func.sum(AIUsageLog.estimated_cost_usd),
            func.count(AIUsageLog.id),
        ).group_by(AIUsageLog.user_id).order_by(func.sum(AIUsageLog.estimated_cost_usd).desc()).all()
        result['byUser'] = [
            {'userId': user_id, 'tokens': int(tokens), 'costUsd': round(float(cost), 6), 'requestCount': int(count)}
            for user_id, tokens, cost, count in by_user
        ]

    return result


@usage_bp.route('/api/usage/me', methods=['GET'])
@require_auth
def get_my_usage():
    """Current user's own AI usage/cost, across every project and agent."""
    days = request.args.get('days', default=30, type=int)
    query = AIUsageLog.query.filter(
        AIUsageLog.user_id == g.user_id,
        AIUsageLog.created_at >= _since(days),
    )
    return jsonify({'success': True, 'days': days, 'usage': _summarize(query)})


@usage_bp.route('/api/projects/<project_id>/usage', methods=['GET'])
@require_auth
def get_project_usage(project_id):
    """AI usage/cost for a project - visible to any project member, same
    visibility rule as the project's AI key settings (members should be
    able to see what their work is costing even if they can't change the
    key)."""
    if not user_can_access_project(g.user_id, project_id):
        return jsonify({'error': 'Project not found'}), 404

    days = request.args.get('days', default=30, type=int)
    query = AIUsageLog.query.filter(
        AIUsageLog.project_id == project_id,
        AIUsageLog.created_at >= _since(days),
    )
    return jsonify({'success': True, 'days': days, 'usage': _summarize(query, group_by_user=True)})


@usage_bp.route('/api/team/usage', methods=['GET'])
@require_auth
def get_team_usage():
    """AI usage/cost across the current user's team. Owner/admin only -
    this rolls up every member's spend, which is billing-sensitive in a
    way individual/project views aren't."""
    member = TeamMember.query.filter_by(user_id=g.user_id).first()
    if not member:
        return jsonify({'error': 'No team found for this user'}), 404
    if member.role not in ('owner', 'admin'):
        return jsonify({'error': 'Only the team owner or an admin can view team-wide usage'}), 403

    days = request.args.get('days', default=30, type=int)
    query = AIUsageLog.query.filter(
        AIUsageLog.team_id == member.team_id,
        AIUsageLog.created_at >= _since(days),
    )
    return jsonify({'success': True, 'days': days, 'usage': _summarize(query, group_by_user=True)})
