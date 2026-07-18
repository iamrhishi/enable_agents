"""
Foundation Test Suite - Team/Project Persistence

Tests:
1. Database migration/creation
2. Team CRUD operations
3. Project CRUD operations
4. Member management
5. Agent data storage
"""

import os
import sys
import json

# Use PostgreSQL (same as Docker setup)
os.environ.setdefault('DATABASE_URI', 'postgresql+psycopg2://enable_agents:enable_agents@localhost:5432/enable_agents')

# Dummy AWS credentials to prevent boto3 auth errors on import
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'testing')
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')

from app import app, db
from core.models import Team, TeamMember, Project, PendingInvite

def test_database_creation():
    """Test 1: Database tables created successfully."""
    print("\n=== Test 1: Database Creation ===")
    with app.app_context():
        db.create_all()

        # Verify tables exist
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        required = ['teams', 'team_members', 'projects', 'pending_invites']
        for t in required:
            assert t in tables, f"Missing table: {t}"
            print(f"  ✓ Table '{t}' exists")

        print("  ✓ All required tables created")
    return True


def test_team_creation():
    """Test 2: Team CRUD via API."""
    print("\n=== Test 2: Team API ===")

    with app.test_client() as client:
        # GET /api/team - should create team for new user
        resp = client.get('/api/team', headers={'X-User-Id': 'test@example.com'})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.get_json()

        assert 'team_id' in data, "Missing team_id in response"
        assert len(data['members']) == 1, "Should have 1 member (owner)"
        assert data['members'][0]['role'] == 'owner', "First member should be owner"
        print(f"  ✓ Team created: {data['team_id'][:8]}...")
        print(f"  ✓ Owner added: {data['members'][0]['email']}")

        team_id = data['team_id']

        # GET again - should return same team
        resp2 = client.get('/api/team', headers={'X-User-Id': 'test@example.com'})
        data2 = resp2.get_json()
        assert data2['team_id'] == team_id, "Should return same team"
        print("  ✓ Team persisted correctly")

    return True


def test_member_invite():
    """Test 3: Member invitation."""
    print("\n=== Test 3: Member Invitation ===")

    with app.test_client() as client:
        # Setup - get team
        client.get('/api/team', headers={'X-User-Id': 'owner@example.com'})

        # Invite member
        resp = client.post('/api/team/invite',
            headers={'X-User-Id': 'owner@example.com', 'Content-Type': 'application/json'},
            data=json.dumps({'email': 'member@example.com', 'role': 'member'})
        )
        assert resp.status_code == 200, f"Invite failed: {resp.get_json()}"
        print("  ✓ Member invited successfully")

        # Verify member added
        resp2 = client.get('/api/team', headers={'X-User-Id': 'owner@example.com'})
        data = resp2.get_json()
        assert len(data['members']) == 2, f"Should have 2 members, got {len(data['members'])}"

        emails = [m['email'] for m in data['members']]
        assert 'member@example.com' in emails, "New member not in team"
        print("  ✓ Member appears in team roster")

        # Try duplicate invite
        resp3 = client.post('/api/team/invite',
            headers={'X-User-Id': 'owner@example.com', 'Content-Type': 'application/json'},
            data=json.dumps({'email': 'member@example.com', 'role': 'member'})
        )
        assert resp3.status_code == 400, "Duplicate invite should fail"
        print("  ✓ Duplicate invite rejected")

    return True


def test_project_crud():
    """Test 4: Project CRUD operations."""
    print("\n=== Test 4: Project CRUD ===")

    with app.test_client() as client:
        user = 'projowner@example.com'

        # Create project
        resp = client.post('/api/projects',
            headers={'X-User-Id': user, 'Content-Type': 'application/json'},
            data=json.dumps({
                'name': 'Test Project',
                'description': 'Testing persistence',
                'agents': ['market_research', 'content_marketing']
            })
        )
        assert resp.status_code == 200, f"Create failed: {resp.get_json()}"
        data = resp.get_json()
        project_id = data['project']['id']
        print(f"  ✓ Project created: {project_id[:8]}...")

        # Read project
        resp2 = client.get(f'/api/projects/{project_id}', headers={'X-User-Id': user})
        assert resp2.status_code == 200
        proj = resp2.get_json()['project']
        assert proj['name'] == 'Test Project'
        assert 'market_research' in proj['agents']
        print("  ✓ Project retrieved correctly")

        # Update project
        resp3 = client.put(f'/api/projects/{project_id}',
            headers={'X-User-Id': user, 'Content-Type': 'application/json'},
            data=json.dumps({'name': 'Updated Project', 'status': 'active'})
        )
        assert resp3.status_code == 200
        assert resp3.get_json()['project']['name'] == 'Updated Project'
        print("  ✓ Project updated")

        # List projects
        resp4 = client.get('/api/projects', headers={'X-User-Id': user})
        assert resp4.status_code == 200
        projects = resp4.get_json()['projects']
        assert len(projects) >= 1
        print(f"  ✓ Listed {len(projects)} project(s)")

        # Delete project
        resp5 = client.delete(f'/api/projects/{project_id}', headers={'X-User-Id': user})
        assert resp5.status_code == 200
        print("  ✓ Project deleted")

        # Verify deleted
        resp6 = client.get(f'/api/projects/{project_id}', headers={'X-User-Id': user})
        assert resp6.status_code == 404
        print("  ✓ Deletion verified")

    return True


def test_agent_data_storage():
    """Test 5: Agent-specific data in projects."""
    print("\n=== Test 5: Agent Data Storage ===")

    with app.test_client() as client:
        user = 'agentdata@example.com'

        # Create project with agents
        resp = client.post('/api/projects',
            headers={'X-User-Id': user, 'Content-Type': 'application/json'},
            data=json.dumps({
                'name': 'Agent Data Test',
                'agents': ['market_research', 'email_outreach']
            })
        )
        project_id = resp.get_json()['project']['id']

        # Store agent-specific data
        agent_data = {
            'company_profile': {'name': 'Acme Corp', 'industry': 'Tech'},
            'prospect_list': [{'email': 'lead@company.com', 'score': 85}]
        }

        resp2 = client.put(f'/api/projects/{project_id}/data',
            headers={'X-User-Id': user, 'Content-Type': 'application/json'},
            data=json.dumps({'agent': 'market_research', 'data': agent_data})
        )
        assert resp2.status_code == 200
        print("  ✓ Agent data stored")

        # Retrieve and verify
        resp3 = client.get(f'/api/projects/{project_id}', headers={'X-User-Id': user})
        proj = resp3.get_json()['project']

        assert 'market_research' in proj['data']
        assert proj['data']['market_research']['company_profile']['name'] == 'Acme Corp'
        print("  ✓ Agent data retrieved correctly")

        # Try storing data for non-enabled agent
        resp4 = client.put(f'/api/projects/{project_id}/data',
            headers={'X-User-Id': user, 'Content-Type': 'application/json'},
            data=json.dumps({'agent': 'content_marketing', 'data': {'test': 1}})
        )
        assert resp4.status_code == 403, "Should reject non-enabled agent"
        print("  ✓ Non-enabled agent data rejected")

    return True


def test_permission_checks():
    """Test 6: Permission/authorization checks."""
    print("\n=== Test 6: Permission Checks ===")

    with app.test_client() as client:
        # No auth header
        resp = client.get('/api/team')
        assert resp.status_code == 401
        print("  ✓ No auth rejected (team)")

        resp2 = client.get('/api/projects')
        assert resp2.status_code == 401
        print("  ✓ No auth rejected (projects)")

        # Create project as owner
        owner = 'permowner@example.com'
        other = 'other@example.com'

        resp3 = client.post('/api/projects',
            headers={'X-User-Id': owner, 'Content-Type': 'application/json'},
            data=json.dumps({'name': 'Owner Project', 'agents': ['market_research']})
        )
        project_id = resp3.get_json()['project']['id']

        # Other user tries to delete
        resp4 = client.delete(f'/api/projects/{project_id}', headers={'X-User-Id': other})
        assert resp4.status_code == 403, "Non-owner delete should be forbidden"
        print("  ✓ Non-owner delete rejected")

        # Member invite by non-admin
        client.get('/api/team', headers={'X-User-Id': owner})
        client.post('/api/team/invite',
            headers={'X-User-Id': owner, 'Content-Type': 'application/json'},
            data=json.dumps({'email': 'viewer@example.com', 'role': 'viewer'})
        )

        # Viewer tries to invite
        resp5 = client.post('/api/team/invite',
            headers={'X-User-Id': 'viewer@example.com', 'Content-Type': 'application/json'},
            data=json.dumps({'email': 'newuser@example.com', 'role': 'member'})
        )
        assert resp5.status_code == 403, "Viewer invite should be forbidden"
        print("  ✓ Viewer invite rejected")

    return True


def test_role_management():
    """Test 7: Role updates."""
    print("\n=== Test 7: Role Management ===")

    with app.test_client() as client:
        owner = 'roleowner@example.com'

        # Setup team with member
        client.get('/api/team', headers={'X-User-Id': owner})
        resp = client.post('/api/team/invite',
            headers={'X-User-Id': owner, 'Content-Type': 'application/json'},
            data=json.dumps({'email': 'tomember@example.com', 'role': 'member'})
        )
        member_id = resp.get_json()['member']['id']

        # Update role to admin
        resp2 = client.put(f'/api/team/members/{member_id}/role',
            headers={'X-User-Id': owner, 'Content-Type': 'application/json'},
            data=json.dumps({'role': 'admin'})
        )
        assert resp2.status_code == 200
        assert resp2.get_json()['member']['role'] == 'admin'
        print("  ✓ Role updated to admin")

        # Non-owner tries to change role
        resp3 = client.put(f'/api/team/members/{member_id}/role',
            headers={'X-User-Id': 'tomember@example.com', 'Content-Type': 'application/json'},
            data=json.dumps({'role': 'viewer'})
        )
        assert resp3.status_code == 403
        print("  ✓ Non-owner role change rejected")

    return True


def cleanup():
    """Clean up test data from PostgreSQL."""
    print("\n✓ Tests complete (using shared PostgreSQL database)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("FOUNDATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Database Creation", test_database_creation),
        ("Team API", test_team_creation),
        ("Member Invitation", test_member_invite),
        ("Project CRUD", test_project_crud),
        ("Agent Data Storage", test_agent_data_storage),
        ("Permission Checks", test_permission_checks),
        ("Role Management", test_role_management),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)

    cleanup()
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
