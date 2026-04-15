#!/usr/bin/env python3
"""
Test script to verify CRUD API integration
Validates that all endpoints are working correctly
"""

import requests
import json
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = 'http://localhost:5000/api'
TIMEOUT = 5

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_error(message):
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

def test_connection():
    """Test basic connectivity to the API"""
    print(f"\n{Colors.BLUE}=== Testing API Connection ==={Colors.RESET}")
    try:
        response = requests.get(f'{API_BASE_URL}/projects', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"API is accessible at {API_BASE_URL}")
            return True
        else:
            print_error(f"API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Make sure Flask server is running on port 5000")
        return False
    except Exception as e:
        print_error(f"Connection test failed: {str(e)}")
        return False

def test_projects():
    """Test Projects CRUD operations"""
    print(f"\n{Colors.BLUE}=== Testing Projects API ==={Colors.RESET}")
    
    try:
        # Test GET all
        print_info("Testing GET /api/projects")
        response = requests.get(f'{API_BASE_URL}/projects', timeout=TIMEOUT)
        if response.status_code == 200:
            projects = response.json()
            print_success(f"Retrieved {len(projects)} projects")
        else:
            print_error(f"GET failed with status {response.status_code}")
            return False

        # Test POST create
        print_info("Testing POST /api/projects")
        test_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        project_data = {
            'project_name': f'Test Project {datetime.now().timestamp()}',
            'project_description': 'Test project for CRUD validation',
            'due_date': test_date,
            'is_active': True
        }
        response = requests.post(f'{API_BASE_URL}/projects', json=project_data, timeout=TIMEOUT)
        if response.status_code == 201:
            created_project = response.json()
            project_id = created_project['id']
            print_success(f"Created project with ID {project_id}")
        else:
            print_error(f"POST failed with status {response.status_code}")
            return False

        # Test GET single
        print_info(f"Testing GET /api/projects/{project_id}")
        response = requests.get(f'{API_BASE_URL}/projects/{project_id}', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Retrieved project {project_id}")
        else:
            print_error(f"GET single failed with status {response.status_code}")
            return False

        # Test PUT update
        print_info(f"Testing PUT /api/projects/{project_id}")
        updated_data = project_data.copy()
        updated_data['project_name'] = f'Updated {updated_data["project_name"]}'
        response = requests.put(f'{API_BASE_URL}/projects/{project_id}', json=updated_data, timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Updated project {project_id}")
        else:
            print_error(f"PUT failed with status {response.status_code}")
            return False

        # Test DELETE
        print_info(f"Testing DELETE /api/projects/{project_id}")
        response = requests.delete(f'{API_BASE_URL}/projects/{project_id}', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Deleted project {project_id}")
        else:
            print_error(f"DELETE failed with status {response.status_code}")
            return False

        return True
    except Exception as e:
        print_error(f"Projects test failed: {str(e)}")
        return False

def test_people():
    """Test People CRUD operations"""
    print(f"\n{Colors.BLUE}=== Testing People API ==={Colors.RESET}")
    
    try:
        # Test GET all
        print_info("Testing GET /api/people")
        response = requests.get(f'{API_BASE_URL}/people', timeout=TIMEOUT)
        if response.status_code == 200:
            people = response.json()
            print_success(f"Retrieved {len(people)} people")
        else:
            print_error(f"GET failed with status {response.status_code}")
            return False

        # Test POST create
        print_info("Testing POST /api/people")
        person_data = {
            'full_name': f'Test Person {datetime.now().timestamp()}',
            'email': f'test{datetime.now().timestamp()}@example.com',
            'phone_number': '+1-555-0123',
            'whatsapp_number': '+1-555-0123',
            'role': 'Test Role'
        }
        response = requests.post(f'{API_BASE_URL}/people', json=person_data, timeout=TIMEOUT)
        if response.status_code == 201:
            created_person = response.json()
            person_id = created_person['id']
            print_success(f"Created person with ID {person_id}")
        else:
            print_error(f"POST failed with status {response.status_code}: {response.text}")
            return False

        # Test GET single
        print_info(f"Testing GET /api/people/{person_id}")
        response = requests.get(f'{API_BASE_URL}/people/{person_id}', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Retrieved person {person_id}")
        else:
            print_error(f"GET single failed with status {response.status_code}")
            return False

        # Test PUT update
        print_info(f"Testing PUT /api/people/{person_id}")
        updated_data = person_data.copy()
        updated_data['role'] = 'Updated Role'
        response = requests.put(f'{API_BASE_URL}/people/{person_id}', json=updated_data, timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Updated person {person_id}")
        else:
            print_error(f"PUT failed with status {response.status_code}")
            return False

        # Test DELETE
        print_info(f"Testing DELETE /api/people/{person_id}")
        response = requests.delete(f'{API_BASE_URL}/people/{person_id}', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Deleted person {person_id}")
        else:
            print_error(f"DELETE failed with status {response.status_code}")
            return False

        return True
    except Exception as e:
        print_error(f"People test failed: {str(e)}")
        return False

def test_tasks():
    """Test Tasks CRUD operations"""
    print(f"\n{Colors.BLUE}=== Testing Tasks API ==={Colors.RESET}")
    
    try:
        # First, create a project for the task
        print_info("Creating project for task testing...")
        test_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        project_data = {
            'project_name': f'Test Project for Tasks {datetime.now().timestamp()}',
            'project_description': 'Temporary project for task testing',
            'due_date': test_date,
            'is_active': True
        }
        project_response = requests.post(f'{API_BASE_URL}/projects', json=project_data, timeout=TIMEOUT)
        if project_response.status_code != 201:
            print_error("Failed to create project for task testing")
            return False
        project_id = project_response.json()['id']
        print_success(f"Created project {project_id} for testing")

        # Test GET all
        print_info("Testing GET /api/tasks")
        response = requests.get(f'{API_BASE_URL}/tasks', timeout=TIMEOUT)
        if response.status_code == 200:
            tasks = response.json()
            print_success(f"Retrieved {len(tasks)} tasks")
        else:
            print_error(f"GET failed with status {response.status_code}")
            return False

        # Test POST create
        print_info("Testing POST /api/tasks")
        task_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        task_data = {
            'project_id': project_id,
            'task_title': f'Test Task {datetime.now().timestamp()}',
            'task_description': 'Test task for CRUD validation',
            'assigned_to': '',
            'due_date': task_date,
            'priority': 'Medium',
            'status': 'Pending'
        }
        response = requests.post(f'{API_BASE_URL}/tasks', json=task_data, timeout=TIMEOUT)
        if response.status_code == 201:
            created_task = response.json()
            task_id = created_task['id']
            print_success(f"Created task with ID {task_id}")
        else:
            print_error(f"POST failed with status {response.status_code}: {response.text}")
            return False

        # Test GET single
        print_info(f"Testing GET /api/tasks/{task_id}")
        response = requests.get(f'{API_BASE_URL}/tasks/{task_id}', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Retrieved task {task_id}")
        else:
            print_error(f"GET single failed with status {response.status_code}")
            return False

        # Test PUT update
        print_info(f"Testing PUT /api/tasks/{task_id}")
        updated_data = task_data.copy()
        updated_data['status'] = 'In Progress'
        response = requests.put(f'{API_BASE_URL}/tasks/{task_id}', json=updated_data, timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Updated task {task_id}")
        else:
            print_error(f"PUT failed with status {response.status_code}")
            return False

        # Test DELETE
        print_info(f"Testing DELETE /api/tasks/{task_id}")
        response = requests.delete(f'{API_BASE_URL}/tasks/{task_id}', timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Deleted task {task_id}")
        else:
            print_error(f"DELETE failed with status {response.status_code}")
            return False

        # Cleanup: delete the test project
        requests.delete(f'{API_BASE_URL}/projects/{project_id}', timeout=TIMEOUT)

        return True
    except Exception as e:
        print_error(f"Tasks test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BLUE}{'='*50}")
    print(f"CRUD API Integration Test Suite")
    print(f"{'='*50}{Colors.RESET}\n")
    
    results = []
    
    # Test connection first
    if not test_connection():
        print_error("\nCannot proceed - API is not accessible")
        return 1
    
    # Run all tests
    results.append(("Projects", test_projects()))
    results.append(("People", test_people()))
    results.append(("Tasks", test_tasks()))
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*50}")
    print(f"Test Summary")
    print(f"{'='*50}{Colors.RESET}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{symbol} {test_name}: {status}{Colors.RESET}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.RESET}\n")
    
    if passed == total:
        print_success("All tests passed! CRUD integration is working correctly.")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit(main())
