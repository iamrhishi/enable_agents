# Executive Assistant Agent - Quick Start Guide

## Prerequisites

- Python 3.7+
- Node.js 14+ and npm
- MySQL 5.7+
- Git (optional)

## Step-by-Step Setup

### 1. Initialize Database

```bash
cd /Users/rhishikeshthakur/Enable/Software_Development/enable_agents

# Initialize the database schema
python tools/migrations.py init

# (Optional) Load sample data
python tools/migrations.py sample-data

# Verify database setup
python tools/migrations.py status
```

Expected output for status:
- Database: enable_agents
- Tables: users, people, projects, tasks
- Records: Sample data populated

### 2. Start Backend (Flask)

```bash
# From the workspace root
python tools/app.py
```

Expected output:
```
 * Running on http://localhost:5000
 * Debug mode: off
```

The API will be available at: `http://localhost:5000/api`

### 3. Start Frontend (React)

```bash
# From the workspace root
cd agent-app
npm install  # (only needed first time)
npm start
```

Expected output:
```
Compiled successfully!

You can now view agent-app in the browser.
  Local:            http://localhost:3000
  On Your Network:  http://192.168...
```

### 4. Access the Application

Open your browser and navigate to:
```
http://localhost:3000/executive-assistant
```

## Quick Test

### Method 1: Browser Testing (Recommended)

1. Open the Executive Assistant page
2. Click **Projects** tab → Click **+ Add Project**
3. Fill in form:
   - Project Name: "Test Project"
   - Description: "My first project"
   - Due Date: (pick a date)
   - Active: (checked)
4. Click **Save Project**
5. Verify the project appears in the list

### Method 2: Command Line Testing

```bash
# Run the automated test suite
python tools/test_crud_api.py
```

Expected output:
```
==================================================
CRUD API Integration Test Suite
==================================================

ℹ Testing API Connection
✓ API is accessible at http://localhost:5000/api

=== Testing Projects API ===
ℹ Testing GET /api/projects
✓ Retrieved X projects
...

=== Test Summary ===
✓ Projects: PASSED
✓ People: PASSED
✓ Tasks: PASSED

Total: 3/3 tests passed
✓ All tests passed! CRUD integration is working correctly.
```

## Common Issues & Solutions

### Issue: "Cannot connect to API"
```
✗ Cannot connect to API. Make sure Flask server is running on port 5000
```
**Solution:**
- Make sure Flask is running: `python tools/app.py`
- Check port 5000 is not in use: `lsof -i :5000`
- If port in use, kill process: `kill -9 <PID>`

### Issue: "Database table does not exist"
```
sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (1146, "Table 'enable_agents.projects' doesn't exist")
```
**Solution:**
- Run migrations: `python tools/migrations.py init`
- Check MySQL is running

### Issue: "Blank page in browser"
```
Errors in console showing 404 or connection refused
```
**Solution:**
- Verify backend is running and accessible at `http://localhost:5000/api/projects`
- Check browser console (F12) for specific errors
- Verify API_BASE_URL in executiveAssistantAPI.js is correct

### Issue: "Email already exists"
```
Error: Email already exists
```
**Solution:**
- This is normal - emails must be unique in the database
- Try a different email address

## Project Structure

```
enable_agents/
├── tools/
│   ├── app.py                          # Flask backend
│   ├── migrations.py                   # Database migrations
│   ├── test_crud_api.py               # API testing script
│   └── requirements.txt
├── agent-app/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ExecutiveAssistantPage.js    # Main component
│   │   │   └── Header.js
│   │   ├── services/
│   │   │   └── executiveAssistantAPI.js     # API service
│   │   ├── styles/
│   │   │   └── ExecutiveAssistantPage.css
│   │   └── index.js
│   ├── package.json
│   ├── public/
│   └── README.md
├── CRUD_INTEGRATION_GUIDE.md           # Detailed API documentation
├── QUICK_START.md                      # This file
└── data/
    └── user_data/
```

## Key Features

### Projects
- ✅ Create new projects
- ✅ View all projects
- ✅ Edit project details
- ✅ Delete projects (cascades to tasks)
- ✅ Mark projects as active/inactive

### Tasks
- ✅ Create tasks linked to projects
- ✅ Assign tasks to team members
- ✅ Set priority (Low/Medium/High)
- ✅ Track status (Pending/In Progress/Completed)
- ✅ Set due dates
- ✅ Edit and delete tasks

### Team Members
- ✅ Add team members with contact info
- ✅ Store email, phone, WhatsApp numbers
- ✅ Assign role/position
- ✅ Edit member details
- ✅ Delete members (unassigns their tasks)

## API Endpoints

All endpoints return JSON and support CORS.

**Base URL:** `http://localhost:5000/api`

### Projects
- `GET /projects` - List all projects
- `GET /projects/<id>` - Get single project
- `POST /projects` - Create project
- `PUT /projects/<id>` - Update project
- `DELETE /projects/<id>` - Delete project

### People
- `GET /people` - List all people
- `GET /people/<id>` - Get single person
- `POST /people` - Create person
- `PUT /people/<id>` - Update person
- `DELETE /people/<id>` - Delete person

### Tasks
- `GET /tasks` - List all tasks (supports filters)
- `GET /tasks/<id>` - Get single task
- `POST /tasks` - Create task
- `PUT /tasks/<id>` - Update task
- `DELETE /tasks/<id>` - Delete task

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Switch to Projects | Click Projects tab |
| Switch to Tasks | Click Tasks tab |
| Switch to People | Click People tab |
| Add new item | Click + button |
| Edit item | Click Edit button |
| Delete item | Click Delete button |

## Database Details

**Database Name:** enable_agents

**Tables:**
- `users` - User accounts
- `people` - Team members (email, phone, WhatsApp)
- `projects` - Projects (name, description, due date)
- `tasks` - Tasks (linked to projects and people)

**Connection String:**
```
mysql+mysqlconnector://root:root@localhost/enable_agents
```

## Support Files

1. **CRUD_INTEGRATION_GUIDE.md** - Comprehensive API documentation
2. **DATABASE_SETUP_GUIDE.md** - Database schema details
3. **DATABASE_API_EXAMPLES.md** - cURL examples for all endpoints
4. **test_crud_api.py** - Automated testing script

## Next Steps

1. ✅ Complete the Quick Start setup
2. ✅ Test CRUD operations in the UI
3. 📖 Read CRUD_INTEGRATION_GUIDE.md for detailed API docs
4. 🔧 Customize styling in ExecutiveAssistantPage.css if needed
5. 🚀 Deploy when ready

## Tips & Tricks

### Testing API with curl

```bash
# Get all projects
curl http://localhost:5000/api/projects

# Create a project
curl -X POST http://localhost:5000/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "My Project",
    "project_description": "Description",
    "due_date": "2025-05-15",
    "is_active": true
  }'

# Get single project
curl http://localhost:5000/api/projects/1

# Update a project
curl -X PUT http://localhost:5000/api/projects/1 \
  -H "Content-Type: application/json" \
  -d '{"project_name": "Updated Name", ...}'

# Delete a project
curl -X DELETE http://localhost:5000/api/projects/1
```

### Enable Debug Mode

In `tools/app.py`, change:
```python
if __name__ == '__main__':
    app.run(debug=True)  # Add debug=True
```

Then restart the server. Changes to Python code will auto-reload.

### Database Inspection

View tables and data:
```bash
python tools/migrations.py schema   # View schema
python tools/migrations.py status   # View record counts
```

## Performance Notes

- Initial load from API: ~200-500ms
- Create/update operations: ~100-300ms
- Delete operations: ~100-300ms
- No pagination implemented (fine for < 1000 records)

## Security Notes

⚠️ **For Development Only**

This setup includes:
- ❌ No authentication/authorization
- ❌ No input sanitization
- ❌ No rate limiting
- ❌ CORS enabled for all origins
- ❌ Debug mode may be enabled

For production deployment:
- ✅ Add JWT authentication
- ✅ Implement input validation
- ✅ Add rate limiting
- ✅ Use environment variables for secrets
- ✅ Enable HTTPS
- ✅ Restrict CORS
- ✅ Disable debug mode

## Helpful Commands

```bash
# Check if ports are available
lsof -i :3000    # React
lsof -i :5000    # Flask

# Kill a process on a port
kill -9 <PID>

# Check MySQL status
brew services list

# Start MySQL (macOS)
brew services start mysql

# Stop MySQL (macOS)
brew services stop mysql

# View logs
tail -f nohup.out

# Clear npm cache
npm cache clean --force
```

## Contact & Support

For issues or questions about the Executive Assistant Agent:
1. Check the CRUD_INTEGRATION_GUIDE.md
2. Review test output from test_crud_api.py
3. Check browser console (F12) for client-side errors
4. Check Flask console for server-side errors

---

**Version:** 1.0.0  
**Last Updated:** March 20, 2025  
**Status:** ✅ Ready for Use
