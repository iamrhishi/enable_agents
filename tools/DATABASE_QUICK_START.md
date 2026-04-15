# Database Quick Start Guide

This is a quick reference for setting up and using the Enable Agents database. For detailed information, see [DATABASE_SCHEMA.md](./DATABASE_SCHEMA.md).

## 🚀 Setup (First Time)

```bash
cd tools/
python -c "from app import app, db; from migrations import init_db; app.app_context().push(); init_db()"
```

Or interactively:
```python
python
>>> from app import app, db
>>> from migrations import init_db
>>> app.app_context().push()
>>> init_db()
>>> exit()
```

You should see: ✅ Database initialized successfully!

## 📊 Database Structure

| Table | Purpose | Location |
|-------|---------|----------|
| `users` | User accounts | enable_agents.db |
| `email_extraction_quotas` | Email limits per user | enable_agents.db |
| `email_extraction_usage_logs` | Billing/audit logs | enable_agents.db |
| `email_campaigns` | Marketing campaigns | enable_agents.db |
| `email_campaign_recipients` | Campaign recipients | enable_agents.db |
| `projects` (CM DB) | Content marketing projects | content_marketing.db |
| `documents` (CM DB) | Project documents | content_marketing.db |
| `knowledge_graphs` (CM DB) | Graph data | content_marketing.db |
| `generated_content` (CM DB) | Generated content | content_marketing.db |
| `conversation_history` (CM DB) | Chat history | content_marketing.db |

## 💻 Common Tasks

### Add a User
```python
from app import app, User, db
from werkzeug.security import generate_password_hash

with app.app_context():
    user = User(
        username='jane_doe',
        password=generate_password_hash('password123'),
        email='jane@company.com',
        first_name='Jane',
        last_name='Doe'
    )
    db.session.add(user)
    db.session.commit()
```

### Query Users
```python
from app import app, User

with app.app_context():
    all_users = User.query.all()
    one_user = User.query.filter_by(username='jane_doe').first()
    print(f"Found {len(all_users)} users")
```

### Create Email Campaign
```python
from app import app, db, EmailCampaign, EmailCampaignRecipient
from uuid import uuid4

with app.app_context():
    campaign = EmailCampaign(
        id=str(uuid4()),
        name='Q2 Campaign',
        subject='Summer 2025 Promotions',
        username='jane_doe'
    )
    db.session.add(campaign)
    
    recipient = EmailCampaignRecipient(
        campaign_id=campaign.id,
        receiver_email='customer@example.com',
        receiver_name='Customer Name'
    )
    db.session.add(recipient)
    db.session.commit()
```

### Reset Database (Development Only)
```bash
python
>>> from migrations import reset_db
>>> reset_db()
# Type 'yes' when prompted
```

## 🔧 Configuration

### Connection String (.env)
```env
# Default (SQLite)
DATABASE_URI=sqlite:///enable_agents.db

# PostgreSQL
DATABASE_URI=postgresql://user:password@localhost/enable_agents

# MySQL
DATABASE_URI=mysql+pymysql://user:password@localhost/enable_agents
```

### Constants
```env
EMAIL_EXTRACTION_UNIT_COST=0.20
EMAIL_EXTRACTION_DEFAULT_LIMIT=500
```

## 📂 File Locations

```
tools/
├── app.py                  # Model definitions
├── migrations.py           # Setup functions
└── DATABASE_SCHEMA.md      # Full documentation
└── DATABASE_QUICK_START.md # This file

data/
├── enable_agents.db        # Main SQLAlchemy database
└── content_marketing.db    # Content marketing database
```

## 🐛 Troubleshooting

### "Database is locked"
```bash
rm *.db-journal
# Restart the application
```

### "Foreign key constraint failed"
- Ensure you're adding records to the right table
- Check for references to non-existent parent records

### Need fresh start?
```python
from migrations import drop_db, init_db
from app import app
app.app_context().push()
drop_db()
init_db()
```

## 📚 Learn More

- **Full schema details:** [DATABASE_SCHEMA.md](./DATABASE_SCHEMA.md)
- **Flask-SQLAlchemy docs:** https://flask-sqlalchemy.palletsprojects.com/
- **SQLAlchemy docs:** https://docs.sqlalchemy.org/

## 🚨 Important

- ⚠️ Never run `reset_db()` or `drop_db()` in production
- Use `.env` for sensitive credentials
- Always close database sessions: `db.session.close()`
- Test database queries before deployment

---

**Questions?** Check [DATABASE_SCHEMA.md](./DATABASE_SCHEMA.md) for detailed reference.
