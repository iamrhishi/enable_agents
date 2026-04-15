# Enable Agents Database Schema

## Overview

This document provides a complete reference for the Enable Agents database schema. The application uses **SQLAlchemy ORM** with **SQLite** as the default database (configurable via environment variables).

### Quick Start

```bash
# Initialize the database
cd tools/
python -c "from app import app, db; from migrations import init_db; app.app_context().push(); init_db()"

# Or use Python interactively
python
>>> from app import app, db
>>> from migrations import init_db
>>> with app.app_context():
...     init_db()
```

---

## Database Configuration

### Connection String

Default:
```
sqlite:///enable_agents.db
```

Custom via environment variable:
```bash
# In .env file
DATABASE_URI=sqlite:///enable_agents.db
# Or for PostgreSQL:
DATABASE_URI=postgresql://user:password@localhost/enable_agents
# Or for MySQL:
DATABASE_URI=mysql+pymysql://user:password@localhost/enable_agents
```

### Flask Configuration

```python
# From app.py
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URI',
    'sqlite:///enable_agents.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
```

---

## Core Tables

### 1. `users` Table

Stores user account information and profile details.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-incremented user ID |
| `username` | VARCHAR(80) | UNIQUE, NOT NULL | Unique username for login |
| `password` | VARCHAR(128) | NOT NULL | Hashed password (use `werkzeug.security.generate_password_hash`) |
| `first_name` | VARCHAR(80) | NULL | User's first name |
| `last_name` | VARCHAR(80) | NULL | User's last name |
| `email` | VARCHAR(120) | NULL | User's email address |
| `company` | VARCHAR(120) | NULL | Company name |
| `linkedin` | VARCHAR(256) | NULL | LinkedIn profile URL |
| `short_intro` | VARCHAR(256) | NULL | Short bio/introduction |
| `company_intro` | VARCHAR(256) | NULL | Company description |

**Model:**
```python
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(80))
    last_name = db.Column(db.String(80))
    email = db.Column(db.String(120))
    company = db.Column(db.String(120))
    linkedin = db.Column(db.String(256))
    short_intro = db.Column(db.String(256))
    company_intro = db.Column(db.String(256))
```

---

### 2. `email_extraction_quotas` Table

Tracks email extraction limits and usage per user.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-incremented record ID |
| `username` | VARCHAR(120) | UNIQUE, NOT NULL, INDEX | User's username |
| `total_allowed` | INTEGER | NOT NULL, DEFAULT=500 | Total email extractions allowed |
| `used_count` | INTEGER | NOT NULL, DEFAULT=0 | Count of extractions used |
| `created_at` | DATETIME | NOT NULL | Record creation timestamp |
| `updated_at` | DATETIME | NOT NULL | Last update timestamp |

**Model:**
```python
class EmailExtractionQuota(db.Model):
    __tablename__ = 'email_extraction_quotas'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False, index=True)
    total_allowed = db.Column(db.Integer, nullable=False, default=500)
    used_count = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, 
                          onupdate=datetime.utcnow)
```

**Usage Notes:**
- Default limit is 500 (configurable via `EMAIL_EXTRACTION_DEFAULT_LIMIT` env var)
- Unit cost is $0.20 per extraction (configurable via `EMAIL_EXTRACTION_UNIT_COST` env var)

---

### 3. `email_extraction_usage_logs` Table

Detailed logging of each email extraction request for billing and auditing.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-incremented log ID |
| `request_id` | VARCHAR(64) | UNIQUE, NOT NULL, INDEX | Unique request identifier |
| `username` | VARCHAR(120) | NOT NULL, INDEX | User who made the request |
| `processed_count` | INTEGER | NOT NULL, DEFAULT=0 | Total records processed |
| `billable_count` | INTEGER | NOT NULL, DEFAULT=0 | Records that are billable |
| `charged_count` | INTEGER | NOT NULL, DEFAULT=0 | Records actually charged |
| `cost_this_request` | FLOAT | NOT NULL, DEFAULT=0.0 | Cost for this request |
| `total_cost_after` | FLOAT | NOT NULL, DEFAULT=0.0 | Cumulative cost after request |
| `created_at` | DATETIME | NOT NULL | Request timestamp |

**Model:**
```python
class EmailExtractionUsageLog(db.Model):
    __tablename__ = 'email_extraction_usage_logs'
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    username = db.Column(db.String(120), nullable=False, index=True)
    processed_count = db.Column(db.Integer, nullable=False, default=0)
    billable_count = db.Column(db.Integer, nullable=False, default=0)
    charged_count = db.Column(db.Integer, nullable=False, default=0)
    cost_this_request = db.Column(db.Float, nullable=False, default=0.0)
    total_cost_after = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
```

---

### 4. `email_campaigns` Table

Stores email marketing campaigns.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | VARCHAR(36) | PRIMARY KEY | UUID for campaign |
| `name` | VARCHAR(255) | NOT NULL | Campaign display name |
| `subject` | VARCHAR(255) | NOT NULL | Email subject line |
| `username` | VARCHAR(120) | INDEX | User who created campaign |
| `created_at` | DATETIME | NOT NULL | Campaign creation timestamp |

**Model:**
```python
class EmailCampaign(db.Model):
    __tablename__ = 'email_campaigns'
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(120), index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
```

**Relationships:**
- One-to-Many with `email_campaign_recipients`

---

### 5. `email_campaign_recipients` Table

Tracks individual recipients and delivery status for each campaign.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-incremented record ID |
| `campaign_id` | VARCHAR(36) | FOREIGN KEY, NOT NULL | Reference to email_campaigns |
| `receiver_email` | VARCHAR(255) | NOT NULL, INDEX | Recipient email address |
| `receiver_name` | VARCHAR(255) | NULL | Recipient name |
| `status` | VARCHAR(50) | DEFAULT='Sent' | Email delivery status (Sent, Bounced, etc.) |
| `reply_status` | VARCHAR(50) | DEFAULT='No Reply' | Reply tracking (Replied, No Reply, etc.) |
| `sent_at` | DATETIME | NOT NULL | When email was sent |
| `replied_at` | DATETIME | NULL | When recipient replied (if applicable) |

**Model:**
```python
class EmailCampaignRecipient(db.Model):
    __tablename__ = 'email_campaign_recipients'
    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.String(36), db.ForeignKey('email_campaigns.id'), 
                           nullable=False)
    receiver_email = db.Column(db.String(255), nullable=False, index=True)
    receiver_name = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(50), default='Sent')
    reply_status = db.Column(db.String(50), default='No Reply')
    sent_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    replied_at = db.Column(db.DateTime, nullable=True)
```

**Relationships:**
- Many-to-One with `email_campaigns` (campaign_id)

---

## SQLite Content Marketing Database

A separate SQLite database manages content marketing agent data.

### Database File
```
Path: ~/Enable/Software_Development/enable_agents/data/content_marketing.db
Created by: init_content_marketing_db()
```

### 6. `projects` Table (Content Marketing DB)

Content marketing projects.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `project_id` | TEXT | PRIMARY KEY | Unique project identifier |
| `user_id` | TEXT | NOT NULL | User who owns project |
| `project_name` | TEXT | NOT NULL | Project name |
| `description` | TEXT | NULL | Project description |
| `industry` | TEXT | NULL | Industry classification |
| `sector` | TEXT | NULL | Business sector |
| `function` | TEXT | NULL | Functional area |
| `role` | TEXT | NULL | User's role in project |
| `created_at` | TIMESTAMP | NULL | Creation timestamp |
| `updated_at` | TIMESTAMP | NULL | Last update timestamp |
| `metadata` | JSON | NULL | Additional metadata |

**SQL:**
```sql
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_name TEXT NOT NULL,
    description TEXT,
    industry TEXT,
    sector TEXT,
    function TEXT,
    role TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata JSON
)
```

---

### 7. `documents` Table (Content Marketing DB)

Documents uploaded to projects.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `doc_id` | TEXT | PRIMARY KEY | Unique document identifier |
| `project_id` | TEXT | FOREIGN KEY, NOT NULL | Reference to projects |
| `file_name` | TEXT | NOT NULL | Original filename |
| `file_type` | TEXT | NULL | File MIME type |
| `file_path` | TEXT | NULL | Storage path |
| `file_size` | INTEGER | NULL | Size in bytes |
| `upload_date` | TIMESTAMP | NULL | Upload timestamp |
| `document_type` | TEXT | NULL | Document classification |
| `extracted_content` | TEXT | NULL | Extracted text content |

**SQL:**
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT,
    file_path TEXT,
    file_size INTEGER,
    upload_date TIMESTAMP,
    document_type TEXT,
    extracted_content TEXT,
    FOREIGN KEY(project_id) REFERENCES projects(project_id)
)
```

**Allowed File Types:**
```python
CONTENT_MARKETING_ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'xlsx', 'html', 'md'}
```

---

### 8. `knowledge_graphs` Table (Content Marketing DB)

Knowledge graphs generated from documents.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `kg_id` | TEXT | PRIMARY KEY | Unique knowledge graph ID |
| `project_id` | TEXT | FOREIGN KEY, NOT NULL | Reference to projects |
| `kg_data` | JSON | NULL | Graph data structure |
| `entities` | INT | NULL | Count of entities |
| `relationships` | INT | NULL | Count of relationships |
| `created_at` | TIMESTAMP | NULL | Creation timestamp |

**SQL:**
```sql
CREATE TABLE knowledge_graphs (
    kg_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kg_data JSON,
    entities INT,
    relationships INT,
    created_at TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(project_id)
)
```

---

### 9. `generated_content` Table (Content Marketing DB)

Content generated by agents.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `content_id` | TEXT | PRIMARY KEY | Unique content identifier |
| `project_id` | TEXT | FOREIGN KEY, NOT NULL | Reference to projects |
| `channel` | TEXT | NULL | Distribution channel |
| `content_type` | TEXT | NULL | Type of content (blog, etc.) |
| `content` | TEXT | NULL | Generated content text |
| `source_docs` | JSON | NULL | Source documents used |
| `domain_context` | JSON | NULL | Domain context info |
| `created_at` | TIMESTAMP | NULL | Creation timestamp |
| `modified_at` | TIMESTAMP | NULL | Last modification timestamp |

**SQL:**
```sql
CREATE TABLE generated_content (
    content_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    channel TEXT,
    content_type TEXT,
    content TEXT,
    source_docs JSON,
    domain_context JSON,
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(project_id)
)
```

---

### 10. `conversation_history` Table (Content Marketing DB)

Chat history for agent conversations.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `msg_id` | TEXT | PRIMARY KEY | Unique message identifier |
| `project_id` | TEXT | FOREIGN KEY, NOT NULL | Reference to projects |
| `user_message` | TEXT | NULL | User's message |
| `agent_response` | TEXT | NULL | Agent's response |
| `context` | JSON | NULL | Conversation context |
| `timestamp` | TIMESTAMP | NULL | Message timestamp |

**SQL:**
```sql
CREATE TABLE conversation_history (
    msg_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    user_message TEXT,
    agent_response TEXT,
    context JSON,
    timestamp TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(project_id)
)
```

---

## Entity Relationship Diagram

```
PRIMARY DATABASES:

SQLAlchemy (enable_agents.db):
├── users
├── email_extraction_quotas (references: username → users.username)
├── email_extraction_usage_logs (references: username → users.username)
├── email_campaigns
└── email_campaign_recipients (references: campaign_id → email_campaigns.id)


Content Marketing (content_marketing.db):
├── projects (user_id → users from main DB)
├── documents (references: project_id → projects.project_id)
├── knowledge_graphs (references: project_id → projects.project_id)
├── generated_content (references: project_id → projects.project_id)
└── conversation_history (references: project_id → projects.project_id)
```

---

## Common Operations

### Initialize Database

```python
from app import app, db
from migrations import init_db

with app.app_context():
    init_db()
```

### Reset Database (Development Only)

```python
from migrations import reset_db

reset_db()
```

### Drop All Tables (Development Only)

```python
from migrations import drop_db

drop_db()
```

### Create a New User

```python
from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    user = User(
        username='john_doe',
        password=generate_password_hash('secure_password_here'),
        first_name='John',
        last_name='Doe',
        email='john@example.com',
        company='Acme Corp'
    )
    db.session.add(user)
    db.session.commit()
```

### Query Users

```python
from app import app, User

with app.app_context():
    # Get all users
    all_users = User.query.all()
    
    # Get by username
    user = User.query.filter_by(username='john_doe').first()
    
    # Get by ID
    user = User.query.get(1)
```

### Initialize Email Extraction Quota

```python
from app import app, db, EmailExtractionQuota

with app.app_context():
    quota = EmailExtractionQuota(
        username='john_doe',
        total_allowed=500,
        used_count=0
    )
    db.session.add(quota)
    db.session.commit()
```

### Log Email Extraction Usage

```python
from app import app, db, EmailExtractionUsageLog
from uuid import uuid4

with app.app_context():
    log = EmailExtractionUsageLog(
        request_id=str(uuid4()),
        username='john_doe',
        processed_count=100,
        billable_count=95,
        charged_count=95,
        cost_this_request=19.00,
        total_cost_after=500.00
    )
    db.session.add(log)
    db.session.commit()
```

### Create Email Campaign

```python
from app import app, db, EmailCampaign, EmailCampaignRecipient
from uuid import uuid4
from datetime import datetime

with app.app_context():
    campaign = EmailCampaign(
        id=str(uuid4()),
        name='Summer 2025 Campaign',
        subject='Check out our summer offers!',
        username='john_doe'
    )
    db.session.add(campaign)
    
    recipient = EmailCampaignRecipient(
        campaign_id=campaign.id,
        receiver_email='jane@example.com',
        receiver_name='Jane Smith',
        status='Sent'
    )
    db.session.add(recipient)
    db.session.commit()
```

---

## Database Maintenance

### Backup Database

```bash
# SQLite
cp enable_agents.db enable_agents.db.backup

# Content Marketing DB
cp data/content_marketing.db data/content_marketing.db.backup
```

### Export to CSV

```bash
# From SQLite
sqlite3 enable_agents.db "SELECT * FROM users;" > users.csv
```

### Run Integrity Check

```bash
sqlite3 enable_agents.db "PRAGMA integrity_check;"
```

---

## Troubleshooting

### Database Locked Error

**Problem:** `sqlite3.OperationalError: database is locked`

**Solution:**
```bash
# Stop all running processes
# Remove lock files (if any)
rm *.db-journal

# Restart the application
```

### Foreign Key Constraint Failed

**Problem:** `IntegrityError: (sqlite3.IntegrityError) FOREIGN KEY constraint failed`

**Solution:**
- Enable foreign key constraints:
```python
@app.before_request
def enable_foreign_keys():
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite'):
        db.engine.execute('PRAGMA foreign_keys = ON')
```

### Reset Database (if corrupted)

```python
from migrations import drop_db, init_db
from app import app

with app.app_context():
    drop_db()
    init_db()
```

---

## Environment Variables

```bash
# Database Configuration
DATABASE_URI=sqlite:///enable_agents.db

# Email Extraction Configuration
EMAIL_EXTRACTION_UNIT_COST=0.20
EMAIL_EXTRACTION_DEFAULT_LIMIT=500

# Content Marketing Configuration
CONTENT_MARKETING_UPLOAD_FOLDER=~/Enable/Software_Development/enable_agents/data/content_marketing_uploads
CONTENT_MARKETING_DB_PATH=~/Enable/Software_Development/enable_agents/data/content_marketing.db
```

---

## For Developers

### Adding a New Table

1. **Define the model** in `app.py`:
```python
class YourModel(db.Model):
    __tablename__ = 'your_table'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

2. **Create migration** (optional, for version control):
```python
# Add to migrations.py
def add_your_table():
    """Add your_table to database"""
    db.create_all()
```

3. **Initialize**:
```python
with app.app_context():
    db.create_all()
```

### Querying Best Practices

```python
from app import app, db, User

with app.app_context():
    # Use filters efficiently
    users = User.query.filter(User.email.isnot(None)).all()
    
    # Pagination for large datasets
    page = User.query.paginate(page=1, per_page=20)
    
    # Use joins for related data
    # (when implementing relationships)
    
    # Always close sessions
    db.session.close()
```

---

## Related Files

- [app.py](./app.py) - Main Flask application and model definitions
- [migrations.py](./migrations.py) - Database initialization and migration utilities

---

**Last Updated:** April 2025  
**Database Version:** 1.0  
**Compatibility:** SQLAlchemy 3.x, Flask 2.x
