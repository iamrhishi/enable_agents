"""
Database Migrations for Enable Agents
======================================

This module handles database schema setup and migrations for the Enable Agents application.
It uses Flask-SQLAlchemy to create and manage the database schema.

Usage:
    To initialize the database schema, run:
    
        python
        >>> from app import app, db
        >>> from migrations import init_db
        >>> with app.app_context():
        ...     init_db()

    Or from command line:
    
        python -c "from app import app, db; from migrations import init_db; 
                   app.app_context().push(); init_db()"

Author: Enable Agents Team
Version: 1.0
"""

from app import app, db, User


def init_db():
    """
    Initialize the database with the current schema.
    
    This function:
    1. Creates all database tables defined in the SQLAlchemy models
    2. Prints confirmation of tables created
    3. Is idempotent (safe to run multiple times)
    
    Safe to run multiple times - existing tables will not be recreated.
    """
    try:
        db.create_all()
        print("✅ Database initialized successfully!")
        print("\nTables created:")
        print("  - users (User model)")
        print("\nDatabase URI:", app.config['SQLALCHEMY_DATABASE_URI'])
        print("\nYour colleague can now start using the application.")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False


def drop_db():
    """
    Drop all database tables.
    
    WARNING: This will delete all data in the database!
    Use only for development/testing purposes.
    """
    try:
        confirm = input(
            "⚠️  WARNING: This will DELETE ALL DATA in the database. "
            "Type 'yes' to confirm: "
        )
        if confirm.lower() == 'yes':
            db.drop_all()
            print("✅ All database tables dropped.")
            return True
        else:
            print("❌ Drop cancelled.")
            return False
    except Exception as e:
        print(f"❌ Error dropping database: {e}")
        return False


def reset_db():
    """
    Reset the database by dropping all tables and recreating them.
    
    WARNING: This will delete all data in the database!
    Use only for development/testing purposes.
    """
    try:
        confirm = input(
            "⚠️  WARNING: This will DELETE ALL DATA and reset the database. "
            "Type 'yes' to confirm: "
        )
        if confirm.lower() == 'yes':
            db.drop_all()
            print("🗑️  All tables dropped.")
            db.create_all()
            print("✅ Database reset and reinitialized successfully!")
            return True
        else:
            print("❌ Reset cancelled.")
            return False
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return False


def get_db_status():
    """
    Get current database status and schema information.
    
    Returns:
        dict: Database status information
    """
    try:
        # Check if users table exists by trying to query it
        user_count = User.query.count()
        
        return {
            'status': 'connected',
            'database_uri': app.config['SQLALCHEMY_DATABASE_URI'],
            'tables': {
                'users': {
                    'exists': True,
                    'row_count': user_count,
                    'columns': [
                        'id (Integer, PK)',
                        'username (String(80), unique)',
                        'password (String(128))',
                        'first_name (String(80))',
                        'last_name (String(80))',
                        'email (String(120))',
                        'company (String(120))',
                        'linkedin (String(256))',
                        'short_intro (String(256))',
                        'company_intro (String(256))'
                    ]
                }
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'message': 'Database connection failed. Check DATABASE_URI and ensure database server is running.'
        }


def print_schema():
    """
    Print a human-readable version of the database schema.
    """
    print("\n" + "="*70)
    print("DATABASE SCHEMA")
    print("="*70)
    
    print("\n📋 TABLE: users")
    print("-" * 70)
    print(f"{'Column':<20} {'Type':<20} {'Constraints':<30}")
    print("-" * 70)
    
    schema_rows = [
        ('id', 'INTEGER', 'PRIMARY KEY, AUTO_INCREMENT'),
        ('username', 'VARCHAR(80)', 'UNIQUE, NOT NULL'),
        ('password', 'VARCHAR(128)', 'NOT NULL'),
        ('first_name', 'VARCHAR(80)', 'NULLABLE'),
        ('last_name', 'VARCHAR(80)', 'NULLABLE'),
        ('email', 'VARCHAR(120)', 'NULLABLE'),
        ('company', 'VARCHAR(120)', 'NULLABLE'),
        ('linkedin', 'VARCHAR(256)', 'NULLABLE'),
        ('short_intro', 'VARCHAR(256)', 'NULLABLE'),
        ('company_intro', 'VARCHAR(256)', 'NULLABLE'),
    ]
    
    for column, dtype, constraints in schema_rows:
        print(f"{column:<20} {dtype:<20} {constraints:<30}")
    
    print("\n" + "="*70)


def create_sample_user(username="demo_user", password="demo_password"):
    """
    Create a sample user for testing purposes.
    
    Args:
        username (str): Username for the sample user
        password (str): Password for the sample user
        
    Returns:
        User: The created User object, or None if error
    """
    try:
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f"⚠️  User '{username}' already exists.")
            return existing_user
        
        from werkzeug.security import generate_password_hash
        
        new_user = User(
            username=username,
            password=generate_password_hash(password),
            first_name="Demo",
            last_name="User",
            email="demo@example.com",
            company="Demo Company"
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        print(f"✅ Sample user created:")
        print(f"   Username: {username}")
        print(f"   Password: {password}")
        print(f"   (Use these credentials to test the application)")
        
        return new_user
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating sample user: {e}")
        return None


def print_help():
    """Print help message for migration commands."""
    help_text = """
╔════════════════════════════════════════════════════════════════════════╗
║          Enable Agents - Database Migration Tool                       ║
╚════════════════════════════════════════════════════════════════════════╝

USAGE:
    python migrations.py <command>

COMMANDS:
    init       Initialize database with schema (creates tables)
    drop       Drop all database tables (DELETE ALL DATA!)
    reset      Reset database (DROP + INIT)
    status     Show current database status
    schema     Print database schema in human-readable format
    sample     Create a sample user for testing
    help       Show this help message

EXAMPLES:

  1. Initialize database for first time:
     $ python migrations.py init

  2. Check current database status:
     $ python migrations.py status

  3. View database schema:
     $ python migrations.py schema

  4. Reset database (for development):
     $ python migrations.py reset

  5. Create a test user:
     $ python migrations.py sample

SETUP INSTRUCTIONS FOR COLLEAGUE:

  1. Ensure MySQL/MariaDB is running and accessible at localhost:3306
  
  2. Create the database:
     $ mysql -u root -p
     mysql> CREATE DATABASE enable_agents CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
     mysql> EXIT;
  
  3. Navigate to tools directory:
     $ cd tools
  
  4. Activate Python environment:
     $ source ../.venv/bin/activate  # On macOS/Linux
     $ ../.venv/Scripts/activate     # On Windows
  
  5. Install dependencies:
     $ pip install -r requirements.txt
  
  6. Initialize database:
     $ python migrations.py init
  
  7. (Optional) Create sample user:
     $ python migrations.py sample
  
  8. Start the application:
     $ python app.py

ENVIRONMENT VARIABLES:

  The app uses the following database configuration from app.py:
  
    Database URI: mysql+mysqlconnector://root:root@localhost/enable_agents
  
  If you need to change database credentials, modify the DATABASE_URI in app.py:
  
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://USER:PASSWORD@HOST/DATABASE'

DATABASE SCHEMA:

  Table: users
  ├── id (Integer, Primary Key, Auto-increment)
  ├── username (String(80), Unique, Not Null)
  ├── password (String(128), Not Null)
  ├── first_name (String(80))
  ├── last_name (String(80))
  ├── email (String(120))
  ├── company (String(120))
  ├── linkedin (String(256))
  ├── short_intro (String(256))
  └── company_intro (String(256))

TROUBLESHOOTING:

  Q: "Can't connect to MySQL server"
  A: Ensure MySQL is running:
     $ sudo systemctl start mysql  # Linux
     $ brew services start mysql   # macOS
     $ mysql.server start          # macOS (older)

  Q: "Access denied for user 'root'"
  A: Check credentials in app.py or update with correct password

  Q: "Database doesn't exist"
  A: Create it first:
     $ mysql -u root -p -e "CREATE DATABASE enable_agents;"

═══════════════════════════════════════════════════════════════════════════
    """
    print(help_text)


# ==================== MIGRATION CLI ====================

if __name__ == '__main__':
    import sys
    
    with app.app_context():
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'init':
                print("Initializing database...")
                init_db()
                
            elif command == 'drop':
                print("Dropping all database tables...")
                drop_db()
                
            elif command == 'reset':
                print("Resetting database...")
                reset_db()
                
            elif command == 'status':
                print("Checking database status...")
                status = get_db_status()
                print("\n" + "="*70)
                print("DATABASE STATUS")
                print("="*70)
                if status['status'] == 'connected':
                    print(f"✅ Status: Connected")
                    print(f"📍 Database: {status['database_uri']}")
                    for table_name, table_info in status['tables'].items():
                        print(f"\n📋 Table: {table_name}")
                        print(f"   Rows: {table_info['row_count']}")
                        print(f"   Columns: {len(table_info['columns'])}")
                else:
                    print(f"❌ Status: {status['status']}")
                    print(f"Error: {status.get('error', 'Unknown error')}")
                print()
                
            elif command == 'schema':
                print("Printing database schema...")
                print_schema()
                
            elif command == 'sample':
                print("Creating sample user...")
                create_sample_user()
                
            else:
                print(f"Unknown command: {command}")
                print_help()
        else:
            # Default action: show help
            print_help()


