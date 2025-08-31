"""
PRISM Database Setup Script
Sets up PostgreSQL database with TimescaleDB and creates all required tables
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def create_database():
    """Create the PRISM database if it doesn't exist"""
    try:
        # Connect to postgres database to create prism_db
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database='postgres',  # Connect to default postgres db
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432)
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'prism_db'")
        if not cursor.fetchone():
            cursor.execute("CREATE DATABASE prism_db")
            print("Database 'prism_db' created successfully")
        else:
            print("Database 'prism_db' already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def setup_database():
    """Setup database schema and extensions"""
    
    # SQL commands to create all tables and extensions
    sql_commands = [
        # Enable extensions
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
        'CREATE EXTENSION IF NOT EXISTS "timescaledb";',
        
        # User profiles table
        '''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id VARCHAR(50) PRIMARY KEY,
            phone_hash VARCHAR(64) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender VARCHAR(10),
            occupation VARCHAR(100),
            income_bracket VARCHAR(20),
            city VARCHAR(50),
            state VARCHAR(50),
            
            movement_entropy DECIMAL(4,3),
            transaction_entropy DECIMAL(4,3),
            social_entropy DECIMAL(4,3),
            temporal_entropy DECIMAL(4,3),
            composite_stability_score DECIMAL(4,3),
            
            credit_risk_score INTEGER,
            default_probability DECIMAL(5,4),
            cascade_risk_score DECIMAL(4,3),
            ghost_defaulter_probability DECIMAL(5,4),
            
            default_status VARCHAR(20) DEFAULT 'active',
            risk_category VARCHAR(20),
            behavioral_fingerprint JSONB,
            
            last_score_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version VARCHAR(20) DEFAULT 'v1.0',
            
            CONSTRAINT valid_entropy_movement CHECK (movement_entropy BETWEEN 0 AND 1),
            CONSTRAINT valid_entropy_transaction CHECK (transaction_entropy BETWEEN 0 AND 1),
            CONSTRAINT valid_entropy_social CHECK (social_entropy BETWEEN 0 AND 1),
            CONSTRAINT valid_entropy_temporal CHECK (temporal_entropy BETWEEN 0 AND 1),
            CONSTRAINT valid_credit_score CHECK (credit_risk_score BETWEEN 300 AND 850),
            CONSTRAINT valid_default_prob CHECK (default_probability BETWEEN 0 AND 1)
        );
        ''',
        
        # Mobile towers table
        '''
        CREATE TABLE IF NOT EXISTS mobile_towers (
            tower_id VARCHAR(20) PRIMARY KEY,
            tower_name VARCHAR(100),
            location_lat DECIMAL(10,7),
            location_lng DECIMAL(10,7),
            coverage_radius INTEGER,
            city VARCHAR(50),
            area_type VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''',
        
        # Movement patterns table
        '''
        CREATE TABLE IF NOT EXISTS movement_patterns (
            id SERIAL,
            user_id VARCHAR(50) NOT NULL REFERENCES user_profiles(user_id),
            timestamp TIMESTAMPTZ NOT NULL,
            tower_id VARCHAR(20) NOT NULL,
            signal_strength INTEGER CHECK (signal_strength BETWEEN -120 AND -30),
            location_lat DECIMAL(10,7),
            location_lng DECIMAL(10,7),
            location_cluster_id INTEGER,
            stay_duration INTEGER,
            transition_probability DECIMAL(5,4),
            movement_vector JSONB,
            
            PRIMARY KEY (user_id, timestamp)
        );
        ''',
        
        # Transaction streams table
        '''
        CREATE TABLE IF NOT EXISTS transaction_streams (
            id SERIAL,
            user_id VARCHAR(50) NOT NULL REFERENCES user_profiles(user_id),
            timestamp TIMESTAMPTZ NOT NULL,
            transaction_id VARCHAR(100) UNIQUE NOT NULL,
            amount DECIMAL(12,2) NOT NULL,
            transaction_type VARCHAR(20) NOT NULL,
            category VARCHAR(50),
            merchant_id VARCHAR(50),
            merchant_category VARCHAR(50),
            payment_method VARCHAR(20),
            
            time_of_day INTEGER CHECK (time_of_day BETWEEN 0 AND 23),
            day_of_week INTEGER CHECK (day_of_week BETWEEN 1 AND 7),
            is_weekend BOOLEAN,
            is_recurring BOOLEAN DEFAULT FALSE,
            
            unusual_time BOOLEAN DEFAULT FALSE,
            unusual_amount BOOLEAN DEFAULT FALSE,
            
            PRIMARY KEY (user_id, timestamp, transaction_id)
        );
        ''',
        
        # Social connections table
        '''
        CREATE TABLE IF NOT EXISTS social_connections (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL REFERENCES user_profiles(user_id),
            connected_user_id VARCHAR(50) NOT NULL REFERENCES user_profiles(user_id),
            connection_type VARCHAR(20) NOT NULL,
            strength DECIMAL(3,2) CHECK (strength BETWEEN 0 AND 1),
            frequency INTEGER DEFAULT 0,
            total_transaction_amount DECIMAL(12,2) DEFAULT 0,
            last_interaction TIMESTAMP,
            risk_correlation DECIMAL(4,3),
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            CONSTRAINT no_self_connection CHECK (user_id != connected_user_id),
            CONSTRAINT unique_connection UNIQUE (user_id, connected_user_id)
        );
        ''',
        
        # Risk score history table
        '''
        CREATE TABLE IF NOT EXISTS risk_score_history (
            id SERIAL,
            user_id VARCHAR(50) NOT NULL REFERENCES user_profiles(user_id),
            timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            credit_risk_score INTEGER,
            default_probability DECIMAL(5,4),
            movement_entropy DECIMAL(4,3),
            transaction_entropy DECIMAL(4,3),
            social_entropy DECIMAL(4,3),
            composite_score DECIMAL(4,3),
            score_change INTEGER,
            trigger_event VARCHAR(100),
            
            PRIMARY KEY (user_id, timestamp)
        );
        ''',
        
        # Behavioral fingerprints table
        '''
        CREATE TABLE IF NOT EXISTS behavioral_fingerprints (
            user_id VARCHAR(50) PRIMARY KEY REFERENCES user_profiles(user_id),
            fingerprint_hash VARCHAR(64) UNIQUE NOT NULL,
            movement_pattern_vector DECIMAL(10,6)[],
            transaction_pattern_vector DECIMAL(10,6)[],
            temporal_pattern_vector DECIMAL(10,6)[],
            social_pattern_vector DECIMAL(10,6)[],
            
            similarity_threshold DECIMAL(4,3) DEFAULT 0.85,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''',
        
        # Ghost defaulter matches table
        '''
        CREATE TABLE IF NOT EXISTS ghost_defaulter_matches (
            id SERIAL PRIMARY KEY,
            primary_user_id VARCHAR(50) REFERENCES user_profiles(user_id),
            suspected_duplicate_id VARCHAR(50) REFERENCES user_profiles(user_id),
            similarity_score DECIMAL(4,3) NOT NULL,
            match_confidence VARCHAR(20),
            matching_features JSONB,
            investigation_status VARCHAR(20) DEFAULT 'pending',
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            CONSTRAINT no_self_match CHECK (primary_user_id != suspected_duplicate_id),
            CONSTRAINT valid_similarity CHECK (similarity_score BETWEEN 0 AND 1)
        );
        '''
    ]
    
    # Performance indexes
    index_commands = [
        'CREATE INDEX IF NOT EXISTS idx_user_risk_score ON user_profiles(credit_risk_score);',
        'CREATE INDEX IF NOT EXISTS idx_user_entropy ON user_profiles(composite_stability_score);',
        'CREATE INDEX IF NOT EXISTS idx_user_update_time ON user_profiles(last_score_update);',
        'CREATE INDEX IF NOT EXISTS idx_user_status ON user_profiles(default_status);',
        'CREATE INDEX IF NOT EXISTS idx_tower_location ON mobile_towers(location_lat, location_lng);',
        'CREATE INDEX IF NOT EXISTS idx_movement_user_time ON movement_patterns(user_id, timestamp DESC);',
        'CREATE INDEX IF NOT EXISTS idx_movement_cluster ON movement_patterns(location_cluster_id);',
        'CREATE INDEX IF NOT EXISTS idx_transaction_user_time ON transaction_streams(user_id, timestamp DESC);',
        'CREATE INDEX IF NOT EXISTS idx_transaction_amount ON transaction_streams(amount);',
        'CREATE INDEX IF NOT EXISTS idx_transaction_category ON transaction_streams(category);',
        'CREATE INDEX IF NOT EXISTS idx_social_user ON social_connections(user_id);',
        'CREATE INDEX IF NOT EXISTS idx_social_strength ON social_connections(strength DESC);',
        'CREATE INDEX IF NOT EXISTS idx_social_risk_corr ON social_connections(risk_correlation DESC);'
    ]
    
    # TimescaleDB hypertables (after tables are created)
    hypertable_commands = [
        "SELECT create_hypertable('movement_patterns', 'timestamp', if_not_exists => TRUE);",
        "SELECT create_hypertable('transaction_streams', 'timestamp', if_not_exists => TRUE);",
        "SELECT create_hypertable('risk_score_history', 'timestamp', if_not_exists => TRUE);"
    ]
    
    # Views for common queries
    view_commands = [
        '''
        CREATE OR REPLACE VIEW high_risk_users AS
        SELECT 
            user_id,
            credit_risk_score,
            default_probability,
            composite_stability_score,
            risk_category,
            last_score_update
        FROM user_profiles
        WHERE default_probability > 0.3 OR credit_risk_score < 500;
        ''',
        
        '''
        CREATE OR REPLACE VIEW recent_user_activity AS
        SELECT 
            u.user_id,
            u.credit_risk_score,
            COUNT(t.transaction_id) as recent_transactions,
            MAX(m.timestamp) as last_movement,
            MAX(t.timestamp) as last_transaction
        FROM user_profiles u
        LEFT JOIN transaction_streams t ON u.user_id = t.user_id 
            AND t.timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days'
        LEFT JOIN movement_patterns m ON u.user_id = m.user_id 
            AND m.timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days'
        GROUP BY u.user_id, u.credit_risk_score;
        '''
    ]
    
    try:
        # Connect to prism_db
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'prism_db'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT', 5432)
        )
        cursor = conn.cursor()
        
        print("Setting up database schema...")
        
        # Execute main table creation commands
        for i, command in enumerate(sql_commands, 1):
            try:
                cursor.execute(command)
                print(f"✓ Executed command {i}/{len(sql_commands)}")
            except Exception as e:
                print(f"✗ Error in command {i}: {e}")
                conn.rollback()
                continue
        
        conn.commit()
        print("✓ All tables created successfully")
        
        # Create indexes
        print("Creating performance indexes...")
        for i, command in enumerate(index_commands, 1):
            try:
                cursor.execute(command)
                print(f"✓ Created index {i}/{len(index_commands)}")
            except Exception as e:
                print(f"✗ Error creating index {i}: {e}")
                continue
        
        conn.commit()
        print("✓ All indexes created successfully")
        
        # Create hypertables (TimescaleDB)
        print("Creating TimescaleDB hypertables...")
        for i, command in enumerate(hypertable_commands, 1):
            try:
                cursor.execute(command)
                print(f"✓ Created hypertable {i}/{len(hypertable_commands)}")
            except Exception as e:
                print(f"✗ Error creating hypertable {i}: {e}")
                print(f"  (This might be normal if TimescaleDB is not installed)")
                continue
        
        conn.commit()
        
        # Create views
        print("Creating database views...")
        for i, command in enumerate(view_commands, 1):
            try:
                cursor.execute(command)
                print(f"✓ Created view {i}/{len(view_commands)}")
            except Exception as e:
                print(f"✗ Error creating view {i}: {e}")
                continue
        
        conn.commit()
        
        # Verify setup
        print("\nVerifying database setup...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        expected_tables = [
            'behavioral_fingerprints', 'ghost_defaulter_matches', 
            'mobile_towers', 'movement_patterns', 'risk_score_history',
            'social_connections', 'transaction_streams', 'user_profiles'
        ]
        
        found_tables = [table[0] for table in tables]
        
        print(f"✓ Found {len(found_tables)} tables:")
        for table in found_tables:
            status = "✓" if table in expected_tables else "?"
            print(f"  {status} {table}")
        
        missing_tables = set(expected_tables) - set(found_tables)
        if missing_tables:
            print(f"✗ Missing tables: {missing_tables}")
        else:
            print("✓ All expected tables created successfully!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False


def main():
    """Main setup function"""
    print("PRISM Database Setup")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"✗ Missing environment variables: {missing_vars}")
        print("Please create a .env file with the required database credentials")
        return False
    
    # Create database
    print("Step 1: Creating database...")
    if not create_database():
        print("✗ Failed to create database")
        return False
    
    # Setup schema
    print("\nStep 2: Setting up schema...")
    if not setup_database():
        print("✗ Failed to setup database schema")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Database setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python scripts/generate_data.py")
    print("2. Run: python scripts/train_models.py")
    print("3. Start API: uvicorn src.api.main:app --reload")
    
    return True


if __name__ == "__main__":
    success = main()
    print("done")
    sys.exit(0 if success else 1)