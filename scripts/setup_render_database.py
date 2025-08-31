import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from pathlib import Path

def setup_database():
    """Setup database with TimescaleDB extension and load data"""
    
    # Database connection parameters
    db_params = {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "prism_db"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", 5432)),
    }
    
    conn = psycopg2.connect(**db_params)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Enable TimescaleDB extension
        print("Enabling TimescaleDB extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        
        # Create tables
        print("Creating tables...")
        
        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id VARCHAR(50) PRIMARY KEY,
                age INTEGER,
                income DECIMAL(12,2),
                city VARCHAR(100),
                occupation VARCHAR(100),
                credit_risk_score INTEGER,
                default_probability DECIMAL(6,4),
                risk_category VARCHAR(20),
                default_status VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Movement patterns table with TimescaleDB hypertable
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movement_patterns (
                id SERIAL,
                user_id VARCHAR(50),
                timestamp TIMESTAMPTZ NOT NULL,
                latitude DECIMAL(10,7),
                longitude DECIMAL(10,7),
                location_type VARCHAR(50),
                duration_minutes INTEGER,
                distance_from_home_km DECIMAL(8,2),
                tower_id VARCHAR(20),
                PRIMARY KEY (id, timestamp)
            );
        """)
        
        # Create hypertable for movement_patterns
        try:
            cursor.execute("SELECT create_hypertable('movement_patterns', 'timestamp', if_not_exists => TRUE);")
            print("Created TimescaleDB hypertable for movement_patterns")
        except Exception as e:
            print(f"Hypertable creation note: {e}")
        
        # Transaction streams table with TimescaleDB hypertable
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transaction_streams (
                id SERIAL,
                user_id VARCHAR(50),
                timestamp TIMESTAMPTZ NOT NULL,
                amount DECIMAL(12,2),
                category VARCHAR(50),
                merchant VARCHAR(100),
                payment_method VARCHAR(30),
                location VARCHAR(100),
                is_weekend BOOLEAN,
                hour_of_day INTEGER,
                PRIMARY KEY (id, timestamp)
            );
        """)
        
        # Create hypertable for transaction_streams
        try:
            cursor.execute("SELECT create_hypertable('transaction_streams', 'timestamp', if_not_exists => TRUE);")
            print("Created TimescaleDB hypertable for transaction_streams")
        except Exception as e:
            print(f"Hypertable creation note: {e}")
        
        # Social connections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS social_connections (
                user_id VARCHAR(50),
                connected_user_id VARCHAR(50),
                connection_type VARCHAR(30),
                strength DECIMAL(4,3),
                risk_correlation DECIMAL(6,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, connected_user_id)
            );
        """)
        
        # Mobile towers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mobile_towers (
                tower_id VARCHAR(20) PRIMARY KEY,
                latitude DECIMAL(10,7),
                longitude DECIMAL(10,7),
                coverage_radius_km DECIMAL(6,2),
                city VARCHAR(100),
                tower_type VARCHAR(20)
            );
        """)
        
        print("Database setup completed successfully!")
        
    except Exception as e:
        print(f"Database setup error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    setup_database()