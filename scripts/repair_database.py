"""
scripts/repair_database.py
Quick fix to check and repair missing transaction/social data
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.generate_data import PRISMDataGenerator
import json
from datetime import datetime

load_dotenv()

def check_database_status():
    """Check what data actually exists in database"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'prism_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("Database Status Check:")
        print("=" * 40)
        
        # Check users
        cursor.execute("SELECT COUNT(*) as count FROM user_profiles")
        user_count = cursor.fetchone()['count']
        print(f"Users: {user_count}")
        
        # Check movements
        cursor.execute("SELECT COUNT(*) as count FROM movement_patterns")
        movement_count = cursor.fetchone()['count']
        print(f"Movement records: {movement_count}")
        
        # Check transactions
        cursor.execute("SELECT COUNT(*) as count FROM transaction_streams")
        transaction_count = cursor.fetchone()['count']
        print(f"Transaction records: {transaction_count}")
        
        # Check social connections
        cursor.execute("SELECT COUNT(*) as count FROM social_connections")
        social_count = cursor.fetchone()['count']
        print(f"Social connections: {social_count}")
        
        # Check towers
        cursor.execute("SELECT COUNT(*) as count FROM mobile_towers")
        tower_count = cursor.fetchone()['count']
        print(f"Mobile towers: {tower_count}")
        
        conn.close()
        
        # Diagnosis
        print(f"\nDiagnosis:")
        if transaction_count == 0:
            print("❌ MISSING: Transaction data")
        if social_count == 0:
            print("❌ MISSING: Social connection data")
        
        if transaction_count == 0 or social_count == 0:
            print("\n🔧 SOLUTION: Re-run data generation with proper database insertion")
            return False
        else:
            print("✅ All data present and ready for training")
            return True
            
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

def fix_missing_data():
    """Generate and insert missing transaction/social data"""
    
    print("\nRegenerating missing data...")
    
    # Initialize generator
    generator = PRISMDataGenerator(num_users=500)
    
    # Load existing users from database
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'prism_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get existing users
    cursor.execute("SELECT user_id, age, income_bracket, city FROM user_profiles ORDER BY user_id")
    existing_users = cursor.fetchall()
    
    print(f"Found {len(existing_users)} existing users")
    
    # Convert to generator format
    generator.users = []
    for user in existing_users:
        generator.users.append({
            'user_id': user['user_id'],
            'age': user['age'],
            'income_bracket': user['income_bracket'],
            'city': user['city'],
            'default_probability': 0.15,  # Default value
            'composite_stability_score': 0.5  # Default value
        })
    
    # Load existing towers
    cursor.execute("SELECT * FROM mobile_towers")
    tower_data = cursor.fetchall()
    generator.towers = [dict(t) for t in tower_data]
    
    conn.close()
    
    print("Generating transactions...")
    generator.generate_transactions()
    print(f"Generated {len(generator.transactions)} transactions")
    
    print("Generating social connections...")
    generator.generate_social_connections()
    print(f"Generated {len(generator.connections)} connections")
    
    # Insert new data
    print("Inserting into database...")
    generator.insert_to_database(db_config)
    
    print("✅ Data repair completed!")

def main():
    print("PRISM Database Repair Tool")
    print("=" * 40)
    
    # Check current status
    if check_database_status():
        print("Database is healthy - no repair needed")
        return True
    
    # Repair if needed
    print("\nStarting database repair...")
    fix_missing_data()
    
    # Verify repair
    print("\nVerifying repair...")
    if check_database_status():
        print("✅ Database repair successful!")
        return True
    else:
        print("❌ Database repair failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to run: python scripts/train_models.py")
    else:
        print("\n❌ Please check database connection and try again")