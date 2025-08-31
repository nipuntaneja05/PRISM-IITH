import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from pathlib import Path

def load_data_to_database():
    """Load CSV data into the database"""
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Database connection
    db_params = {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "prism_db"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", 5432)),
    }
    
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
    try:
        # Load user profiles
        print("Loading user profiles...")
        df = pd.read_csv(data_dir / "user_profiles.csv")
        execute_values(
            cursor,
            """INSERT INTO user_profiles (user_id, age, income, city, occupation, 
               credit_risk_score, default_probability, risk_category, default_status) 
               VALUES %s ON CONFLICT (user_id) DO NOTHING""",
            df.values.tolist(),
            template=None
        )
        
        # Load mobile towers
        print("Loading mobile towers...")
        df = pd.read_csv(data_dir / "mobile_towers.csv")
        execute_values(
            cursor,
            """INSERT INTO mobile_towers (tower_id, latitude, longitude, 
               coverage_radius_km, city, tower_type) VALUES %s 
               ON CONFLICT (tower_id) DO NOTHING""",
            df.values.tolist()
        )
        
        # Load movement patterns
        print("Loading movement patterns...")
        df = pd.read_csv(data_dir / "movement_patterns.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        execute_values(
            cursor,
            """INSERT INTO movement_patterns (user_id, timestamp, latitude, longitude,
               location_type, duration_minutes, distance_from_home_km, tower_id) 
               VALUES %s""",
            df.values.tolist()
        )
        
        # Load transaction streams
        print("Loading transaction streams...")
        df = pd.read_csv(data_dir / "transaction_streams.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        execute_values(
            cursor,
            """INSERT INTO transaction_streams (user_id, timestamp, amount, category,
               merchant, payment_method, location, is_weekend, hour_of_day) 
               VALUES %s""",
            df.values.tolist()
        )
        
        # Load social connections
        print("Loading social connections...")
        df = pd.read_csv(data_dir / "social_connections.csv")
        execute_values(
            cursor,
            """INSERT INTO social_connections (user_id, connected_user_id, 
               connection_type, strength, risk_correlation) VALUES %s
               ON CONFLICT (user_id, connected_user_id) DO NOTHING""",
            df.values.tolist()
        )
        
        conn.commit()
        print("Data loading completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"Data loading error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    load_data_to_database()