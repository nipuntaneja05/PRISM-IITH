"""
scripts/fix_missing_data.py
Targeted fix to insert only missing transaction and social data
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import json

load_dotenv()
fake = Faker(['hi_IN', 'en_IN'])

class TargetedDataFixer:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'prism_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        # Transaction categories
        self.categories = [
            'grocery', 'fuel', 'restaurant', 'pharmacy', 'transport',
            'utilities', 'shopping', 'entertainment', 'education', 'healthcare',
            'bills', 'recharge', 'investment', 'insurance', 'rent'
        ]
        
        self.payment_methods = ['upi', 'card', 'netbanking', 'wallet']

    def get_existing_users(self):
        """Get existing users from database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT user_id, age, income_bracket, city FROM user_profiles ORDER BY user_id")
        users = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return users

    def generate_transactions_for_user(self, user):
        """Generate transactions for a single user"""
        transactions = []
        
        # Base transaction frequency
        base_transactions_per_day = {
            'high': 3,
            'medium': 2, 
            'low': 1
        }.get(user['income_bracket'], 2)
        
        # Generate for 90 days
        start_date = datetime.now() - timedelta(days=90)
        
        transaction_id = random.randint(1000000, 9999999)
        
        for day in range(90):
            current_date = start_date + timedelta(days=day)
            
            # Number of transactions for this day
            num_transactions = np.random.poisson(base_transactions_per_day)
            
            for _ in range(num_transactions):
                # Random hour (weighted towards daytime)
                hour_weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0,
                               2.0, 1.8, 1.5, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6]
                hour = int(np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights)))
                
                # Transaction time
                transaction_time = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59)
                )
                
                # Category and amount
                category = random.choice(self.categories)
                amount = self.generate_amount(category, user['income_bracket'])
                
                transaction = {
                    'user_id': user['user_id'],
                    'timestamp': transaction_time,
                    'transaction_id': f'TXN{transaction_id:010d}',
                    'amount': round(float(amount), 2),  # Convert to native float
                    'transaction_type': 'debit',
                    'category': category,
                    'merchant_id': f'MER{random.randint(1000, 9999)}',
                    'merchant_category': category,
                    'payment_method': str(np.random.choice(self.payment_methods, p=[0.6, 0.25, 0.10, 0.05])),  # Convert to native str
                    'time_of_day': int(hour),  # Convert to native int
                    'day_of_week': int(transaction_time.weekday() + 1),  # Convert to native int
                    'is_weekend': bool(transaction_time.weekday() >= 5),  # Convert to native bool
                    'is_recurring': bool(category in ['utilities', 'bills', 'rent'] and random.random() < 0.3),
                    'unusual_time': bool(hour < 6 or hour > 23),
                    'unusual_amount': bool(amount > 10000 or amount < 1)
                }
                
                transactions.append(transaction)
                transaction_id += 1
        
        return transactions

    def generate_amount(self, category, income_bracket):
        """Generate realistic transaction amounts"""
        base_amounts = {
            'grocery': (100, 2000),
            'fuel': (500, 3000),
            'restaurant': (200, 1500),
            'pharmacy': (50, 500),
            'transport': (20, 500),
            'utilities': (500, 5000),
            'shopping': (300, 10000),
            'entertainment': (200, 3000),
            'education': (1000, 50000),
            'healthcare': (200, 10000),
            'bills': (100, 2000),
            'recharge': (100, 500),
            'investment': (1000, 100000),
            'insurance': (2000, 25000),
            'rent': (5000, 50000)
        }
        
        min_amt, max_amt = base_amounts.get(category, (100, 1000))
        
        # Adjust for income
        if income_bracket == 'high':
            min_amt *= 1.5
            max_amt *= 2
        elif income_bracket == 'low':
            min_amt *= 0.6
            max_amt *= 0.7
        
        return random.uniform(min_amt, max_amt)

    def generate_social_connections(self, users):
        """Generate social connections between users"""
        connections = []
        
        for user in users:
            # Each user has 5-15 connections
            num_connections = random.randint(5, 15)
            
            # Same city users more likely to connect
            same_city_users = [u for u in users if u['city'] == user['city'] and u['user_id'] != user['user_id']]
            other_users = [u for u in users if u['city'] != user['city'] and u['user_id'] != user['user_id']]
            
            # 70% same city, 30% other cities
            same_city_count = int(num_connections * 0.7)
            other_city_count = num_connections - same_city_count
            
            potential_connections = []
            
            if same_city_users:
                potential_connections.extend(random.sample(
                    same_city_users,
                    min(same_city_count, len(same_city_users))
                ))
            
            if other_users and other_city_count > 0:
                potential_connections.extend(random.sample(
                    other_users,
                    min(other_city_count, len(other_users))
                ))
            
            # Create connections
            for connected_user in potential_connections:
                same_city = user['city'] == connected_user['city']
                strength = random.uniform(0.3, 0.9) if same_city else random.uniform(0.1, 0.5)
                
                connection = {
                    'user_id': user['user_id'],
                    'connected_user_id': connected_user['user_id'],
                    'connection_type': random.choice(['family', 'friend', 'transaction', 'contact']),
                    'strength': round(strength, 2),
                    'frequency': random.randint(1, 50),
                    'total_transaction_amount': round(random.uniform(100, 50000), 2),
                    'last_interaction': fake.date_time_between(start_date='-30d', end_date='now'),
                    'risk_correlation': round(random.uniform(0.1, 0.9), 3)
                }
                
                connections.append(connection)
        
        return connections

    def insert_transactions(self, transactions):
        """Insert transactions into database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        print(f"Inserting {len(transactions)} transactions...")
        
        batch_size = 1000
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i+batch_size]
            
            transaction_data = [
                (t['user_id'], t['timestamp'], t['transaction_id'], t['amount'],
                 t['transaction_type'], t['category'], t['merchant_id'], t['merchant_category'],
                 t['payment_method'], t['time_of_day'], t['day_of_week'], t['is_weekend'],
                 t['is_recurring'], t['unusual_time'], t['unusual_amount'])
                for t in batch
            ]
            
            try:
                cursor.executemany("""
                    INSERT INTO transaction_streams (
                        user_id, timestamp, transaction_id, amount, transaction_type,
                        category, merchant_id, merchant_category, payment_method,
                        time_of_day, day_of_week, is_weekend, is_recurring,
                        unusual_time, unusual_amount
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                """, transaction_data)
                
                conn.commit()
                print(f"  Inserted batch {i//batch_size + 1}/{len(transactions)//batch_size + 1}")
                
            except Exception as e:
                print(f"  Error in batch {i//batch_size + 1}: {e}")
                conn.rollback()
        
        conn.close()

    def insert_social_connections(self, connections):
        """Insert social connections into database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        print(f"Inserting {len(connections)} social connections...")
        
        connection_data = [
            (c['user_id'], c['connected_user_id'], c['connection_type'], c['strength'],
             c['frequency'], c['total_transaction_amount'], c['last_interaction'],
             c['risk_correlation'])
            for c in connections
        ]
        
        try:
            cursor.executemany("""
                INSERT INTO social_connections (
                    user_id, connected_user_id, connection_type, strength,
                    frequency, total_transaction_amount, last_interaction, risk_correlation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, connected_user_id) DO NOTHING
            """, connection_data)
            
            conn.commit()
            print("Social connections inserted successfully!")
            
        except Exception as e:
            print(f"Error inserting social connections: {e}")
            conn.rollback()
        
        conn.close()

    def fix_data(self):
        """Main data fixing process"""
        print("Loading existing users...")
        users = self.get_existing_users()
        print(f"Found {len(users)} users")
        
        print("Generating transactions...")
        all_transactions = []
        for i, user in enumerate(users):
            if i % 50 == 0:
                print(f"  Processing user {i+1}/{len(users)}")
            
            user_transactions = self.generate_transactions_for_user(user)
            all_transactions.extend(user_transactions)
        
        print(f"Generated {len(all_transactions)} total transactions")
        
        print("Generating social connections...")
        connections = self.generate_social_connections(users)
        print(f"Generated {len(connections)} connections")
        
        # Insert data
        self.insert_transactions(all_transactions)
        self.insert_social_connections(connections)
        
        print("Data insertion completed!")

def main():
    print("PRISM Targeted Data Fix")
    print("=" * 30)
    
    fixer = TargetedDataFixer()
    fixer.fix_data()
    
    # Verify
    conn = psycopg2.connect(**fixer.db_config)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM transaction_streams")
    transaction_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM social_connections")
    social_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\nFinal counts:")
    print(f"Transactions: {transaction_count}")
    print(f"Social connections: {social_count}")
    
    if transaction_count > 0 and social_count > 0:
        print("\n✅ SUCCESS! Ready to run training.")
        print("Next: python scripts/train_models.py")
        return True
    else:
        print("\n❌ FAILED! Data still missing.")
        return False

if __name__ == "__main__":
    main()