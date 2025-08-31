"""
PRISM Credit Risk Management System - Dummy Data Generator
Generates realistic synthetic data for 500 users with movement, transaction, and social patterns
"""

import pandas as pd
import numpy as np
import random
import hashlib
import json
from datetime import datetime, timedelta
from faker import Faker
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import math

# Initialize Faker for Indian locale
fake = Faker(['hi_IN', 'en_IN'])
np.random.seed(42)
random.seed(42)

class PRISMDataGenerator:
    def __init__(self, num_users=500):
        self.num_users = num_users
        self.users = []
        self.towers = []
        self.connections = []
        self.movements = []
        self.transactions = []
        
        # Indian cities and coordinates
        self.cities = {
            'Mumbai': {'lat': 19.0760, 'lng': 72.8777},
            'Delhi': {'lat': 28.7041, 'lng': 77.1025},
            'Bangalore': {'lat': 12.9716, 'lng': 77.5946},
            'Chennai': {'lat': 13.0827, 'lng': 80.2707},
            'Hyderabad': {'lat': 17.3850, 'lng': 78.4867},
            'Pune': {'lat': 18.5204, 'lng': 73.8567},
            'Kolkata': {'lat': 22.5726, 'lng': 88.3639},
            'Ahmedabad': {'lat': 23.0225, 'lng': 72.5714}
        }
        
        # Merchant categories for transactions
        self.merchant_categories = [
            'grocery', 'fuel', 'restaurant', 'pharmacy', 'transport',
            'utilities', 'shopping', 'entertainment', 'education', 'healthcare',
            'bills', 'recharge', 'investment', 'insurance', 'rent'
        ]
        
        # Payment methods
        self.payment_methods = ['upi', 'card', 'netbanking', 'wallet']
        
        # Time patterns
        self.start_date = datetime.now() - timedelta(days=90)
        self.end_date = datetime.now()
    
    def calculate_entropy(self, values):
        """Calculate Shannon entropy for a list of values"""
        if not values:
            return 0.0
        
        value_counts = pd.Series(values).value_counts()
        probabilities = value_counts / len(values)
        entropy = -sum(probabilities * np.log2(probabilities))
        
        # Normalize to 0-1 range
        max_entropy = np.log2(len(set(values))) if len(set(values)) > 0 else 1
        return min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
    
    def generate_mobile_towers(self):
        """Generate mobile tower data for each city"""
        towers = []
        tower_id = 1
        
        for city, coords in self.cities.items():
            # Generate 20-30 towers per city
            num_towers = random.randint(20, 30)
            
            for i in range(num_towers):
                # Spread towers around city center
                lat_offset = np.random.normal(0, 0.05)  # ~5km radius spread
                lng_offset = np.random.normal(0, 0.05)
                
                tower = {
                    'tower_id': f'T{tower_id:04d}',
                    'tower_name': f'{city}_Tower_{i+1}',
                    'location_lat': round(coords['lat'] + lat_offset, 6),
                    'location_lng': round(coords['lng'] + lng_offset, 6),
                    'coverage_radius': random.randint(500, 2000),  # meters
                    'city': city,
                    'area_type': random.choice(['residential', 'commercial', 'industrial'])
                }
                towers.append(tower)
                tower_id += 1
        
        self.towers = towers
        return towers
    
    def generate_user_profiles(self):
        """Generate diverse user profiles with realistic Indian demographics"""
        users = []
        
        # Define risk distribution (10% high risk, 20% medium-high, 50% medium, 20% low)
        risk_distribution = (['high'] * 50 + ['medium-high'] * 100 + 
                           ['medium'] * 250 + ['low'] * 100)
        random.shuffle(risk_distribution)
        
        for i in range(self.num_users):
            phone = f"+91{random.randint(6000000000, 9999999999)}"
            phone_hash = hashlib.sha256(phone.encode()).hexdigest()
            
            city = random.choice(list(self.cities.keys()))
            risk_category = risk_distribution[i]
            
            # Risk-based score generation
            if risk_category == 'high':
                credit_score = random.randint(300, 500)
                default_prob = random.uniform(0.3, 0.8)
            elif risk_category == 'medium-high':
                credit_score = random.randint(500, 600)
                default_prob = random.uniform(0.15, 0.35)
            elif risk_category == 'medium':
                credit_score = random.randint(600, 720)
                default_prob = random.uniform(0.05, 0.18)
            else:  # low risk
                credit_score = random.randint(720, 850)
                default_prob = random.uniform(0.01, 0.08)
            
            # Generate entropy scores (lower entropy = more predictable = lower risk)
            base_entropy = 0.3 if risk_category == 'low' else 0.7 if risk_category == 'high' else 0.5
            entropy_noise = random.uniform(-0.15, 0.15)
            
            user = {
                'user_id': f'USR{i+1:05d}',
                'phone_hash': phone_hash,
                'created_at': fake.date_time_between(start_date='-2y', end_date='now'),
                'age': random.randint(18, 65),
                'gender': random.choice(['M', 'F', 'O']),
                'occupation': fake.job(),
                'income_bracket': random.choice(['low', 'medium', 'high']),
                'city': city,
                'state': fake.state(),
                
                # Entropy scores (with some correlation to risk)
                'movement_entropy': max(0, min(1, base_entropy + entropy_noise)),
                'transaction_entropy': max(0, min(1, base_entropy + random.uniform(-0.1, 0.1))),
                'social_entropy': max(0, min(1, base_entropy + random.uniform(-0.1, 0.1))),
                'temporal_entropy': max(0, min(1, base_entropy + random.uniform(-0.1, 0.1))),
                
                # Risk scores
                'credit_risk_score': credit_score,
                'default_probability': round(default_prob, 4),
                'cascade_risk_score': random.uniform(0.1, 0.9),
                'ghost_defaulter_probability': random.uniform(0.01, 0.15),
                
                'default_status': 'defaulted' if random.random() < default_prob else 'active',
                'risk_category': risk_category,
                
                # Behavioral fingerprint (simplified)
                'behavioral_fingerprint': {
                    'movement_signature': [round(random.uniform(0, 1), 4) for _ in range(10)],
                    'transaction_signature': [round(random.uniform(0, 1), 4) for _ in range(10)],
                    'temporal_signature': [round(random.uniform(0, 1), 4) for _ in range(7)],  # weekday patterns
                    'social_signature': [round(random.uniform(0, 1), 4) for _ in range(5)]
                }
            }
            
            # Calculate composite stability score
            user['composite_stability_score'] = round(
                0.3 * user['movement_entropy'] + 
                0.3 * user['transaction_entropy'] + 
                0.2 * user['social_entropy'] + 
                0.2 * user['temporal_entropy'], 3
            )
            
            users.append(user)
        
        self.users = users
        return users
    
    def generate_movement_patterns(self):
        """Generate realistic movement patterns for each user"""
        movements = []
        
        for user in self.users:
            user_city = user['city']
            city_towers = [t for t in self.towers if t['city'] == user_city]
            
            if not city_towers:
                continue
            
            # Define user's regular locations
            home_tower = random.choice(city_towers)
            work_tower = random.choice([t for t in city_towers if t != home_tower])
            social_towers = random.sample([t for t in city_towers if t not in [home_tower, work_tower]], 
                                        min(3, len(city_towers) - 2))
            
            # Generate 90 days of movement data
            current_date = self.start_date
            
            while current_date <= self.end_date:
                # Daily movement pattern
                daily_movements = self.generate_daily_movement(
                    user, current_date, home_tower, work_tower, social_towers
                )
                movements.extend(daily_movements)
                current_date += timedelta(days=1)
        
        self.movements = movements
        return movements
    
    def generate_daily_movement(self, user, date, home_tower, work_tower, social_towers):
        """Generate movement pattern for a single day"""
        movements = []
        is_weekend = date.weekday() >= 5
        
        # Start at home
        current_tower = home_tower
        current_time = date.replace(hour=random.randint(6, 8))  # Wake up time
        
        # Morning routine
        movements.append(self.create_movement_record(user, current_time, current_tower, 0, 
                                                   random.randint(60, 180)))  # Stay at home
        
        if not is_weekend and random.random() > 0.1:  # 90% go to work on weekdays
            # Go to work
            travel_time = random.randint(15, 90)  # Travel duration
            work_arrival = current_time + timedelta(minutes=random.randint(60, 120))
            
            movements.append(self.create_movement_record(user, work_arrival, work_tower, 1,
                                                       random.randint(480, 600)))  # Work duration
            current_tower = work_tower
            current_time = work_arrival + timedelta(minutes=random.randint(480, 600))
        
        # Evening activities (30% chance)
        if random.random() < 0.3 and social_towers:
            social_tower = random.choice(social_towers)
            social_time = current_time + timedelta(minutes=random.randint(30, 120))
            
            movements.append(self.create_movement_record(user, social_time, social_tower, 2,
                                                       random.randint(60, 180)))
            current_tower = social_tower
            current_time = social_time + timedelta(minutes=random.randint(60, 180))
        
        # Return home
        if current_tower != home_tower:
            home_return = current_time + timedelta(minutes=random.randint(15, 60))
            movements.append(self.create_movement_record(user, home_return, home_tower, 0,
                                                       random.randint(600, 800)))
        
        return movements
    
    def create_movement_record(self, user, timestamp, tower, cluster_id, duration):
        """Create a single movement record"""
        return {
            'user_id': user['user_id'],
            'timestamp': timestamp,
            'tower_id': tower['tower_id'],
            'signal_strength': random.randint(-120, -30),
            'location_lat': tower['location_lat'],
            'location_lng': tower['location_lng'],
            'location_cluster_id': cluster_id,
            'stay_duration': duration,
            'transition_probability': random.uniform(0.1, 0.9),
            'movement_vector': {
                'distance': random.randint(0, 15000),  # meters
                'direction': random.randint(0, 360),   # degrees
                'speed': random.randint(0, 60)         # km/h
            }
        }
    
    def generate_transactions(self):
        """Generate realistic transaction patterns"""
        transactions = []
        
        for user in self.users:
            # Transaction frequency based on income and risk
            base_transactions_per_day = 3 if user['income_bracket'] == 'high' else 2 if user['income_bracket'] == 'medium' else 1
            
            current_date = self.start_date
            transaction_id = 1
            
            while current_date <= self.end_date:
                # Generate daily transactions
                num_transactions = np.random.poisson(base_transactions_per_day)
                
                for _ in range(num_transactions):
                    # Transaction timing (realistic hourly distribution)
                    hour_weights = [0.5, 0.3, 0.2, 0.1, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8,
                                  2.0, 1.8, 1.5, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6]
                    hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
                    
                    transaction_time = current_date.replace(
                        hour=hour, 
                        minute=random.randint(0, 59),
                        second=random.randint(0, 59)
                    )
                    
                    # Amount distribution based on category
                    category = np.random.choice(self.merchant_categories)
                    amount = self.generate_transaction_amount(category, user['income_bracket'])
                    
                    transaction = {
                        'user_id': user['user_id'],
                        'timestamp': transaction_time,
                        'transaction_id': f'TXN{transaction_id:010d}',
                        'amount': round(amount, 2),
                        'transaction_type': 'debit',  # Most transactions are debits
                        'category': category,
                        'merchant_id': f'MER{random.randint(1000, 9999)}',
                        'merchant_category': category,
                        'payment_method': np.random.choice(self.payment_methods, 
                                                         p=[0.6, 0.25, 0.10, 0.05]),  # UPI dominance
                        'time_of_day': hour,
                        'day_of_week': transaction_time.weekday() + 1,
                        'is_weekend': transaction_time.weekday() >= 5,
                        'is_recurring': category in ['utilities', 'bills', 'rent'] and random.random() < 0.3,
                        'unusual_time': hour < 6 or hour > 23,
                        'unusual_amount': amount > 10000 or amount < 1
                    }
                    
                    transactions.append(transaction)
                    transaction_id += 1
                
                current_date += timedelta(days=1)
        
        self.transactions = transactions
        return transactions
    
    def generate_transaction_amount(self, category, income_bracket):
        """Generate realistic transaction amounts based on category and income"""
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
        
        # Adjust for income bracket
        if income_bracket == 'high':
            min_amt *= 1.5
            max_amt *= 2
        elif income_bracket == 'low':
            min_amt *= 0.6
            max_amt *= 0.7
        
        return random.uniform(min_amt, max_amt)
    
    def generate_social_connections(self):
        """Generate social network connections between users"""
        connections = []
        
        # Create connections based on city proximity and transaction patterns
        for i, user in enumerate(self.users):
            # Each user has 5-15 connections
            num_connections = random.randint(5, 15)
            
            # Same city users are more likely to be connected
            same_city_users = [u for u in self.users if u['city'] == user['city'] and u != user]
            other_users = [u for u in self.users if u['city'] != user['city'] and u != user]
            
            # 70% connections from same city, 30% from other cities
            same_city_count = int(num_connections * 0.7)
            other_city_count = num_connections - same_city_count
            
            potential_connections = []
            
            # Add same city connections
            if same_city_users:
                potential_connections.extend(random.sample(
                    same_city_users, 
                    min(same_city_count, len(same_city_users))
                ))
            
            # Add other city connections
            if other_users and other_city_count > 0:
                potential_connections.extend(random.sample(
                    other_users,
                    min(other_city_count, len(other_users))
                ))
            
            # Create connection records
            for connected_user in potential_connections:
                # Connection strength based on various factors
                same_city = user['city'] == connected_user['city']
                strength = random.uniform(0.3, 0.9) if same_city else random.uniform(0.1, 0.5)
                
                # Risk correlation (similar risk users tend to connect)
                risk_diff = abs(user['default_probability'] - connected_user['default_probability'])
                risk_correlation = max(0.1, 1.0 - risk_diff * 2)  # Higher correlation for similar risk
                
                connection = {
                    'user_id': user['user_id'],
                    'connected_user_id': connected_user['user_id'],
                    'connection_type': random.choice(['transaction', 'contact', 'family']),
                    'strength': round(strength, 2),
                    'frequency': random.randint(1, 50),
                    'total_transaction_amount': round(random.uniform(100, 50000), 2),
                    'last_interaction': fake.date_time_between(start_date='-30d', end_date='now'),
                    'risk_correlation': round(risk_correlation, 3)
                }
                
                connections.append(connection)
        
        self.connections = connections
        return connections
    
    def generate_behavioral_fingerprints(self):
        """Generate behavioral DNA fingerprints for each user"""
        fingerprints = []
        
        for user in self.users:
            # Generate feature vectors based on user's patterns
            movement_vector = [round(random.uniform(0, 1), 6) for _ in range(20)]
            transaction_vector = [round(random.uniform(0, 1), 6) for _ in range(15)]
            temporal_vector = [round(random.uniform(0, 1), 6) for _ in range(12)]  # hourly patterns
            social_vector = [round(random.uniform(0, 1), 6) for _ in range(10)]
            
            # Create fingerprint hash from combined vectors
            combined_vector = movement_vector + transaction_vector + temporal_vector + social_vector
            fingerprint_string = ''.join([str(x) for x in combined_vector])
            fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
            
            fingerprint = {
                'user_id': user['user_id'],
                'fingerprint_hash': fingerprint_hash,
                'movement_pattern_vector': movement_vector,
                'transaction_pattern_vector': transaction_vector,
                'temporal_pattern_vector': temporal_vector,
                'social_pattern_vector': social_vector,
                'similarity_threshold': 0.85,
                'last_updated': datetime.now(),
                'created_at': user['created_at']
            }
            
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    def save_to_csv(self, output_dir='data'):
        """Save all generated data to CSV files"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save users
        users_df = pd.DataFrame(self.users)
        users_df.to_csv(f'{output_dir}/user_profiles.csv', index=False)
        
        # Save towers
        towers_df = pd.DataFrame(self.towers)
        towers_df.to_csv(f'{output_dir}/mobile_towers.csv', index=False)
        
        # Save movements
        movements_df = pd.DataFrame(self.movements)
        movements_df.to_csv(f'{output_dir}/movement_patterns.csv', index=False)
        
        # Save transactions
        transactions_df = pd.DataFrame(self.transactions)
        transactions_df.to_csv(f'{output_dir}/transaction_streams.csv', index=False)
        
        # Save connections
        connections_df = pd.DataFrame(self.connections)
        connections_df.to_csv(f'{output_dir}/social_connections.csv', index=False)
        
        print(f"Data saved to {output_dir}/ directory")
        print(f"Generated {len(self.users)} users with:")
        print(f"  - {len(self.towers)} mobile towers")
        print(f"  - {len(self.movements)} movement records")
        print(f"  - {len(self.transactions)} transaction records") 
        print(f"  - {len(self.connections)} social connections")
    
    def insert_to_database(self, db_config):
        """Insert generated data into PostgreSQL database"""
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            print("Inserting data into database...")
            
            # Insert mobile towers
            tower_data = [
                (t['tower_id'], t['tower_name'], t['location_lat'], t['location_lng'],
                 t['coverage_radius'], t['city'], t['area_type'])
                for t in self.towers
            ]
            
            cursor.executemany("""
                INSERT INTO mobile_towers (tower_id, tower_name, location_lat, location_lng,
                                         coverage_radius, city, area_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tower_id) DO NOTHING
            """, tower_data)
            
            # Insert user profiles
            user_data = [
                (u['user_id'], u['phone_hash'], u['created_at'], u['age'], u['gender'],
                 u['occupation'], u['income_bracket'], u['city'], u['state'],
                 u['movement_entropy'], u['transaction_entropy'], u['social_entropy'],
                 u['temporal_entropy'], u['composite_stability_score'],
                 u['credit_risk_score'], u['default_probability'], u['cascade_risk_score'],
                 u['ghost_defaulter_probability'], u['default_status'], u['risk_category'],
                 json.dumps(u['behavioral_fingerprint']))
                for u in self.users
            ]
            
            cursor.executemany("""
                INSERT INTO user_profiles (
                    user_id, phone_hash, created_at, age, gender, occupation, income_bracket,
                    city, state, movement_entropy, transaction_entropy, social_entropy,
                    temporal_entropy, composite_stability_score, credit_risk_score,
                    default_probability, cascade_risk_score, ghost_defaulter_probability,
                    default_status, risk_category, behavioral_fingerprint
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, user_data)
            
            # Insert movement patterns (batch insert for performance)
            movement_data = [
                (m['user_id'], m['timestamp'], m['tower_id'], m['signal_strength'],
                 m['location_lat'], m['location_lng'], m['location_cluster_id'],
                 m['stay_duration'], m['transition_probability'], json.dumps(m['movement_vector']))
                for m in self.movements
            ]
            
            # Insert in batches of 1000 for performance
            batch_size = 1000
            for i in range(0, len(movement_data), batch_size):
                batch = movement_data[i:i+batch_size]
                cursor.executemany("""
                    INSERT INTO movement_patterns (
                        user_id, timestamp, tower_id, signal_strength, location_lat,
                        location_lng, location_cluster_id, stay_duration,
                        transition_probability, movement_vector
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, timestamp) DO NOTHING
                """, batch)
                conn.commit()  # Commit each batch
                print(f"Inserted movement batch {i//batch_size + 1}/{len(movement_data)//batch_size + 1}")
            
            # Insert transactions
            transaction_data = [
                (t['user_id'], t['timestamp'], t['transaction_id'], t['amount'],
                 t['transaction_type'], t['category'], t['merchant_id'], t['merchant_category'],
                 t['payment_method'], t['time_of_day'], t['day_of_week'], t['is_weekend'],
                 t['is_recurring'], t['unusual_time'], t['unusual_amount'])
                for t in self.transactions
            ]
            
            for i in range(0, len(transaction_data), batch_size):
                batch = transaction_data[i:i+batch_size]
                cursor.executemany("""
                    INSERT INTO transaction_streams (
                        user_id, timestamp, transaction_id, amount, transaction_type,
                        category, merchant_id, merchant_category, payment_method,
                        time_of_day, day_of_week, is_weekend, is_recurring,
                        unusual_time, unusual_amount
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                """, batch)
                conn.commit()
                print(f"Inserted transaction batch {i//batch_size + 1}/{len(transaction_data)//batch_size + 1}")
            
            # Insert social connections
            connection_data = [
                (c['user_id'], c['connected_user_id'], c['connection_type'], c['strength'],
                 c['frequency'], c['total_transaction_amount'], c['last_interaction'],
                 c['risk_correlation'])
                for c in self.connections
            ]
            
            cursor.executemany("""
                INSERT INTO social_connections (
                    user_id, connected_user_id, connection_type, strength,
                    frequency, total_transaction_amount, last_interaction, risk_correlation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, connected_user_id) DO NOTHING
            """, connection_data)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("Successfully inserted all data into database!")
            
        except Exception as e:
            print(f"Error inserting data: {e}")
            if conn:
                conn.rollback()
                conn.close()
    
    def generate_all_data(self):
        """Generate all synthetic data"""
        print("Generating mobile towers...")
        self.generate_mobile_towers()
        
        print("Generating user profiles...")
        self.generate_user_profiles()
        
        print("Generating movement patterns...")
        self.generate_movement_patterns()
        
        print("Generating transactions...")
        self.generate_transactions()
        
        print("Generating social connections...")
        self.generate_social_connections()
        
        print("Data generation complete!")
        
        return {
            'users': len(self.users),
            'towers': len(self.towers),
            'movements': len(self.movements),
            'transactions': len(self.transactions),
            'connections': len(self.connections)
        }


# Main execution
if __name__ == "__main__":
    # Initialize data generator
    generator = PRISMDataGenerator(num_users=500)
    
    # Generate all data
    stats = generator.generate_all_data()
    
    print(f"\nGenerated synthetic data:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Save to CSV files
    generator.save_to_csv('data')
    
    # Database configuration (update with your credentials)
    db_config = {
        'host': 'localhost',
        'database': 'prism_db',
        'user': 'postgres',
        'password': 'your_password',
        'port': 5432
    }
    
    # Uncomment to insert into database
    # generator.insert_to_database(db_config)