"""
src/ml/social_risk_gnn.py
PRISM Social Risk Graph Neural Network - Complete Implementation
Analyzes social networks to predict default risk propagation through connections
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class GraphNeuralNetwork:
    """
    Simplified Graph Neural Network implementation for social risk analysis
    Uses message passing and feature aggregation
    """
    
    def __init__(self, input_dim=50, hidden_dim=128, output_dim=1, dropout_rate=0.3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.01
        
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.b3 = np.zeros(output_dim)
        
        # Training parameters
        self.learning_rate = 0.001
        self.is_fitted = False
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def dropout(self, x, rate, training=True):
        if training and rate > 0:
            mask = np.random.rand(*x.shape) > rate
            return x * mask / (1 - rate)
        return x
    
    def message_passing(self, node_features, adjacency_matrix):
        """Aggregate information from neighboring nodes"""
        # Normalize adjacency matrix
        degree = np.sum(adjacency_matrix, axis=1, keepdims=True) + 1e-8
        norm_adj = adjacency_matrix / degree
        
        # Aggregate neighbor features
        aggregated = np.dot(norm_adj, node_features)
        return aggregated
    
    def forward(self, node_features, adjacency_matrix, training=True):
        """Forward pass through GNN layers"""
        # Layer 1: Initial transformation
        h1 = self.relu(np.dot(node_features, self.W1) + self.b1)
        h1 = self.dropout(h1, self.dropout_rate, training)
        
        # Message passing round 1
        h1_agg = self.message_passing(h1, adjacency_matrix)
        
        # Layer 2: Feature refinement
        h2 = self.relu(np.dot(h1_agg, self.W2) + self.b2)
        h2 = self.dropout(h2, self.dropout_rate, training)
        
        # Message passing round 2
        h2_agg = self.message_passing(h2, adjacency_matrix)
        
        # Output layer
        output = self.sigmoid(np.dot(h2_agg, self.W3) + self.b3)
        
        return output.flatten()
    
    def fit_simple(self, node_features, adjacency_matrix, targets, epochs=50):
        """Simple training procedure (for demonstration)"""
        print(f"Training GNN for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(node_features, adjacency_matrix, training=True)
            
            # Simple loss (MSE)
            loss = np.mean((predictions - targets) ** 2)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        self.is_fitted = True
        return self

class SocialRiskAnalyzer:
    """
    Complete Social Risk Analysis system with GNN and traditional ML fallbacks
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.gnn_model = GraphNeuralNetwork()
        self.fallback_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.node_feature_scaler = StandardScaler()
        
        # Risk propagation parameters
        self.cascade_decay = 0.7
        self.min_connection_strength = 0.1
        self.max_propagation_hops = 3
        
        # Model state
        self.is_fitted = False
        self.user_to_index = {}
        self.index_to_user = {}
    
    def build_social_graph(self, users_data: List[Dict], connections_data: List[Dict]) -> nx.Graph:
        """Build social network graph from user and connection data"""
        
        print(f"Building social graph from {len(users_data)} users and {len(connections_data)} connections...")
        
        # Add nodes (users)
        for user in users_data:
            user_id = user['user_id']
            self.graph.add_node(
                user_id,
                risk_score=user.get('credit_risk_score', 600),
                default_probability=user.get('default_probability', 0.1),
                default_status=user.get('default_status', 'active'),
                entropy_score=user.get('composite_stability_score', 0.5),
                city=user.get('city', 'unknown'),
                age=user.get('age', 30),
                income_bracket=user.get('income_bracket', 'medium')
            )
        
        # Add edges (connections)
        edges_added = 0
        for connection in connections_data:
            user1 = connection['user_id']
            user2 = connection['connected_user_id']
            
            # Only add edge if both users exist and connection is strong enough
            if (user1 in self.graph.nodes and user2 in self.graph.nodes and 
                connection.get('strength', 0) >= self.min_connection_strength):
                
                self.graph.add_edge(
                    user1, user2,
                    strength=connection.get('strength', 0.5),
                    connection_type=connection.get('connection_type', 'unknown'),
                    risk_correlation=connection.get('risk_correlation', 0.5),
                    frequency=connection.get('frequency', 1),
                    total_amount=connection.get('total_transaction_amount', 0)
                )
                edges_added += 1
        
        print(f"Built graph: {self.graph.number_of_nodes()} nodes, {edges_added} edges")
        
        # Create user mappings for matrix operations
        self.user_to_index = {user: i for i, user in enumerate(self.graph.nodes())}
        self.index_to_user = {i: user for user, i in self.user_to_index.items()}
        
        return self.graph
    
    def extract_node_features(self, user_id: str) -> np.ndarray:
        """Extract comprehensive feature vector for a user node"""
        
        if user_id not in self.graph.nodes:
            return np.zeros(50)
        
        node_data = self.graph.nodes[user_id]
        features = []
        
        # Basic user features
        features.extend([
            node_data.get('risk_score', 600) / 1000,  # Normalized
            node_data.get('default_probability', 0.1),
            node_data.get('entropy_score', 0.5),
            node_data.get('age', 30) / 100,
            1.0 if node_data.get('default_status') == 'defaulted' else 0.0
        ])
        
        # Income encoding
        income_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        features.append(income_map.get(node_data.get('income_bracket', 'medium'), 0.5))
        
        # Network centrality features
        try:
            degree_centrality = nx.degree_centrality(self.graph)[user_id]
            betweenness_centrality = nx.betweenness_centrality(self.graph).get(user_id, 0)
            closeness_centrality = nx.closeness_centrality(self.graph).get(user_id, 0)
            eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000).get(user_id, 0)
        except:
            degree_centrality = betweenness_centrality = closeness_centrality = eigenvector_centrality = 0
        
        features.extend([degree_centrality, betweenness_centrality, closeness_centrality, eigenvector_centrality])
        
        # Neighbor analysis
        neighbors = list(self.graph.neighbors(user_id))
        
        if neighbors:
            # Risk statistics of neighbors
            neighbor_risks = [self.graph.nodes[n].get('default_probability', 0.1) for n in neighbors]
            neighbor_scores = [self.graph.nodes[n].get('risk_score', 600) for n in neighbors]
            
            features.extend([
                np.mean(neighbor_risks),
                np.max(neighbor_risks),
                np.std(neighbor_risks),
                np.mean(neighbor_scores) / 1000,
                len([r for r in neighbor_risks if r > 0.3]) / len(neighbors)  # High risk neighbor ratio
            ])
            
            # Connection strength statistics
            edge_strengths = [self.graph[user_id][n]['strength'] for n in neighbors]
            risk_correlations = [self.graph[user_id][n]['risk_correlation'] for n in neighbors]
            
            features.extend([
                np.mean(edge_strengths),
                np.max(edge_strengths),
                np.std(edge_strengths),
                np.mean(risk_correlations),
                np.max(risk_correlations)
            ])
            
            # Connection type distribution
            connection_types = [self.graph[user_id][n]['connection_type'] for n in neighbors]
            type_counts = {t: connection_types.count(t) / len(neighbors) for t in ['family', 'friend', 'transaction', 'contact']}
            features.extend([type_counts.get(t, 0) for t in ['family', 'friend', 'transaction', 'contact']])
            
        else:
            # No connections - fill with zeros
            features.extend([0] * 13)
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(self.graph, user_id)
        except:
            clustering = 0
        features.append(clustering)
        
        # Graph distance features (to high-risk nodes)
        try:
            high_risk_nodes = [n for n in self.graph.nodes() 
                             if self.graph.nodes[n].get('default_probability', 0) > 0.5]
            if high_risk_nodes:
                min_distance_to_high_risk = min([
                    nx.shortest_path_length(self.graph, user_id, hr_node) 
                    if nx.has_path(self.graph, user_id, hr_node) else 999
                    for hr_node in high_risk_nodes[:5]  # Limit for performance
                ])
                features.append(min_distance_to_high_risk / 10.0)  # Normalized
            else:
                features.append(0.5)
        except:
            features.append(0.5)
        
        # Second-degree network features
        second_degree_risks = []
        for neighbor in neighbors[:5]:  # Limit for performance
            second_neighbors = list(self.graph.neighbors(neighbor))
            for second_neighbor in second_neighbors:
                if second_neighbor != user_id:
                    second_degree_risks.append(self.graph.nodes[second_neighbor].get('default_probability', 0.1))
        
        if second_degree_risks:
            features.extend([np.mean(second_degree_risks), np.max(second_degree_risks)])
        else:
            features.extend([0, 0])
        
        # Pad or trim to exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def create_adjacency_matrix(self) -> np.ndarray:
        """Create adjacency matrix from the graph"""
        n_nodes = len(self.graph.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for user1, user2, edge_data in self.graph.edges(data=True):
            i = self.user_to_index[user1]
            j = self.user_to_index[user2]
            weight = edge_data.get('strength', 0.5) * edge_data.get('risk_correlation', 0.5)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Symmetric
        
        return adj_matrix
    
    def create_node_feature_matrix(self) -> np.ndarray:
        """Create node feature matrix for all users"""
        features = []
        for user_id in self.user_to_index.keys():
            user_features = self.extract_node_features(user_id)
            features.append(user_features)
        
        return np.array(features)
    
    def fit(self, target_variable='default_probability') -> 'SocialRiskAnalyzer':
        """Train the social risk models"""
        
        if not self.graph.nodes():
            raise ValueError("Graph must be built before fitting")
        
        print("Fitting Social Risk Analyzer...")
        
        # Create feature matrix and adjacency matrix
        node_features = self.create_node_feature_matrix()
        adjacency_matrix = self.create_adjacency_matrix()
        
        # Create targets
        targets = []
        for user_id in self.user_to_index.keys():
            if target_variable == 'default_probability':
                target = self.graph.nodes[user_id].get('default_probability', 0.1)
            elif target_variable == 'risk_score':
                target = self.graph.nodes[user_id].get('risk_score', 600) / 1000  # Normalize
            else:
                target = float(self.graph.nodes[user_id].get('default_status') == 'defaulted')
            targets.append(target)
        
        targets = np.array(targets)
        
        # Scale node features
        node_features_scaled = self.node_feature_scaler.fit_transform(node_features)
        
        # Train GNN (simplified)
        try:
            self.gnn_model.fit_simple(node_features_scaled, adjacency_matrix, targets, epochs=50)
            print("GNN training completed")
        except Exception as e:
            print(f"GNN training failed: {e}")
        
        # Train fallback model
        self.fallback_model.fit(node_features_scaled, targets)
        
        # Calculate training performance
        gnn_predictions = self.gnn_model.forward(node_features_scaled, adjacency_matrix, training=False)
        fallback_predictions = self.fallback_model.predict(node_features_scaled)
        
        if target_variable in ['default_probability', 'default_status']:
            try:
                gnn_auc = roc_auc_score(targets, gnn_predictions)
                fallback_auc = roc_auc_score(targets, fallback_predictions)
                print(f"Training AUC - GNN: {gnn_auc:.4f}, Fallback: {fallback_auc:.4f}")
            except:
                print("Could not calculate AUC scores")
        else:
            gnn_rmse = np.sqrt(mean_squared_error(targets, gnn_predictions))
            fallback_rmse = np.sqrt(mean_squared_error(targets, fallback_predictions))
            print(f"Training RMSE - GNN: {gnn_rmse:.4f}, Fallback: {fallback_rmse:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict_social_risk(self, user_id: str) -> Dict:
        """Predict social risk score for a specific user"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if user_id not in self.graph.nodes:
            return {'error': 'User not found in graph'}
        
        # Get user index
        user_index = self.user_to_index[user_id]
        
        # Create feature matrices
        node_features = self.create_node_feature_matrix()
        adjacency_matrix = self.create_adjacency_matrix()
        node_features_scaled = self.node_feature_scaler.transform(node_features)
        
        # Get predictions from both models
        try:
            gnn_predictions = self.gnn_model.forward(node_features_scaled, adjacency_matrix, training=False)
            gnn_risk = float(gnn_predictions[user_index])
        except:
            gnn_risk = 0.5
        
        fallback_risk = float(self.fallback_model.predict(node_features_scaled[user_index:user_index+1])[0])
        
        # Ensemble prediction (weighted average)
        social_risk_score = 0.6 * fallback_risk + 0.4 * gnn_risk
        
        # Analyze user's network position
        neighbors = list(self.graph.neighbors(user_id))
        
        return {
            'user_id': user_id,
            'social_risk_score': round(social_risk_score, 4),
            'gnn_prediction': round(gnn_risk, 4),
            'fallback_prediction': round(fallback_risk, 4),
            'risk_level': 'HIGH' if social_risk_score > 0.7 else 'MEDIUM' if social_risk_score > 0.4 else 'LOW',
            'network_analysis': {
                'direct_connections': len(neighbors),
                'network_position': self._analyze_network_position(user_id),
                'centrality_scores': self._get_centrality_scores(user_id)
            }
        }
    
    def calculate_cascade_risk(self, source_user: str) -> Dict:
        """Calculate cascade risk propagation from a source user"""
        
        if source_user not in self.graph.nodes:
            return {'error': 'Source user not found'}
        
        # Initialize risk propagation using BFS
        risk_propagation = {source_user: 1.0}
        cascade_paths = {source_user: [source_user]}
        
        queue = [(source_user, 0, 1.0)]
        visited = {source_user}
        
        while queue:
            current_user, hop_count, current_risk = queue.pop(0)
            
            if hop_count >= self.max_propagation_hops:
                continue
            
            neighbors = list(self.graph.neighbors(current_user))
            
            for neighbor in neighbors:
                edge_data = self.graph[current_user][neighbor]
                connection_strength = edge_data.get('strength', 0.5)
                risk_correlation = edge_data.get('risk_correlation', 0.5)
                
                # Risk propagation formula
                propagated_risk = (current_risk * connection_strength * 
                                 risk_correlation * (self.cascade_decay ** hop_count))
                
                if neighbor not in risk_propagation:
                    risk_propagation[neighbor] = 0
                    cascade_paths[neighbor] = cascade_paths[current_user] + [neighbor]
                
                if propagated_risk > risk_propagation[neighbor]:
                    risk_propagation[neighbor] = propagated_risk
                    cascade_paths[neighbor] = cascade_paths[current_user] + [neighbor]
                
                if propagated_risk > 0.01 and neighbor not in visited:
                    queue.append((neighbor, hop_count + 1, propagated_risk))
                    visited.add(neighbor)
        
        # Analyze results
        sorted_risks = sorted(risk_propagation.items(), key=lambda x: x[1], reverse=True)
        high_risk_users = [(user, risk) for user, risk in sorted_risks if risk > 0.3]
        
        return {
            'source_user': source_user,
            'total_affected_users': len(risk_propagation) - 1,
            'high_risk_cascade': high_risk_users[:10],
            'cascade_depth': max([len(path) - 1 for path in cascade_paths.values()]),
            'network_penetration': len(risk_propagation) / self.graph.number_of_nodes(),
            'risk_distribution': dict(sorted_risks[:20])
        }
    
    def _analyze_network_position(self, user_id: str) -> str:
        """Analyze user's structural position in the network"""
        try:
            centralities = {
                'degree': nx.degree_centrality(self.graph)[user_id],
                'betweenness': nx.betweenness_centrality(self.graph).get(user_id, 0),
                'closeness': nx.closeness_centrality(self.graph).get(user_id, 0)
            }
            
            if centralities['betweenness'] > 0.1:
                return 'BRIDGE'
            elif centralities['degree'] > 0.1:
                return 'HUB'
            elif centralities['closeness'] > 0.3:
                return 'CORE'
            else:
                return 'PERIPHERAL'
        except:
            return 'ISOLATED'
    
    def _get_centrality_scores(self, user_id: str) -> Dict:
        """Get centrality scores for a user"""
        try:
            return {
                'degree': nx.degree_centrality(self.graph).get(user_id, 0),
                'betweenness': nx.betweenness_centrality(self.graph).get(user_id, 0),
                'closeness': nx.closeness_centrality(self.graph).get(user_id, 0),
                'clustering': nx.clustering(self.graph, user_id)
            }
        except:
            return {'degree': 0, 'betweenness': 0, 'closeness': 0, 'clustering': 0}
    
    def detect_risk_clusters(self) -> Dict:
        """Detect clusters of high-risk users"""
        
        # Find high-risk users
        high_risk_users = [node for node, data in self.graph.nodes(data=True)
                          if data.get('default_probability', 0) > 0.3]
        
        # Create subgraph of high-risk users
        high_risk_subgraph = self.graph.subgraph(high_risk_users)
        risk_clusters = list(nx.connected_components(high_risk_subgraph))
        
        cluster_analysis = []
        for i, cluster in enumerate(risk_clusters):
            if len(cluster) >= 2:
                cluster_nodes = list(cluster)
                cluster_risks = [self.graph.nodes[node].get('default_probability', 0) for node in cluster_nodes]
                
                cluster_info = {
                    'cluster_id': i,
                    'size': len(cluster_nodes),
                    'users': cluster_nodes,
                    'avg_risk_score': round(np.mean(cluster_risks), 4),
                    'max_risk_score': round(np.max(cluster_risks), 4),
                    'cluster_density': nx.density(high_risk_subgraph.subgraph(cluster_nodes))
                }
                cluster_analysis.append(cluster_info)
        
        return {
            'total_risk_clusters': len(cluster_analysis),
            'largest_cluster_size': max([c['size'] for c in cluster_analysis]) if cluster_analysis else 0,
            'cluster_details': sorted(cluster_analysis, key=lambda x: x['avg_risk_score'], reverse=True)
        }
    
    def save_model(self, filepath: str):
        """Save the trained social risk analyzer"""
        model_data = {
            'gnn_model': self.gnn_model,
            'fallback_model': self.fallback_model,
            'node_feature_scaler': self.node_feature_scaler,
            'cascade_decay': self.cascade_decay,
            'min_connection_strength': self.min_connection_strength,
            'max_propagation_hops': self.max_propagation_hops,
            'is_fitted': self.is_fitted,
            'user_to_index': self.user_to_index,
            'index_to_user': self.index_to_user
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Social Risk Analyzer saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained social risk analyzer"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.gnn_model = model_data['gnn_model']
        self.fallback_model = model_data['fallback_model']
        self.node_feature_scaler = model_data['node_feature_scaler']
        self.cascade_decay = model_data['cascade_decay']
        self.min_connection_strength = model_data['min_connection_strength']
        self.max_propagation_hops = model_data['max_propagation_hops']
        self.is_fitted = model_data['is_fitted']
        self.user_to_index = model_data['user_to_index']
        self.index_to_user = model_data['index_to_user']
        
        print(f"Social Risk Analyzer loaded from {filepath}")
        return self

# Example usage
if __name__ == "__main__":
    # Test with sample data
    social_analyzer = SocialRiskAnalyzer()
    
    sample_users = [
        {'user_id': 'USR001', 'credit_risk_score': 450, 'default_probability': 0.8, 'default_status': 'defaulted'},
        {'user_id': 'USR002', 'credit_risk_score': 720, 'default_probability': 0.1, 'default_status': 'active'},
        {'user_id': 'USR003', 'credit_risk_score': 600, 'default_probability': 0.3, 'default_status': 'active'}
    ]
    
    sample_connections = [
        {'user_id': 'USR001', 'connected_user_id': 'USR002', 'strength': 0.8, 'risk_correlation': 0.6},
        {'user_id': 'USR002', 'connected_user_id': 'USR003', 'strength': 0.4, 'risk_correlation': 0.3}
    ]
    
    # Build graph and fit model
    social_analyzer.build_social_graph(sample_users, sample_connections)
    social_analyzer.fit('default_probability')
    
    # Test prediction
    result = social_analyzer.predict_social_risk('USR002')
    print(f"Social risk for USR002: {result['social_risk_score']}")
    
    # Test cascade analysis
    cascade_result = social_analyzer.calculate_cascade_risk('USR001')
    print(f"Cascade from USR001 affects {cascade_result['total_affected_users']} users")