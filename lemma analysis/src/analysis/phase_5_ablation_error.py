"""
Phase 5: Ablation & Error Analysis
Refactored from Phase_5.ipynb
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class Phase5Analyzer:
    """Comprehensive analyzer for ablation studies and error analysis."""
    
    def __init__(self, data_path='new_data_clean.csv'):
        """Initialize Phase 5 analyzer with clean data and feature files."""
        self.data_path = data_path
        self.feature_files = ['x_ft_1.pkl', 'x_byt5_1.pkl', 'x_bert_1.pkl']
        self.feature_names = ['FastText', 'ByT5', 'BERT']
        
        # Define feature blocks
        self.feature_blocks = {
            'latin_ngrams': ['lat_ngram_1', 'lat_ngram_2', 'lat_ngram_3', 'lat_ngram_4'],
            'occitan_ngrams': ['occ_ngram_1', 'occ_ngram_2', 'occ_ngram_3', 'occ_ngram_4'],
            'vc_patterns': ['vc_lat', 'vc_occ'],
            'stress_patterns': ['stress_lat', 'stress_occ'],
            'syllable_counts': ['syl_lat', 'syl_occ'],
            'meta_features': ['word_len', 'frequency']
        }
        
        self.load_data()
    
    def load_data(self):
        """Load clean data and feature matrices."""
        print("Loading data and features...")
        
        # Load clean data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples from {self.data_path}")
        
        # Load feature matrices
        self.features = {}
        for i, file_path in enumerate(self.feature_files):
            try:
                with open(file_path, 'rb') as f:
                    self.features[self.feature_names[i]] = pickle.load(f)
                print(f"Loaded {self.feature_names[i]} features: {self.features[self.feature_names[i]].shape}")
            except FileNotFoundError:
                print(f"Warning: {file_path} not found, skipping {self.feature_names[i]}")
        
        # Extract target variable
        if 'Genus_ok' in self.df.columns:
            self.y = self.df['Genus_ok'].values
        elif 'genus_occ' in self.df.columns:
            self.y = self.df['genus_occ'].values
        else:
            genus_cols = [col for col in self.df.columns if 'genus' in col.lower()]
            if genus_cols:
                self.y = self.df[genus_cols[0]].values
                print(f"Using {genus_cols[0]} as target variable")
        
        print(f"Target distribution: {Counter(self.y)}")
        
        # Filter neuter subset
        if 'Genus_lat' in self.df.columns:
            self.neuter_mask = self.df['Genus_lat'] == 'n'
        elif 'genus_lat' in self.df.columns:
            self.neuter_mask = self.df['genus_lat'] == 'n'
        else:
            lat_cols = [col for col in self.df.columns if 'lat' in col.lower() and 'genus' in col.lower()]
            if lat_cols:
                self.neuter_mask = self.df[lat_cols[0]] == 'n'
        
        print(f"Found {self.neuter_mask.sum()} neuter nouns out of {len(self.df)} total")
    
    def ablation_study(self, model_type='rf', cv_folds=5):
        """Perform ablation study by removing each feature block."""
        print(f"\n=== ABLATION STUDY ({model_type.upper()}) ===")
        
        results = defaultdict(dict)
        
        for embedding_type in self.feature_names:
            if embedding_type not in self.features:
                continue
                
            print(f"\n--- Analyzing {embedding_type} features ---")
            X_full = self.features[embedding_type]
            
            # Get baseline performance
            baseline_scores = self._cross_validate_model(X_full, self.y, model_type, cv_folds)
            baseline_f1 = baseline_scores['f1_macro']
            baseline_acc = baseline_scores['accuracy']
            
            print(f"Baseline - F1: {baseline_f1:.4f} (±{baseline_scores['f1_std']:.4f}), "
                  f"Acc: {baseline_acc:.4f} (±{baseline_scores['acc_std']:.4f})")
            
            results[embedding_type]['baseline'] = baseline_scores
            
            # Test each feature block removal
            for block_name, feature_cols in self.feature_blocks.items():
                available_features = [col for col in feature_cols if col in X_full.columns]
                if not available_features:
                    continue
                
                # Remove feature block
                X_ablated = X_full.drop(columns=available_features)
                
                # Evaluate without this block
                ablated_scores = self._cross_validate_model(X_ablated, self.y, model_type, cv_folds)
                
                # Calculate performance drop
                delta_f1 = baseline_f1 - ablated_scores['f1_macro']
                delta_acc = baseline_acc - ablated_scores['accuracy']
                
                results[embedding_type][block_name] = {
                    'scores': ablated_scores,
                    'delta_f1': delta_f1,
                    'delta_acc': delta_acc,
                    'removed_features': available_features
                }
                
                print(f"  -{block_name:15} | ΔF1: {delta_f1:+.4f} | ΔAcc: {delta_acc:+.4f} | "
                      f"F1: {ablated_scores['f1_macro']:.4f}")
        
        self.ablation_results = results
        self._save_ablation_results(results)
        return results
    
    def _cross_validate_model(self, X, y, model_type='rf', cv_folds=5):
        """Cross-validate a model and return performance metrics."""
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'f1_macro': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'accuracy': acc_scores.mean(),
            'acc_std': acc_scores.std()
        }
    
    def cluster_neuter_analysis(self, embedding_type=None, n_clusters_range=(2, 6)):
        """Cluster neuter nouns in embedding space and analyze gender distribution."""
        print(f"\n=== CLUSTERING ANALYSIS - {embedding_type} ===")
        
        if embedding_type not in self.features:
            print(f"Warning: {embedding_type} features not available")
            return {}
        
        X = self.features[embedding_type]
        X_neuter = X[self.neuter_mask]
        y_neuter = self.y[self.neuter_mask]
        
        print(f"Analyzing {len(X_neuter)} neuter nouns")
        print(f"Neuter→Occitan gender distribution: {Counter(y_neuter)}")
        
        cluster_results = {}
        
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            print(f"\n--- {n_clusters} Clusters ---")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_neuter)
            
            cluster_analysis = defaultdict(Counter)
            for i, cluster_id in enumerate(cluster_labels):
                cluster_analysis[cluster_id][y_neuter[i]] += 1
            
            cluster_results[n_clusters] = {
                'model': kmeans,
                'labels': cluster_labels,
                'analysis': dict(cluster_analysis)
            }
            
            # Print cluster composition
            for cluster_id in range(n_clusters):
                counts = cluster_analysis[cluster_id]
                total = sum(counts.values())
                print(f"  Cluster {cluster_id}: {total} nouns - ", end="")
                for gender, count in counts.items():
                    print(f"{gender}: {count} ({count/total:.2%})", end=" ")
                print()
        
        # Visualize clusters
        best_n_clusters = max(cluster_results.keys())
        best_clusters = cluster_results[best_n_clusters]['labels']
        self.visualize_clusters(X_neuter, y_neuter, best_clusters, embedding_type)
        
        return cluster_results
    
    def visualize_clusters(self, X_neuter, y_neuter_raw, clusters, embedding_type=None):
        """Create t-SNE visualization of clusters."""
        print("\nCreating t-SNE visualization...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_neuter)-1))
        X_tsne = tsne.fit_transform(X_neuter)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot by cluster
        scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        ax1.set_title('Former Neuter Nouns Clustered in Feature Space')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Plot by resulting Occitan gender
        occitan_gender_colors = {'m': 'red', 'f': 'blue', 'n': 'green'}
        for gender in np.unique(y_neuter_raw):
            mask = y_neuter_raw == gender
            if np.any(mask):
                ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          c=occitan_gender_colors.get(gender, 'gray'),
                          label=f'Occitan: {gender}', alpha=0.7)
        
        ax2.set_title('Former Neuter Nouns by Occitan Gender')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.legend()
        
        plt.tight_layout()
        if embedding_type:
            filename = f'occitan_cluster_analysis_{embedding_type.lower()}.png'
        else:
            filename = 'occitan_cluster_analysis.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {filename}")
    
    def comprehensive_cluster_analysis(self, n_clusters_range=(2, 6)):
        """Run clustering analysis on all embedding types."""
        print(f"\n=== COMPREHENSIVE CLUSTERING ANALYSIS ===")
        
        all_results = {}
        cluster_csv_data = []
        
        for embedding_type in self.feature_names:
            if embedding_type not in self.features:
                continue
                
            print(f"\n{'='*20} {embedding_type} EMBEDDINGS {'='*20}")
            
            cluster_results = self.cluster_neuter_analysis(
                embedding_type=embedding_type,
                n_clusters_range=n_clusters_range
            )
            
            all_results[embedding_type] = cluster_results
            
            # Collect data for CSV
            X = self.features[embedding_type]
            X_neuter = X[self.neuter_mask]
            y_neuter = self.y[self.neuter_mask]
            
            for n_clusters, result in cluster_results.items():
                cluster_analysis = result['analysis']
                
                for cluster_id in range(n_clusters):
                    counts = cluster_analysis.get(cluster_id, {})
                    total = sum(counts.values())
                    max_count = max(counts.values()) if counts else 0
                    purity = max_count / total if total > 0 else 0
                    
                    for gender, count in counts.items():
                        cluster_csv_data.append({
                            'embedding_type': embedding_type,
                            'n_clusters': n_clusters,
                            'cluster_id': cluster_id,
                            'gender': gender,
                            'count': count,
                            'cluster_size': total,
                            'cluster_purity': purity,
                            'percentage': count / total if total > 0 else 0
                        })
        
        # Save cluster analysis to CSV
        cluster_df = pd.DataFrame(cluster_csv_data)
        cluster_df.to_csv('occitan_cluster_analysis.csv', index=False)
        print(f"\nSaved cluster analysis to occitan_cluster_analysis.csv")
        
        return all_results
    
    def error_analysis(self, embedding_type='BERT'):
        """Perform detailed error analysis on model predictions."""
        print(f"\n=== ERROR ANALYSIS - {embedding_type} ===")
        
        if embedding_type not in self.features:
            print(f"Warning: {embedding_type} features not available")
            return [], None
        
        # Train model
        X = self.features[embedding_type]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, self.y)
        predictions = model.predict(X)
        
        # Find misclassified samples
        errors = predictions != self.y
        error_indices = np.where(errors)[0]
        
        print(f"Total errors: {errors.sum()} / {len(self.y)} ({errors.mean():.2%})")
        
        # Analyze error patterns
        lat_col = 'Lemma_std' if 'Lemma_std' in self.df.columns else 'Lemma_lat'
        occ_col = 'Akk_Sing_std' if 'Akk_Sing_std' in self.df.columns else 'Lemma_ok'
        lat_genus_col = 'Genus_lat' if 'Genus_lat' in self.df.columns else 'genus_lat'
        
        error_data = []
        for idx in error_indices:
            error_info = {
                'index': idx,
                'lat_lemma': self.df.iloc[idx][lat_col],
                'occ_lemma': self.df.iloc[idx][occ_col],
                'true_gender': self.y[idx],
                'pred_gender': predictions[idx],
                'lat_genus': self.df.iloc[idx][lat_genus_col],
                'embedding_type': embedding_type
            }
            error_data.append(error_info)
        
        # Save error analysis to CSV
        error_df = pd.DataFrame(error_data)
        error_csv_filename = f'occitan_error_analysis_{embedding_type.lower()}.csv'
        error_df.to_csv(error_csv_filename, index=False)
        print(f"Saved detailed error analysis to {error_csv_filename}")
        
        # Print error examples
        print(f"\nFirst 20 misclassified examples:")
        print(f"{'Index':<6} {'Latin':<15} {'Occitan':<15} {'Lat.Gen':<7} {'True':<4} {'Pred':<4}")
        print("-" * 65)
        
        for error in error_data[:20]:
            print(f"{error['index']:<6} {error['lat_lemma']:<15} {error['occ_lemma']:<15} "
                  f"{error['lat_genus']:<7} {error['true_gender']:<4} {error['pred_gender']:<4}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 most important features:")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return error_data, feature_importance
    
    def _save_ablation_results(self, results):
        """Save ablation study results to CSV."""
        rows = []
        for embedding in results:
            baseline = results[embedding]['baseline']
            for block_name in results[embedding]:
                if block_name == 'baseline':
                    continue
                block_data = results[embedding][block_name]
                rows.append({
                    'embedding': embedding,
                    'removed_block': block_name,
                    'baseline_f1': baseline['f1_macro'],
                    'ablated_f1': block_data['scores']['f1_macro'],
                    'delta_f1': block_data['delta_f1'],
                    'baseline_acc': baseline['accuracy'],
                    'ablated_acc': block_data['scores']['accuracy'],
                    'delta_acc': block_data['delta_acc'],
                    'removed_features': ', '.join(block_data['removed_features'])
                })
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv('ablation_results.csv', index=False)
        print(f"Saved ablation results to ablation_results.csv")


def run_phase5_analysis():
    """Run complete Phase 5 analysis."""
    print("PHASE 5: ABLATION & ERROR ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = Phase5Analyzer()
    
    # 1. Ablation Study
    print("\n1. ABLATION STUDY")
    ablation_results = analyzer.ablation_study(model_type='rf')
    
    # 2. Comprehensive Clustering Analysis
    print("\n2. COMPREHENSIVE CLUSTERING ANALYSIS")
    comprehensive_results = analyzer.comprehensive_cluster_analysis(n_clusters_range=(2, 6))
    
    # 3. Error Analysis
    print("\n3. ERROR ANALYSIS")
    for embedding_type in analyzer.feature_names:
        if embedding_type in analyzer.features:
            print(f"\n--- Error Analysis for {embedding_type} ---")
            analyzer.error_analysis(embedding_type=embedding_type)
    
    print("\n" + "=" * 50)
    print("PHASE 5 ANALYSIS COMPLETE")
    print("=" * 50)
    
    return analyzer


if __name__ == "__main__":
    analyzer = run_phase5_analysis()

