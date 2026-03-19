import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# THREAT MODEL SIMULATOR (FINAL ROBUST VERSION)
# =============================================================================

class PrivacyAttackSimulator:
    def __init__(self, original_gdf: gpd.GeoDataFrame, anonymized_gdf: gpd.GeoDataFrame, 
                 method_name: str):
        self.original = original_gdf
        self.anonymized = anonymized_gdf
        self.method = method_name
        
    def homogeneity_attack(self, sensitive_attr: str = 'flaechentyp') -> Dict:
        """Simulates homogeneity attack. Includes check for singleton artifacts."""
        # Identify grouping column
        group_col = None
        if 'cluster_id' in self.anonymized.columns: group_col = 'cluster_id'
        elif 'h3_index' in self.anonymized.columns: group_col = 'h3_index'
        elif 'geohash' in self.anonymized.columns: group_col = 'geohash'
            
        if group_col is None:
            return {'applicable': False, 'success_rate': 0.0}
        
        # DP Grid check (Aggregated data is exempt)
        if 'dominant_type' in self.anonymized.columns:
             return {'applicable': False, 'note': 'Aggregated data'}

        try:
            # SANITY CHECK: If groups are just single rows, it's not a cluster method
            # This catches 1:1 methods that accidentally preserved a 'cluster_id' column
            if self.anonymized.groupby(group_col).size().mean() < 1.1:
                return {'applicable': False, 'note': 'Singleton clusters detected (Artifact)'}

            joined = gpd.sjoin(self.anonymized, self.original, how='left', predicate='intersects')
            
            target_col = sensitive_attr
            if sensitive_attr not in joined.columns:
                if f"{sensitive_attr}_right" in joined.columns:
                    target_col = f"{sensitive_attr}_right"
                elif f"{sensitive_attr}_left" in joined.columns:
                    target_col = f"{sensitive_attr}_left"
                else:
                    return {'applicable': False}
                
            groups = joined.groupby(group_col)
            vulnerable_groups = 0
            total_groups = 0
            
            for _, group in groups:
                total_groups += 1
                if group[target_col].nunique() == 1:
                    vulnerable_groups += 1
            
            success_rate = vulnerable_groups / total_groups if total_groups > 0 else 0
            return {'applicable': True, 'success_rate': success_rate}
            
        except Exception:
            return {'applicable': False}

    def background_knowledge_attack(self, knowledge: Dict) -> Dict:
        """Simulates finding a unique parcel using approximate location + area."""
        matches = 0
        total_attempts = 0
        targets = self.original.sample(min(100, len(self.original)), random_state=42)
        
        for _, target in targets.iterrows():
            total_attempts += 1
            target_area = target.geometry.area
            target_centroid = target.geometry.centroid
            
            # 1. Spatial Filter
            candidates = self.anonymized[
                self.anonymized.geometry.distance(target_centroid) < 2000
            ]
            if len(candidates) == 0: continue
                
            # 2. Area Filter
            if 'flaecheAmtlich' in candidates.columns:
                # Safely convert column to numbers. If it hits 'MASKED', it becomes NaN.
                # Then fill those NaNs with the actual geometric area of the polygon.
                candidate_areas = pd.to_numeric(candidates['flaecheAmtlich'], errors='coerce').fillna(candidates.geometry.area)
            else:
                # If the column was removed entirely, just use the geometric area
                candidate_areas = candidates.geometry.area
                
            # Filter candidates using our safely calculated areas
            candidates = candidates[
                (candidate_areas >= target_area * 0.9) &
                (candidate_areas <= target_area * 1.1)
            ]
            
            # 3. Uniqueness Check
            if len(candidates) == 1:
                matches += 1
                
        success_rate = matches / total_attempts if total_attempts > 0 else 0
        return {'success_rate': success_rate}

    def satellite_correlation_attack(self) -> Dict:
        """
        Simulates satellite shape matching.
        FIXED: Forces index reset to guarantee alignment for 1:1 methods.
        """
        # If lengths differ significantly, it's aggregation (Attack fails)
        if abs(len(self.anonymized) - len(self.original)) > 10:
            return {'success_rate': 0.0}
            
        successful_matches = 0
        sample_size = min(50, len(self.original))
        
        # FORCE ALIGNMENT: Reset indices to 0..N to ensure row i matches row i
        # This fixes the "0% success" bug caused by index mismatch
        orig_reset = self.original.reset_index(drop=True)
        anon_reset = self.anonymized.reset_index(drop=True)
        
        for i in range(sample_size):
            orig_poly = orig_reset.geometry.iloc[i]
            
            try:
                # 1. Broad Search (5km)
                candidates = anon_reset[
                    anon_reset.geometry.distance(orig_poly.centroid) < 5000
                ]
                if len(candidates) == 0: continue

                # 2. Find Best Shape Match (Hausdorff)
                best_score = float('inf')
                best_match_idx = -1
                
                for idx_label, anon_row in candidates.iterrows():
                    try:
                        h_dist = orig_poly.hausdorff_distance(anon_row.geometry)
                    except Exception:
                        h_dist = float('inf')

                    if h_dist < best_score:
                        best_score = h_dist
                        best_match_idx = idx_label
                
                # 3. Verdict Logic
                # A: Exact Index Match (Since we reset indices, this is Position Match)
                if best_match_idx == i:
                    successful_matches += 1
                
                # B: Shape Similarity Fallback
                # Relaxed to 25m because geo_indist can add ~15m noise
                elif best_score < 25.0:
                    successful_matches += 1
                    
            except Exception:
                continue
                
        success_rate = successful_matches / sample_size
        return {'success_rate': success_rate}

    def run_all_attacks(self) -> Dict:
        return {
            'homogeneity': self.homogeneity_attack(),
            'background': self.background_knowledge_attack({}),
            'satellite': self.satellite_correlation_attack()
        }

def evaluate_anonymization_robustness(original_gdf, anonymized_results_dict):
    print("\n" + "="*80)
    print("🛡️  THREAT MODEL: COMPREHENSIVE PARAMETER SWEEP  🛡️")
    print("="*80)
    
    evaluation_log = []
    
    for method_name in sorted(anonymized_results_dict.keys()):
        anon_gdf = anonymized_results_dict[method_name]
        if len(anon_gdf) == 0: continue
            
        simulator = PrivacyAttackSimulator(original_gdf, anon_gdf, method_name)
        results = simulator.run_all_attacks()
        
        vuln_score = (
            results['homogeneity'].get('success_rate', 0) * 0.3 +
            results['background'].get('success_rate', 0) * 0.4 +
            results['satellite'].get('success_rate', 0) * 0.3
        )
        
        evaluation_log.append({
            'Method_Config': method_name,
            'Homogeneity': results['homogeneity'].get('success_rate', 0),
            'Background': results['background'].get('success_rate', 0),
            'Satellite': results['satellite'].get('success_rate', 0),
            'Overall_Vuln': vuln_score
        })
        print(f"Testing: {method_name} [Done] -> Vuln: {vuln_score:.1%}")
        
    df = pd.DataFrame(evaluation_log)
    out_dir = Path("thesis_outputs")
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "full_parameter_threat_analysis.csv", index=False)
    
    print("\n" + "-"*80)
    print(f"FULL RESULTS ({len(df)} Configurations)")
    print("-"*80)
    print(df.to_string(index=False, float_format=lambda x: "{:.1%}".format(x)))
    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_comprehensive_evaluation():
    try:
        from cadastral_anonymizer import OptimizedCadastralAnonymizer, load_geojson_text, clean_geojson_data
        from hybrid_anon import HybridAnonymizer
        import json
    except ImportError as e:
        print(f"Error importing anonymizer classes: {e}")
        return

    # 1. Load Data
    geojson_path = "minifix.geojson.json"
    print(f"📂 Loading {geojson_path}...")
    try:
        geojson_str = load_geojson_text(geojson_path)
        cleaned = clean_geojson_data(geojson_str)
        data = json.loads(cleaned)
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        if gdf.crs is None: gdf = gdf.set_crs('EPSG:25832')
        
        # --- DATA SANITIZATION ---
        # Remove leftover artifact columns that confuse the threat model
        cols_to_drop = ['cluster_id', 'h3_index', 'geohash', 'agg_geom']
        gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns], errors='ignore')
        print(f"Initialized with {len(gdf)} features (Sanitized)")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    trad_anon = OptimizedCadastralAnonymizer(gdf)
    hybrid_anon = HybridAnonymizer(gdf)
    results = {}

    print(f"\n🏗️  GENERATING DATASETS FOR PARAMETER SWEEP...")

    # --- 1. K-Anonymity Variations ---
    for k in [3, 5, 10]:
        results[f"k_anon_k{k}"] = trad_anon.optimized_k_anonymity(k=k)

    # --- 2. DP Grid Variations ---
    for eps in [0.5, 1.0, 5.0]:
        results[f"dp_grid_eps{eps}"] = trad_anon.dp_grid_aggregation(epsilon=eps, grid_resolution=8)

    # --- 3. Geohash Variations ---
    for prec in [5, 6, 7]:
        results[f"geohash_prec{prec}"] = hybrid_anon.geohashing_anonymization(precision=prec)

    # --- 4. H3 Variations ---
    for res in [7, 8, 9]:
        results[f"h3_res{res}"] = hybrid_anon.h3_hexagonal_anonymization(resolution=res)

    # --- 5. Geo-Indistinguishability ---
    for eps in [1.0, 2.0, 5.0]:
        results[f"geo_indist_eps{eps}"] = trad_anon.conservative_geo_indistinguishability(epsilon=eps)

    # --- 6. Precise Generalization ---
    for tol in [1.0, 2.0, 5.0]:
        results[f"precise_gen_tol{tol}m"] = trad_anon.precise_generalization(tolerance=tol)

    # --- 7. Donut Geomasking ---
    for k in [5, 50]: 
        results[f"donut_mask_k{k}"] = trad_anon.donut_geomasking(k=k)

    # --- 8. Hybrid Methods ---
    for eps in [1.0, 2.0]:
        results[f"hybrid_donut_cons_k5_eps{eps}"] = hybrid_anon.hybrid_donut_conservative_geo(k=5, epsilon=eps)

    for eps in [1.0, 5.0]:
        results[f"donut_vertex_eps{eps}"] = hybrid_anon.donut_geo_indist_only(epsilon=eps, geometry_noise_scale=10)

    for prec in [5, 6]:
        results[f"hybrid_geohash_dp_prec{prec}"] = hybrid_anon.hybrid_geohash_noise(precision=prec, epsilon=1.0)

    results["hybrid_h3_k_res8_k3"] = hybrid_anon.hybrid_h3_clustering(resolution=8, k=3)

    # Run Analysis
    evaluate_anonymization_robustness(gdf, results)

if __name__ == "__main__":
    run_comprehensive_evaluation()
    
class EnhancedPrivacyDefenses:
    def __init__(self, gdf): pass
def create_privacy_preserving_pipeline(*args, **kwargs): return None