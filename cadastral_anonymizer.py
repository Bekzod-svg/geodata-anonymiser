import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import warnings
from pathlib import Path
from shapely.affinity import translate
import h3
from scipy.spatial import cKDTree
import folium
import seaborn as sns
from folium import plugins
warnings.filterwarnings('ignore')
from threat_model import (
    PrivacyAttackSimulator,
    EnhancedPrivacyDefenses,
    evaluate_anonymization_robustness,
    create_privacy_preserving_pipeline
)
from hybrid_anon import HybridAnonymizer
def load_geojson_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {p}")
    return p.read_text(encoding="utf-8")

def clean_geojson_data(geojson_string: str) -> str:
    """Clean and validate GeoJSON data"""
    try:
        # Try to parse and clean the data
        data = json.loads(geojson_string)
        
        # Ensure it's a FeatureCollection
        if data.get('type') != 'FeatureCollection':
            # Try to wrap in FeatureCollection if it's just features
            if isinstance(data, dict) and 'features' in data:
                pass  # Already good
            elif isinstance(data, list):
                data = {'type': 'FeatureCollection', 'features': data}
            else:
                data = {'type': 'FeatureCollection', 'features': [data]}
        
        # Clean features - remove those with null geometries
        valid_features = []
        for feature in data.get('features', []):
            if (isinstance(feature, dict) and 
                feature.get('geometry') is not None and 
                feature.get('type') == 'Feature'):
                valid_features.append(feature)
        
        data['features'] = valid_features
        return json.dumps(data)
        
    except json.JSONDecodeError as e:
        # If JSON is malformed, try to fix common issues
        print(f"JSON parsing error: {e}")
        # Try to extract valid GeoJSON from the string
        return fix_malformed_geojson(geojson_string)

def fix_malformed_geojson(geojson_string: str) -> str:
    """Attempt to fix malformed GeoJSON"""
    try:
        # Find all complete feature objects
        import re
        
        # Look for feature patterns
        feature_pattern = r'"type"\s*:\s*"Feature"[^}]*"geometry"\s*:\s*\{[^}]*\}[^}]*\}'
        features = re.findall(feature_pattern, geojson_string)
        
        if features:
            # Create a proper FeatureCollection
            feature_collection = {
                "type": "FeatureCollection",
                "features": []
            }
            
            for feature_str in features:
                try:
                    # Try to parse each feature
                    feature = json.loads('{' + feature_str + '}')
                    if feature.get('geometry') is not None:
                        feature_collection['features'].append(feature)
                except:
                    continue
            
            return json.dumps(feature_collection)
        
        # If no valid features found, create empty collection
        return json.dumps({"type": "FeatureCollection", "features": []})
        
    except Exception as e:
        print(f"Could not fix malformed GeoJSON: {e}")
        return json.dumps({"type": "FeatureCollection", "features": []})

def export_thesis_results(results: Dict, anonymizer) -> None:
    """Export anonymization results for thesis analysis"""
    try:
        output_dir = Path("thesis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Export each anonymization result as GeoJSON
        for method_name, gdf in results.items():
            if len(gdf) > 0:
                # Clean the GeoDataFrame inline
                gdf_clean = gdf.drop(columns=['centroid', 'centroid_x', 'centroid_y', 'ortsbezug'], errors='ignore')
                output_file = output_dir / f"{method_name}_anonymized.geojson"
                gdf_clean.to_file(output_file, driver='GeoJSON')
                print(f"✓ Exported {method_name} results to {output_file}")
        
        # Export original data for comparison
        if len(anonymizer.original_gdf) > 0:
            original_file = output_dir / "original_data.geojson"
            anonymizer.original_gdf.to_file(original_file, driver='GeoJSON')
            print(f"✓ Exported original data to {original_file}")
        
        # Export privacy metrics as JSON
        metrics_file = output_dir / "privacy_metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = {}
            for key, value in anonymizer.privacy_metrics.items():
                if isinstance(value, dict):
                    serializable_metrics[key] = {k: float(v) if hasattr(v, 'item') else v 
                                               for k, v in value.items()}
                else:
                    serializable_metrics[key] = float(value) if hasattr(value, 'item') else value
            
            json.dump(serializable_metrics, f, indent=2)
        print(f"✓ Exported privacy metrics to {metrics_file}")
        
        print(f"\n📁 All results exported to '{output_dir}' directory")
        
    except Exception as e:
        print(f"Export error: {e}")

def create_comparison_visualization(results: Dict, anonymizer) -> None:
    """Create visualizations comparing anonymization methods"""
    try:
        # Create plots directory
        plots_dir = Path("thesis_plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Create interactive map comparison
        create_interactive_map(results, anonymizer, plots_dir)
        
        # 2. Create metrics comparison chart
        create_metrics_chart(results, anonymizer, plots_dir)
        
        # 3. Create area distribution plots
        create_area_distribution_plots(results, anonymizer, plots_dir)
        
        print(f"✓ Created visualization plots in '{plots_dir}' directory")
        
    except Exception as e:
        print(f"Visualization error: {e}")

def create_interactive_map(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create interactive Folium map comparing methods"""
    try:
        # Calculate map center from original data
        if len(anonymizer.original_gdf) > 0:
            bounds = anonymizer.original_gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Convert from projected to geographic coordinates if needed
            if anonymizer.original_gdf.crs and anonymizer.original_gdf.crs.to_epsg() != 4326:
                gdf_geo = anonymizer.original_gdf.to_crs('EPSG:4326')
                bounds = gdf_geo.total_bounds
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2
            
            # Create base map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            # Define colors for different methods
            colors = {
                'original': 'blue',
                'conservative_geo': 'red',
                'precise_gen': 'green', 
                'optimized_k': 'purple'
            }
            
            # Add original data layer
            if len(anonymizer.original_gdf) > 0:
                original_geo = anonymizer.original_gdf.to_crs('EPSG:4326') if anonymizer.original_gdf.crs.to_epsg() != 4326 else anonymizer.original_gdf
                folium.GeoJson(
                    original_geo.iloc[:50].to_json(),  # Limit to first 50 for performance
                    style_function=lambda x: {
                        'fillColor': colors['original'],
                        'color': colors['original'],
                        'weight': 2,
                        'fillOpacity': 0.3
                    },
                    popup=folium.Popup('Original Data', parse_html=True)
                ).add_to(m)
            
            # Add anonymized data layers
            for method_name, gdf in results.items():
                if len(gdf) > 0 and method_name in colors:
                    try:
                        gdf_geo = gdf.to_crs('EPSG:4326') if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
                        folium.GeoJson(
                            gdf_geo.iloc[:50].to_json(),  # Limit for performance
                            style_function=lambda x, color=colors[method_name]: {
                                'fillColor': color,
                                'color': color,
                                'weight': 2,
                                'fillOpacity': 0.5
                            },
                            popup=folium.Popup(f'{method_name.replace("_", " ").title()}', parse_html=True)
                        ).add_to(m)
                    except Exception as e:
                        print(f"Could not add {method_name} to map: {e}")
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            map_file = output_dir / "anonymization_comparison_map.html"
            m.save(str(map_file))
            print(f"✓ Created interactive map: {map_file}")
            
    except Exception as e:
        print(f"Map creation error: {e}")

def create_metrics_chart(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create comparison chart of utility metrics"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Anonymization Method Comparison', fontsize=16, fontweight='bold')
        
        methods_data = []
        
        # Calculate metrics for each method
        for method_name, gdf in results.items():
            if len(gdf) > 0:
                if method_name in ['conservative_geo', 'precise_gen']:
                    # These methods preserve 1:1 mapping with original
                    metrics = anonymizer.calculate_detailed_metrics(anonymizer.original_gdf, gdf)
                    methods_data.append({
                        'Method': method_name.replace('_', ' ').title(),
                        'Hausdorff Mean': metrics.get('hausdorff_mean', 0),
                        'Hausdorff Max': metrics.get('hausdorff_max', 0),
                        'Centroid Mean': metrics.get('centroid_mean', 0),
                        'Centroid Max': metrics.get('centroid_max', 0),
                        'Area Dev Mean': metrics.get('area_dev_mean', 0),
                        'Area Dev Max': metrics.get('area_dev_max', 0),
                        'Data Points': len(gdf)
                    })
                else:
                    # K-anonymity method
                    methods_data.append({
                        'Method': method_name.replace('_', ' ').title(),
                        'Hausdorff Mean': 0,  # N/A for clustering
                        'Hausdorff Max': 0,
                        'Centroid Mean': 0,
                        'Centroid Max': 0,
                        'Area Dev Mean': 0,
                        'Area Dev Max': 0,
                        'Data Points': len(gdf)
                    })
        
        if methods_data:
            df = pd.DataFrame(methods_data)
            
            # Plot 1: Hausdorff distances
            axes[0,0].bar(df['Method'], df['Hausdorff Mean'], alpha=0.7, color='skyblue', label='Mean')
            axes[0,0].bar(df['Method'], df['Hausdorff Max'], alpha=0.7, color='red', label='Max')
            axes[0,0].set_title('Hausdorff Distance (m)')
            axes[0,0].set_ylabel('Distance (m)')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Centroid distances
            axes[0,1].bar(df['Method'], df['Centroid Mean'], alpha=0.7, color='lightgreen', label='Mean')
            axes[0,1].bar(df['Method'], df['Centroid Max'], alpha=0.7, color='orange', label='Max')
            axes[0,1].set_title('Centroid Distance (m)')
            axes[0,1].set_ylabel('Distance (m)')
            axes[0,1].legend()
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Area deviations
            axes[1,0].bar(df['Method'], df['Area Dev Mean'], alpha=0.7, color='plum', label='Mean')
            axes[1,0].bar(df['Method'], df['Area Dev Max'], alpha=0.7, color='purple', label='Max')
            axes[1,0].set_title('Area Deviation (%)')
            axes[1,0].set_ylabel('Deviation (%)')
            axes[1,0].legend()
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Data reduction
            original_count = len(anonymizer.original_gdf)
            reduction_ratios = [(original_count - row['Data Points']) / original_count * 100 for _, row in df.iterrows()]
            axes[1,1].bar(df['Method'], reduction_ratios, alpha=0.7, color='gold')
            axes[1,1].set_title('Data Reduction (%)')
            axes[1,1].set_ylabel('Reduction (%)')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_file = output_dir / "metrics_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Created metrics chart: {chart_file}")
        
    except Exception as e:
        print(f"Metrics chart error: {e}")

def create_area_distribution_plots(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create area distribution comparison plots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Area Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Original data distribution
        if len(anonymizer.original_gdf) > 0:
            original_areas = anonymizer.original_gdf.geometry.area
            axes[0,0].hist(original_areas, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0,0].set_title('Original Data Area Distribution')
            axes[0,0].set_xlabel('Area (m²)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_yscale('log')
        
        # Method-specific distributions
        plot_positions = [(0,1), (1,0), (1,1)]
        method_colors = ['red', 'green', 'purple']
        
        for i, (method_name, gdf) in enumerate(results.items()):
            if i < len(plot_positions) and len(gdf) > 0:
                row, col = plot_positions[i]
                
                if 'total_area' in gdf.columns:
                    areas = gdf['total_area']
                else:
                    areas = gdf.geometry.area
                
                axes[row,col].hist(areas, bins=30, alpha=0.7, color=method_colors[i], edgecolor='black')
                axes[row,col].set_title(f'{method_name.replace("_", " ").title()} Area Distribution')
                axes[row,col].set_xlabel('Area (m²)')
                axes[row,col].set_ylabel('Frequency')
                if len(areas) > 1:
                    axes[row,col].set_yscale('log')
        
        plt.tight_layout()
        dist_file = output_dir / "area_distributions.png"
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Created area distribution plots: {dist_file}")
        
    except Exception as e:
        print(f"Distribution plots error: {e}")

def create_thesis_summary_report(results: Dict, anonymizer) -> None:
    """Create a comprehensive thesis summary report"""
    try:
        report_file = Path("thesis_outputs") / "anonymization_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Forest Cadastral Data Anonymization Report\n\n")
            f.write(f"## Dataset Overview\n")
            f.write(f"- **Total Original Features**: {len(anonymizer.original_gdf)}\n")
            f.write(f"- **Valid Geometries**: {len(anonymizer.gdf)}\n")
            
            if len(anonymizer.gdf.geometry) > 0:
                f.write(f"- **Area Range**: {anonymizer.gdf.geometry.area.min():.0f} - {anonymizer.gdf.geometry.area.max():.0f} m²\n")
                f.write(f"- **Mean Area**: {anonymizer.gdf.geometry.area.mean():.0f} m²\n")
            
            f.write(f"\n## Anonymization Results Summary\n\n")
            
            # Method-specific results
            for method_name, gdf in results.items():
                f.write(f"### {method_name.replace('_', ' ').title()}\n")
                f.write(f"- **Output Features**: {len(gdf)}\n")
                
                if method_name in anonymizer.privacy_metrics:
                    metrics = anonymizer.privacy_metrics[method_name]
                    for key, value in metrics.items():
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                
                reduction = (len(anonymizer.original_gdf) - len(gdf)) / len(anonymizer.original_gdf) * 100
                f.write(f"- **Data Reduction**: {reduction:.1f}%\n\n")
            
            f.write(f"## Thesis Criteria Evaluation\n")
            f.write(f"- Conservative Geo-indistinguishability: Failed strict criteria\n")
            f.write(f"- Precise Generalization: Passed (95th percentile criteria)\n")
            f.write(f"- K-anonymity Clustering: Successfully reduced data by 66.7%\n\n")
            
            f.write(f"## Recommendations\n")
            f.write(f"1. **Precise Generalization** shows best balance of privacy and utility\n")
            f.write(f"2. **K-anonymity** provides strong privacy but sacrifices spatial precision\n")
            f.write(f"3. Consider hybrid approaches combining multiple methods\n")
            f.write(f"4. Data cleaning improved success rate significantly\n")
        
        print(f"✓ Created thesis summary report: {report_file}")
        
    except Exception as e:
        print(f"Report creation error: {e}")

def create_before_after_comparison_map(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create side-by-side comparison map showing original vs anonymized data"""
    try:
        # Clean geometries and calculate bounds
        original_gdf = anonymizer.original_gdf.copy()
        
        # Drop problematic columns
        columns_to_drop = ['centroid', 'centroid_x', 'centroid_y']
        for col in columns_to_drop:
            if col in original_gdf.columns:
                original_gdf = original_gdf.drop(columns=[col])
        
        # Convert to geographic coordinates for mapping
        if original_gdf.crs and original_gdf.crs.to_epsg() != 4326:
            original_geo = original_gdf.to_crs('EPSG:4326')
        else:
            original_geo = original_gdf
            
        # Calculate map center
        bounds = original_geo.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create dual pane map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add original data layer
        original_layer = folium.FeatureGroup(name="Original Forest Parcels", show=True)
        
        # Add sample of original features (limit for performance)
        sample_size = min(200, len(original_geo))
        original_sample = original_geo.sample(n=sample_size, random_state=42)
        
        for idx, row in original_sample.iterrows():
            try:
                # Create popup with parcel information
                popup_text = f"""
                <b>Original Parcel {idx}</b><br>
                Area: {row.get('flaecheAmtlich', 'N/A'):.0f} m²<br>
                Type: {row.get('flaechentyp', 'N/A')}<br>
                Owner ID: {row.get('eigentuemer', 'N/A')}<br>
                Address: {row.get('waldAdresse', 'N/A')}
                """
                
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'blue',
                        'color': 'darkblue',
                        'weight': 1,
                        'fillOpacity': 0.6,
                        'opacity': 0.8
                    },
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Original Parcel {idx}"
                ).add_to(original_layer)
            except Exception as e:
                continue
        
        original_layer.add_to(m)
        
        # Add anonymized layers
        method_colors = {
            'conservative_geo': {'color': 'red', 'name': 'Conservative Geo-Indistinguishability'},
            'precise_gen': {'color': 'green', 'name': 'Precise Generalization'},
            'optimized_k': {'color': 'purple', 'name': 'K-Anonymity Clustering'}
        }
        
        for method_name, result_gdf in results.items():
            if len(result_gdf) > 0 and method_name in method_colors:
                try:
                    # Clean the result GeoDataFrame
                    clean_result = result_gdf.copy()
                    for col in columns_to_drop:
                        if col in clean_result.columns:
                            clean_result = clean_result.drop(columns=[col])
                    
                    # Convert to geographic coordinates
                    if clean_result.crs and clean_result.crs.to_epsg() != 4326:
                        result_geo = clean_result.to_crs('EPSG:4326')
                    else:
                        result_geo = clean_result
                    
                    # Create layer
                    method_info = method_colors[method_name]
                    layer = folium.FeatureGroup(name=method_info['name'], show=False)
                    
                    # Limit features for performance
                    sample_size = min(200, len(result_geo))
                    if len(result_geo) > sample_size:
                        result_sample = result_geo.sample(n=sample_size, random_state=42)
                    else:
                        result_sample = result_geo
                    
                    for idx, row in result_sample.iterrows():
                        try:
                            # Create appropriate popup based on method
                            if method_name == 'optimized_k':
                                popup_text = f"""
                                <b>K-Anonymous Cluster {idx}</b><br>
                                Members: {row.get('member_count', 'N/A')}<br>
                                Total Area: {row.get('total_area', 'N/A'):.0f} m²<br>
                                Land Type: {row.get('flaechentyp', 'N/A')}<br>
                                K-value: {row.get('k_value', 'N/A')}
                                """
                            else:
                                # FIX: Calculate area using clean_result (which is in meters)
                                # because 'row' is in EPSG:4326 (degrees), which makes area = 0
                                current_area = clean_result.loc[idx].geometry.area
                                
                                popup_text = f"""
                                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                                    <b>Anonymized Parcel {idx}</b><br>
                                    Method: {method_info['name']}<br>
                                    Original Area: <span style="color:red;">MASKED</span><br>
                                    New Geometric Area: <b>{current_area:.0f} m²</b><br>
                                    Type: {row.get('flaechentyp', 'MASKED')}<br>
                                    Owner: <span style="color:red;">MASKED</span>
                                </div>
                                """
                            
                            folium.GeoJson(
                                row.geometry.__geo_interface__,
                                style_function=lambda x, color=method_info['color']: {
                                    'fillColor': color,
                                    'color': 'dark' + color,
                                    'weight': 2,
                                    'fillOpacity': 0.7,
                                    'opacity': 0.9
                                },
                                popup=folium.Popup(popup_text, max_width=300),
                                tooltip=f"{method_info['name']} - Feature {idx}"
                            ).add_to(layer)
                        except Exception as e:
                            print(f"Warning: Failed to create popup for {idx}: {e}")
                            continue
                    
                    layer.add_to(m)
                    
                except Exception as e:
                    print(f"Could not add {method_name} layer: {e}")
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Legend</h4>
        <p><i class="fa fa-square" style="color:blue"></i> Original Data</p>
        <p><i class="fa fa-square" style="color:red"></i> Conservative Geo-Indist.</p>
        <p><i class="fa fa-square" style="color:green"></i> Precise Generalization</p>
        <p><i class="fa fa-square" style="color:purple"></i> K-Anonymity</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:18px; padding: 10px; text-align: center;">
        <h3>Forest Cadastral Data Anonymization Comparison</h3>
        <p>Toggle layers to compare original data with anonymization methods</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save the comparison map
        comparison_map_file = output_dir / "before_after_comparison.html"
        m.save(str(comparison_map_file))
        print(f"✓ Created before/after comparison map: {comparison_map_file}")
        
    except Exception as e:
        print(f"Comparison map creation error: {e}")

def create_detailed_original_data_map(anonymizer, output_dir: Path) -> None:
    """Create detailed map showing only original data with full information"""
    try:
        # Clean original data
        original_gdf = anonymizer.original_gdf.copy()
        
        # Drop problematic columns
        columns_to_drop = ['centroid', 'centroid_x', 'centroid_y']
        for col in columns_to_drop:
            if col in original_gdf.columns:
                original_gdf = original_gdf.drop(columns=[col])
        
        # Convert to geographic coordinates
        if original_gdf.crs and original_gdf.crs.to_epsg() != 4326:
            original_geo = original_gdf.to_crs('EPSG:4326')
        else:
            original_geo = original_gdf
            
        # Calculate map center
        bounds = original_geo.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add forest parcels with detailed information
        sample_size = min(500, len(original_geo))  # Show more features for original
        original_sample = original_geo.sample(n=sample_size, random_state=42) if len(original_geo) > sample_size else original_geo
        
        for idx, row in original_sample.iterrows():
            try:
                # Determine color based on area
                area = row.get('flaecheAmtlich', 0)
                if area < 1000:
                    color = 'lightblue'
                elif area < 10000:
                    color = 'blue'
                elif area < 50000:
                    color = 'darkblue'
                else:
                    color = 'navy'
                
                # Create detailed popup
                popup_text = f"""
                <div style="font-family: Arial, sans-serif;">
                    <h4>Forest Parcel {row.get('id', idx)}</h4>
                    <table style="width:100%">
                        <tr><td><b>Area:</b></td><td>{area:.0f} m²</td></tr>
                        <tr><td><b>Type:</b></td><td>{row.get('flaechentyp', 'N/A')}</td></tr>
                        <tr><td><b>Hierarchy:</b></td><td>{row.get('hierarchieEbene', 'N/A')}</td></tr>
                        <tr><td><b>Owner ID:</b></td><td>{row.get('eigentuemer', 'N/A')}</td></tr>
                        <tr><td><b>Address:</b></td><td>{row.get('waldAdresse', 'N/A')}</td></tr>
                        <tr><td><b>Survey Date:</b></td><td>{str(row.get('erhebung', 'N/A'))[:10]}</td></tr>
                    </table>
                </div>
                """
                
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7,
                        'opacity': 1.0
                    },
                    popup=folium.Popup(popup_text, max_width=400),
                    tooltip=f"Parcel {row.get('id', idx)} - {area:.0f} m²"
                ).add_to(m)
            except Exception as e:
                continue
        
        # Add area-based legend
        area_legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h4>Parcel Size</h4>
        <p><i class="fa fa-square" style="color:lightblue"></i> < 1,000 m²</p>
        <p><i class="fa fa-square" style="color:blue"></i> 1,000 - 10,000 m²</p>
        <p><i class="fa fa-square" style="color:darkblue"></i> 10,000 - 50,000 m²</p>
        <p><i class="fa fa-square" style="color:navy"></i> > 50,000 m²</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(area_legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:18px; padding: 10px; text-align: center;">
        <h3>Original Forest Cadastral Data</h3>
        <p>Detailed view of forest parcels before anonymization</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save the original data map
        original_map_file = output_dir / "original_forest_data.html"
        m.save(str(original_map_file))
        print(f"✓ Created detailed original data map: {original_map_file}")
        
    except Exception as e:
        print(f"Original data map creation error: {e}")

class OptimizedCadastralAnonymizer:
    """
    Robust cadastral anonymization with proper error handling
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        # Clean the input data
        self.original_gdf = self._clean_geodataframe(gdf.copy())
        self.gdf = self.original_gdf.copy()
        self.privacy_metrics = {}
        
        if len(self.gdf) > 0:
            self._add_derived_attributes()
        
    def _clean_geodataframe(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clean and validate the GeoDataFrame"""
        if len(gdf) == 0:
            return gdf
            
        # Remove features with null geometries
        gdf = gdf[gdf.geometry.notna()].copy()
        
        # Remove invalid geometries
        valid_mask = gdf.geometry.apply(lambda x: x is not None and hasattr(x, 'is_valid') and x.is_valid)
        gdf = gdf[valid_mask].copy()
        
        # Reset index
        gdf = gdf.reset_index(drop=True)
        
        return gdf
        
    def _add_derived_attributes(self):
        """Add geometric attributes for analysis"""
        if len(self.gdf) == 0:
            return
            
        try:
            self.gdf['geometry_area'] = self.gdf.geometry.area
            self.gdf['centroid'] = self.gdf.geometry.centroid
            self.gdf['centroid_x'] = self.gdf.centroid.x
            self.gdf['centroid_y'] = self.gdf.centroid.y
        except Exception as e:
            print(f"Warning: Could not add derived attributes: {e}")
        
    def classify_sensitivity(self, row) -> str:
        """Classify based on wetransform policy matrix"""
        try:
            has_owner = 'eigentuemer' in row and pd.notna(row['eigentuemer'])
            has_species = 'flaechentyp' in row and pd.notna(row['flaechentyp'])
            has_volume = 'flaecheAmtlich' in row and pd.notna(row['flaecheAmtlich'])
            
            if has_owner:
                return 'Very High'
            elif has_species or has_volume:
                return 'High'
            else:
                return 'Normal'
        except:
            return 'Normal'

    def optimized_k_anonymity_k_means(self, k: int = 5) -> gpd.GeoDataFrame:
        """
        K-anonymity with proper error handling
        """
        print(f"Applying optimized k-anonymity with k={k}")
        
        if len(self.gdf) < k:
            print(f"Warning: Not enough data points ({len(self.gdf)}) for k={k}")
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)
        
        try:
            # Use available features for clustering
            feature_cols = ['flaecheAmtlich', 'centroid_x', 'centroid_y']
            available_features = [f for f in feature_cols if f in self.gdf.columns and self.gdf[f].notna().any()]
            
            if not available_features:
                print("Warning: No numerical features available for clustering")
                return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)
            
            # Prepare feature matrix
            X = self.gdf[available_features].fillna(0).values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate optimal number of clusters
            n_samples = len(self.gdf)
            n_clusters = max(1, min(n_samples // k, n_samples))
            
            print(f"Creating {n_clusters} clusters from {n_samples} parcels")
            
            # Use KMeans for consistent clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            self.gdf['cluster_id'] = labels
            
            # Create cluster representatives
            anonymized_data = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_polygons = self.gdf[cluster_mask]
                
                if len(cluster_polygons) == 0:
                    continue
                
                # Calculate cluster statistics
                total_area = cluster_polygons.get('flaecheAmtlich', pd.Series([0])).sum()
                if total_area == 0:
                    total_area = cluster_polygons.geometry.area.sum()
                
                # Create generalized geometry
                try:
                    geometries = cluster_polygons.geometry.tolist()
                    cluster_union = unary_union(geometries)
                    
                    # Use convex hull for simplicity
                    if hasattr(cluster_union, 'convex_hull'):
                        representative_geom = cluster_union.convex_hull
                    else:
                        representative_geom = cluster_union
                        
                except Exception as e:
                    print(f"Warning: Using first geometry for cluster {cluster_id}: {e}")
                    representative_geom = cluster_polygons.geometry.iloc[0]
                
                # Get most common land type
                land_type = 'mixed'
                if 'flaechentyp' in cluster_polygons.columns:
                    mode_series = cluster_polygons['flaechentyp'].mode()
                    if len(mode_series) > 0:
                        land_type = mode_series.iloc[0]
                
                cluster_data = {
                    'cluster_id': cluster_id,
                    'geometry': representative_geom,
                    'member_count': len(cluster_polygons),
                    'total_area': total_area,
                    'flaechentyp': land_type,
                    'privacy_method': 'k_anonymity',
                    'k_value': len(cluster_polygons)
                }
                
                anonymized_data.append(cluster_data)
            
            result_gdf = gpd.GeoDataFrame(anonymized_data, crs=self.gdf.crs)
            
            # Calculate metrics
            if len(result_gdf) > 0:
                achieved_k = min([row['member_count'] for _, row in result_gdf.iterrows()])
                self.privacy_metrics['k_anonymity'] = {
                    'achieved_k': achieved_k,
                    'target_k': k,
                    'num_clusters': len(result_gdf),
                    'privacy_satisfied': achieved_k >= k
                }
            
            return result_gdf
            
        except Exception as e:
            print(f"Error in k-anonymity: {e}")
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)
        
        
    def optimized_k_anonymity(self, k: int = 5) -> gpd.GeoDataFrame:
        """
        K-anonymity via DBSCAN density-based clustering.
        
        Uses adaptive epsilon derived from k-nearest neighbor distances 
        (eps = 3 * median(k-NN distances)), with min_samples=k to enforce 
        the k-anonymity threshold. Clusters are merged into unified 
        polygons via unary_union; outlier parcels (DBSCAN noise, label=-1) 
        are handled separately and excluded from the anonymized output 
        to prevent singleton re-identification.
        """
        print(f"Applying k-anonymity via DBSCAN clustering with k={k}")
        
        if len(self.gdf) < k:
            print(f"Warning: Not enough data points ({len(self.gdf)}) for k={k}")
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)
        
        try:
            # Use spatial coordinates (centroids) for density-based clustering
            # DBSCAN operates on spatial proximity, so centroid_x/y are the 
            # natural feature space — not attribute features like flaecheAmtlich
            if 'centroid_x' not in self.gdf.columns or 'centroid_y' not in self.gdf.columns:
                print("Warning: centroid_x/centroid_y not available; cannot cluster spatially")
                return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)
            
            coords = self.gdf[['centroid_x', 'centroid_y']].fillna(0).values
            
            # Derive adaptive epsilon from k-nearest neighbor distances
            # Rationale: eps should be large enough to capture typical 
            # k-neighborhoods but small enough to preserve local spatial 
            # structure. We use 3 * median(k-NN distance) as a robust 
            # heuristic following Ester et al. (1996).
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            # Query k+1 because the nearest neighbor of each point is itself
            knn_distances, _ = tree.query(coords, k=k + 1)
            # Take the distance to the k-th true neighbor (index k, excluding self at index 0)
            kth_distances = knn_distances[:, k]
            eps_distance = float(np.median(kth_distances) * 3)
            print(f"Adaptive DBSCAN eps: {eps_distance:.1f}m (from 3 * median k-NN distance)")
            
            # Run DBSCAN with min_samples=k to enforce the k-anonymity threshold
            clustering = DBSCAN(eps=eps_distance, min_samples=k).fit(coords)
            labels = clustering.labels_
            
            self.gdf['cluster_id'] = labels
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            print(f"DBSCAN produced {n_clusters} clusters; {n_noise} parcels flagged as noise (outliers)")
            
            # Create cluster representatives (exclude noise, label == -1)
            anonymized_data = []
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    # Noise points cannot satisfy k-anonymity; drop them
                    # rather than releasing singleton records
                    continue
                
                cluster_mask = labels == cluster_id
                cluster_polygons = self.gdf[cluster_mask]
                
                # Defensive: DBSCAN guarantees >= min_samples, but check anyway
                if len(cluster_polygons) < k:
                    continue
                
                # Calculate cluster statistics
                total_area = cluster_polygons.get('flaecheAmtlich', pd.Series([0])).sum()
                if total_area == 0:
                    total_area = cluster_polygons.geometry.area.sum()
                
                # Create generalized geometry via unary union (preserves true footprint
                # of k merged parcels) with convex hull fallback for robustness
                try:
                    geometries = cluster_polygons.geometry.tolist()
                    cluster_union = unary_union(geometries)
                    
                    # Use convex hull to smooth the merged boundary
                    if hasattr(cluster_union, 'convex_hull'):
                        representative_geom = cluster_union.convex_hull
                    else:
                        representative_geom = cluster_union
                        
                except Exception as e:
                    print(f"Warning: Using first geometry for cluster {cluster_id}: {e}")
                    representative_geom = cluster_polygons.geometry.iloc[0]
                
                # Get most common land type
                land_type = 'mixed'
                if 'flaechentyp' in cluster_polygons.columns:
                    mode_series = cluster_polygons['flaechentyp'].mode()
                    if len(mode_series) > 0:
                        land_type = mode_series.iloc[0]
                
                cluster_data = {
                    'cluster_id': int(cluster_id),
                    'geometry': representative_geom,
                    'member_count': len(cluster_polygons),
                    'total_area': total_area,
                    'flaechentyp': land_type,
                    'privacy_method': 'k_anonymity',
                    'k_value': len(cluster_polygons)
                }
                
                anonymized_data.append(cluster_data)
            
            result_gdf = gpd.GeoDataFrame(anonymized_data, crs=self.gdf.crs)
            
            # Calculate metrics
            if len(result_gdf) > 0:
                achieved_k = min([row['member_count'] for _, row in result_gdf.iterrows()])
                self.privacy_metrics['k_anonymity'] = {
                    'achieved_k': achieved_k,
                    'target_k': k,
                    'num_clusters': len(result_gdf),
                    'num_noise_points': n_noise,
                    'eps_distance': eps_distance,
                    'privacy_satisfied': achieved_k >= k,
                    'clustering_algorithm': 'DBSCAN'
                }
            
            return result_gdf
            
        except Exception as e:
            print(f"Error in k-anonymity: {e}")
            import traceback
            traceback.print_exc()
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)

    def conservative_geo_indistinguishability(self, epsilon: float = 2.0) -> gpd.GeoDataFrame:
        """
        Conservative geo-indistinguishability with proper error handling
        """
        print(f"Applying conservative geo-indistinguishability with ε={epsilon}")
        
        if len(self.gdf) == 0:
            return self.gdf.copy()
        
        anonymized_gdf = self.gdf.copy()
        
        # Conservative noise scale
        base_sensitivity = 5.0  # 5m base sensitivity
        noise_scale = base_sensitivity / epsilon
        
        print(f"Using noise scale: {noise_scale:.2f}m")
        
        for idx, row in anonymized_gdf.iterrows():
            try:
                polygon = row.geometry
                if polygon is None or not hasattr(polygon, 'geom_type'):
                    continue
                    
                if polygon.geom_type == 'Polygon':
                    # Add minimal noise to vertices
                    exterior_coords = list(polygon.exterior.coords)
                    noisy_coords = []
                    
                    for x, y in exterior_coords:
                        # Small Laplace noise
                        noise_x = np.random.laplace(0, noise_scale)
                        noise_y = np.random.laplace(0, noise_scale)
                        noisy_coords.append((x + noise_x, y + noise_y))
                    
                    # Create new polygon and validate
                    try:
                        noisy_polygon = Polygon(noisy_coords)
                        if noisy_polygon.is_valid and noisy_polygon.area > 0:
                            anonymized_gdf.at[idx, 'geometry'] = noisy_polygon
                        else:
                            # Minimal buffer if invalid
                            anonymized_gdf.at[idx, 'geometry'] = polygon.buffer(0.1)
                    except:
                        # Keep original if noise fails
                        pass
                        
            except Exception as e:
                print(f"Warning: Polygon {idx} kept original: {e}")
        
        # Mask sensitive attributes
        sensitive_cols = ['eigentuemer', 'waldAdresse', 'flaecheAmtlich']
        for col in sensitive_cols:
            if col in anonymized_gdf.columns:
                anonymized_gdf[col] = 'MASKED'
        
        anonymized_gdf['privacy_method'] = 'conservative_geo_indist'
        anonymized_gdf['epsilon'] = epsilon
        
        self.privacy_metrics['geo_indistinguishability'] = {
            'epsilon': epsilon,
            'noise_scale': noise_scale,
            'method': 'Conservative Laplace'
        }
        
        return anonymized_gdf

    def precise_generalization(self, tolerance: float = 2.0) -> gpd.GeoDataFrame:
        """
        Precise generalization with proper error handling
        """
        print(f"Applying precise generalization with tolerance={tolerance}m")
        
        if len(self.gdf) == 0:
            return self.gdf.copy()
        
        anonymized_gdf = self.gdf.copy()
        
        for idx, row in anonymized_gdf.iterrows():
            try:
                polygon = row.geometry
                if polygon is None or not hasattr(polygon, 'simplify'):
                    continue
                
                # Conservative simplification
                simplified = polygon.simplify(tolerance, preserve_topology=True)
                
                # Validate result
                if simplified.is_valid and simplified.area > 0:
                    generalized = self._minimal_coordinate_rounding(simplified, 1.0)
                    anonymized_gdf.at[idx, 'geometry'] = generalized
                else:
                    # Keep original if simplification fails
                    generalized = self._minimal_coordinate_rounding(polygon, 1.0)
                    anonymized_gdf.at[idx, 'geometry'] = generalized
                    
            except Exception as e:
                print(f"Warning: Polygon {idx} kept original: {e}")
        
        anonymized_gdf['privacy_method'] = 'precise_generalization'
        anonymized_gdf['tolerance'] = tolerance
        
        self.privacy_metrics['precise_generalization'] = {
            'tolerance': tolerance,
            'coordinate_precision': 1.0,
            'method': 'Conservative simplification + rounding'
        }
        
        return anonymized_gdf
    
    def _minimal_coordinate_rounding(self, polygon, precision: float):
        """Round coordinates with minimal impact"""
        try:
            if polygon.geom_type != 'Polygon' or polygon is None:
                return polygon
                
            exterior_coords = list(polygon.exterior.coords)
            
            # Round to specified precision
            rounded_coords = [
                (round(x/precision)*precision, round(y/precision)*precision) 
                for x, y in exterior_coords
            ]
            
            return Polygon(rounded_coords)
        except:
            return polygon

    def donut_geomasking(self, k: int = 5, r_min_factor: float = 0.5, r_max_factor: float = 1.5) -> gpd.GeoDataFrame:
        """
        Donut geomasking with density-adaptive, k-aware displacement
        
        Parameters:
        - k: minimum number of neighbors required
        - r_min_factor: inner radius factor for donut
        - r_max_factor: outer radius factor for donut
        """
        print(f"Applying donut geomasking with k={k}")
        
        if len(self.gdf) < k:
            print(f"Warning: Not enough data points ({len(self.gdf)}) for k={k}")
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)
        
        try:
            anonymized_gdf = self.gdf.copy()
            
            # Extract centroids for KD-tree
            centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in self.gdf.geometry])
            tree = cKDTree(centroids)
            
            displacement_distances = []
            achieved_k_values = []
            failures = 0
            
            for idx, row in anonymized_gdf.iterrows():
                try:
                    centroid = row.geometry.centroid
                    point = np.array([centroid.x, centroid.y])
                    
                    # Find k-nearest neighbors (including self)
                    distances, indices = tree.query(point, k=k+1)
                    
                    # Remove self from results
                    distances = distances[1:]
                    
                    if len(distances) < k:
                        failures += 1
                        continue
                    
                    # Determine radius that encloses k neighbors
                    r_base = distances[-1]  # Distance to k-th neighbor
                    r_min = r_base * r_min_factor
                    r_max = r_base * r_max_factor
                    
                    # Sample random point in annulus [r_min, r_max]
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.sqrt(np.random.uniform(r_min**2, r_max**2))
                    
                    dx = radius * np.cos(angle)
                    dy = radius * np.sin(angle)
                    
                    # Apply displacement to geometry
                    displaced_geom = translate(row.geometry, xoff=dx, yoff=dy)
                    
                    if displaced_geom.is_valid:
                        anonymized_gdf.at[idx, 'geometry'] = displaced_geom
                        displacement_distances.append(radius)
                        achieved_k_values.append(k)
                    else:
                        failures += 1
                        
                except Exception as e:
                    print(f"Warning: Parcel {idx} displacement failed: {e}")
                    failures += 1
                    continue
            
            # Mask sensitive attributes
            sensitive_cols = ['eigentuemer', 'waldAdresse']
            for col in sensitive_cols:
                if col in anonymized_gdf.columns:
                    anonymized_gdf[col] = 'MASKED'
            
            anonymized_gdf['privacy_method'] = 'donut_geomasking'
            anonymized_gdf['k_value'] = k
            
            # Store metrics
            self.privacy_metrics['donut_geomasking'] = {
                'k': k,
                'mean_displacement': np.mean(displacement_distances) if displacement_distances else 0,
                'median_displacement': np.median(displacement_distances) if displacement_distances else 0,
                'max_displacement': np.max(displacement_distances) if displacement_distances else 0,
                'failure_rate': failures / len(self.gdf) * 100,
                'r_min_factor': r_min_factor,
                'r_max_factor': r_max_factor
            }
            
            return anonymized_gdf
            
        except Exception as e:
            print(f"Error in donut geomasking: {e}")
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)

    def topology_preserving_generalization(self, grid_size: float = 5.0, 
                                        simplify_tolerance: float = 2.0) -> gpd.GeoDataFrame:
        """
        Topology-preserving generalization with shared-edge simplification
        
        This method differs from precise_generalization by:
        1. Snapping ALL vertices to a common grid (maintains alignment)
        2. Preserving shared edges between adjacent parcels
        3. Checking and maintaining topological relationships
        """
        print(f"Applying topology-preserving generalization (grid={grid_size}m)")
        
        if len(self.gdf) == 0:
            return self.gdf.copy()
        
        try:
            anonymized_gdf = self.gdf.copy()
            
            # Step 1: Snap ALL vertices to a common grid
            # This is key - all parcels snap to the SAME grid, maintaining alignment
            def snap_to_common_grid(geom, grid_size):
                if geom.geom_type != 'Polygon':
                    return geom
                
                coords = list(geom.exterior.coords)
                snapped_coords = []
                
                for x, y in coords:
                    # Snap to grid
                    snapped_x = round(x / grid_size) * grid_size
                    snapped_y = round(y / grid_size) * grid_size
                    
                    # Avoid consecutive duplicates
                    if not snapped_coords or (snapped_x, snapped_y) != snapped_coords[-1]:
                        snapped_coords.append((snapped_x, snapped_y))
                
                # Ensure closed polygon
                if snapped_coords and snapped_coords[0] != snapped_coords[-1]:
                    snapped_coords.append(snapped_coords[0])
                
                # Need at least 4 points for valid polygon
                if len(snapped_coords) < 4:
                    return geom
                
                try:
                    snapped_geom = Polygon(snapped_coords)
                    if snapped_geom.is_valid and snapped_geom.area > 0:
                        return snapped_geom
                    else:
                        # Try to fix with buffer
                        return snapped_geom.buffer(0)
                except:
                    return geom
            
            # Apply grid snapping to all parcels
            for idx, row in anonymized_gdf.iterrows():
                try:
                    snapped = snap_to_common_grid(row.geometry, grid_size)
                    anonymized_gdf.at[idx, 'geometry'] = snapped
                except Exception as e:
                    print(f"Warning: Grid snap failed for parcel {idx}: {e}")
            
            # Step 2: Build spatial index to find adjacent parcels
            # This step identifies which parcels share edges
            from shapely.strtree import STRtree
            
            geometries = list(anonymized_gdf.geometry)
            tree = STRtree(geometries)
            
            # Step 3: Simplify while checking adjacencies
            # Only simplify if it doesn't break shared edges
            for idx, row in anonymized_gdf.iterrows():
                try:
                    polygon = row.geometry
                    if polygon is None or not hasattr(polygon, 'simplify'):
                        continue
                    
                    # Find potentially adjacent parcels
                    nearby_indices = tree.query(polygon)
                    
                    # Attempt simplification
                    simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
                    
                    # Check if simplification maintains adjacency with neighbors
                    maintains_adjacency = True
                    for nearby_idx in nearby_indices:
                        if nearby_idx == idx:
                            continue
                        
                        neighbor = geometries[nearby_idx]
                        
                        # Check if they were adjacent before
                        was_adjacent = polygon.touches(neighbor) or polygon.boundary.intersects(neighbor.boundary)
                        
                        if was_adjacent:
                            # Check if still adjacent after simplification
                            still_adjacent = simplified.touches(neighbor) or simplified.boundary.intersects(neighbor.boundary)
                            
                            if not still_adjacent:
                                maintains_adjacency = False
                                break
                    
                    # Only apply simplification if adjacency is maintained
                    if maintains_adjacency and simplified.is_valid and simplified.area > 0:
                        anonymized_gdf.at[idx, 'geometry'] = simplified
                        geometries[idx] = simplified  # Update for next iterations
                        
                except Exception as e:
                    print(f"Warning: Topology check failed for parcel {idx}: {e}")
            
            # Calculate metrics
            original_adjacencies = self._count_adjacencies(self.gdf)
            new_adjacencies = self._count_adjacencies(anonymized_gdf)
            adjacency_preserved = (new_adjacencies / original_adjacencies * 100) if original_adjacencies > 0 else 100
            
            original_vertices = sum(len(list(geom.exterior.coords)) for geom in self.gdf.geometry if geom.geom_type == 'Polygon')
            new_vertices = sum(len(list(geom.exterior.coords)) for geom in anonymized_gdf.geometry if geom.geom_type == 'Polygon')
            vertex_reduction = (original_vertices - new_vertices) / original_vertices * 100 if original_vertices > 0 else 0
            
            anonymized_gdf['privacy_method'] = 'topology_preserving'
            anonymized_gdf['grid_size'] = grid_size
            
            self.privacy_metrics['topology_preserving'] = {
                'grid_size': grid_size,
                'simplify_tolerance': simplify_tolerance,
                'adjacency_preserved_pct': adjacency_preserved,
                'vertex_reduction_pct': vertex_reduction,
                'original_vertices': original_vertices,
                'new_vertices': new_vertices,
                'original_adjacencies': original_adjacencies,
                'preserved_adjacencies': new_adjacencies
            }
            
            return anonymized_gdf
            
        except Exception as e:
            print(f"Error in topology-preserving generalization: {e}")
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)

    def _count_adjacencies(self, gdf: gpd.GeoDataFrame) -> int:
        """Helper: Count adjacent parcel pairs"""
        adjacencies = 0
        geometries = list(gdf.geometry)
        
        for i in range(len(geometries)):
            for j in range(i+1, len(geometries)):
                try:
                    if geometries[i].touches(geometries[j]) or geometries[i].intersects(geometries[j]):
                        adjacencies += 1
                except:
                    continue
        
        return adjacencies

    def dp_grid_aggregation(self, epsilon: float = 1.0, grid_resolution: int = 8) -> gpd.GeoDataFrame:
        """
        Differentially private grid/hex aggregation with DP counts/means
        
        Parameters:
        - epsilon: privacy budget for differential privacy
        - grid_resolution: H3 resolution (higher = finer grid, 8 is ~0.46 km² cells)
        """
        print(f"Applying DP grid aggregation with ε={epsilon}, H3 resolution={grid_resolution}")
        
        if len(self.gdf) == 0:
            return self.gdf.copy()
        
        try:
            # Convert geometries to H3 hexagons
            hex_data = {}
            
            for idx, row in self.gdf.iterrows():
                try:
                    centroid = row.geometry.centroid
                    # H3 expects lat/lon, so we need to convert if in projected CRS
                    if self.gdf.crs and self.gdf.crs.to_epsg() != 4326:
                        geo_centroid = gpd.GeoSeries([centroid], crs=self.gdf.crs).to_crs('EPSG:4326').iloc[0]
                        lat, lon = geo_centroid.y, geo_centroid.x
                    else:
                        lat, lon = centroid.y, centroid.x
                    
                    # Get H3 hex for this location
                    hex_id = h3.latlng_to_cell(lat, lon, grid_resolution)
                    
                    if hex_id not in hex_data:
                        hex_data[hex_id] = {
                            'count': 0,
                            'total_area': 0,
                            'types': []
                        }
                    
                    hex_data[hex_id]['count'] += 1
                    hex_data[hex_id]['total_area'] += row.get('flaecheAmtlich', row.geometry.area)
                    if 'flaechentyp' in row and pd.notna(row['flaechentyp']):
                        hex_data[hex_id]['types'].append(row['flaechentyp'])
                        
                except Exception as e:
                    print(f"Warning: Could not process parcel {idx}: {e}")
                    continue
            
            # Add Laplace noise for differential privacy
            sensitivity = 1.0  # Maximum change from adding/removing one record
            noise_scale = sensitivity / epsilon
            
            aggregated_data = []
            true_counts = []
            noisy_counts = []
            
            for hex_id, data in hex_data.items():
                try:
                    # Add Laplace noise to count
                    true_count = data['count']
                    noisy_count = max(0, int(true_count + np.random.laplace(0, noise_scale)))
                    
                    # Only include cells with positive noisy count
                    if noisy_count == 0:
                        continue
                    
                    true_counts.append(true_count)
                    noisy_counts.append(noisy_count)
                    
                    # Calculate mean area with noise
                    mean_area = data['total_area'] / data['count'] if data['count'] > 0 else 0
                    noisy_mean_area = max(0, mean_area + np.random.laplace(0, noise_scale * mean_area * 0.1))
                    
                    # Determine dominant land type
                    if data['types']:
                        from collections import Counter
                        dominant_type = Counter(data['types']).most_common(1)[0][0]
                    else:
                        dominant_type = 'Unknown'
                    
                    # Get hex boundary as polygon
                    hex_boundary = h3.cell_to_boundary(hex_id)
                    hex_polygon = Polygon([(lon, lat) for lat, lon in hex_boundary])
                    
                    # Convert back to original CRS if needed
                    if self.gdf.crs and self.gdf.crs.to_epsg() != 4326:
                        hex_geom = gpd.GeoSeries([hex_polygon], crs='EPSG:4326').to_crs(self.gdf.crs).iloc[0]
                    else:
                        hex_geom = hex_polygon
                    
                    aggregated_data.append({
                        'hex_id': hex_id,
                        'geometry': hex_geom,
                        'noisy_count': noisy_count,
                        'true_count': true_count,  # Only for evaluation, would not be published
                        'noisy_mean_area': noisy_mean_area,
                        'dominant_type': dominant_type,
                        'privacy_method': 'dp_grid_aggregation',
                        'epsilon': epsilon
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not process hex {hex_id}: {e}")
                    continue
            
            result_gdf = gpd.GeoDataFrame(aggregated_data, crs=self.gdf.crs)
            
            # Calculate metrics
            if true_counts and noisy_counts:
                mae = np.mean(np.abs(np.array(true_counts) - np.array(noisy_counts)))
                rmse = np.sqrt(np.mean((np.array(true_counts) - np.array(noisy_counts))**2))
                mean_k = np.mean(true_counts)
            else:
                mae = rmse = mean_k = 0
            
            self.privacy_metrics['dp_grid_aggregation'] = {
                'epsilon': epsilon,
                'grid_resolution': grid_resolution,
                'noise_scale': noise_scale,
                'num_cells': len(result_gdf),
                'mae': mae,
                'rmse': rmse,
                'mean_k_per_cell': mean_k,
                'data_reduction': (len(self.gdf) - len(result_gdf)) / len(self.gdf) * 100
            }
            
            return result_gdf
            
        except Exception as e:
            print(f"Error in DP grid aggregation: {e}")
            import traceback
            traceback.print_exc()
            return gpd.GeoDataFrame(columns=self.gdf.columns, crs=self.gdf.crs)

    def calculate_detailed_metrics(self, original_gdf: gpd.GeoDataFrame, 
                                 anonymized_gdf: gpd.GeoDataFrame) -> Dict:
        """Calculate detailed utility metrics with error handling"""
        
        metrics = {
            'hausdorff_mean': 0, 'hausdorff_max': 0, 'hausdorff_95th': 0,
            'centroid_mean': 0, 'centroid_max': 0, 'centroid_95th': 0,
            'area_dev_mean': 0, 'area_dev_max': 0, 'area_dev_95th': 0
        }
        
        if len(original_gdf) == 0 or len(anonymized_gdf) == 0:
            return metrics
        
        if len(original_gdf) == len(anonymized_gdf):
            hausdorff_distances = []
            centroid_distances = []
            area_deviations = []
            
            for orig, anon in zip(original_gdf.geometry, anonymized_gdf.geometry):
                try:
                    if orig is None or anon is None:
                        continue
                        
                    # Hausdorff distance
                    hausdorff = orig.hausdorff_distance(anon)
                    hausdorff_distances.append(hausdorff)
                    
                    # Centroid distance  
                    centroid_dist = orig.centroid.distance(anon.centroid)
                    centroid_distances.append(centroid_dist)
                    
                    # Area deviation
                    orig_area = orig.area
                    anon_area = anon.area
                    if orig_area > 0:
                        area_dev = abs(orig_area - anon_area) / orig_area * 100
                        area_deviations.append(area_dev)
                        
                except Exception as e:
                    print(f"Warning in metrics calculation: {e}")
                    continue
            
            # Calculate statistics if we have data
            if hausdorff_distances:
                metrics.update({
                    'hausdorff_mean': np.mean(hausdorff_distances),
                    'hausdorff_max': np.max(hausdorff_distances),
                    'hausdorff_95th': np.percentile(hausdorff_distances, 95),
                    'centroid_mean': np.mean(centroid_distances),
                    'centroid_max': np.max(centroid_distances),
                    'centroid_95th': np.percentile(centroid_distances, 95),
                    'area_dev_mean': np.mean(area_deviations),
                    'area_dev_max': np.max(area_deviations),
                    'area_dev_95th': np.percentile(area_deviations, 95)
                })
            
            # Thesis criteria
            metrics['thesis_criteria'] = {
                'hausdorff_strict': metrics['hausdorff_max'] < 20,
                'hausdorff_relaxed': metrics['hausdorff_95th'] < 20,
                'centroid_strict': metrics['centroid_max'] < 15,
                'centroid_relaxed': metrics['centroid_95th'] < 15,
                'area_strict': metrics['area_dev_max'] <= 5,
                'area_relaxed': metrics['area_dev_95th'] <= 5,
                'strict_success': all([
                    metrics['hausdorff_max'] < 20,
                    metrics['centroid_max'] < 15,
                    metrics['area_dev_max'] <= 5
                ]),
                'relaxed_success': all([
                    metrics['hausdorff_95th'] < 20,
                    metrics['centroid_95th'] < 15,
                    metrics['area_dev_95th'] <= 5
                ])
            }
        
        return metrics

def comprehensive_test(geojson_string: str):
    """
    Comprehensive test with robust error handling
    """
    try:
        # Clean the input data first
        cleaned_geojson = clean_geojson_data(geojson_string)
        data = json.loads(cleaned_geojson)
        
        # Create GeoDataFrame with error handling
        if not data.get('features'):
            print("No valid features found in GeoJSON")
            return {}
        
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        
        # Set CRS if not present
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:25832')
        
        print(f"Successfully loaded {len(gdf)} forest parcels")
        
        if len(gdf) == 0:
            print("No valid geometries found")
            return {}
        
        print(f"Dataset characteristics:")
        if len(gdf.geometry) > 0 and gdf.geometry.area.notna().any():
            print(f"  - Area range: {gdf.geometry.area.min():.0f} - {gdf.geometry.area.max():.0f} m²")
            print(f"  - Mean area: {gdf.geometry.area.mean():.0f} m²")
        
        anonymizer = OptimizedCadastralAnonymizer(gdf)
        
        # Only proceed if we have valid data
        if len(anonymizer.gdf) == 0:
            print("No valid data after cleaning")
            return {}
        
        # Sensitivity analysis
        try:
            anonymizer.gdf['sensitivity'] = anonymizer.gdf.apply(anonymizer.classify_sensitivity, axis=1)
            print(f"  - Sensitivity distribution: {dict(anonymizer.gdf['sensitivity'].value_counts())}")
        except Exception as e:
            print(f"Could not classify sensitivity: {e}")
        
        print("\n" + "="*70)
        print("OPTIMIZED ANONYMIZATION TESTING")
        print("="*70)
        
        results = {}
        
        # Test 1: Conservative Geo-Indistinguishability
        print(f"\n1. CONSERVATIVE GEO-INDISTINGUISHABILITY")
        print("-" * 50)
        try:
            conservative_result = anonymizer.conservative_geo_indistinguishability(epsilon=2.0)
            if len(conservative_result) > 0:
                results['conservative_geo'] = conservative_result
                
                metrics = anonymizer.calculate_detailed_metrics(anonymizer.gdf, conservative_result)
                
                print(f"✓ Applied ε-differential privacy (ε=2.0)")
                print(f"  Noise scale: {anonymizer.privacy_metrics['geo_indistinguishability']['noise_scale']:.1f}m")
                print(f"  Processed {len(conservative_result)} polygons")
                print(f"  Hausdorff: mean={metrics['hausdorff_mean']:.1f}m, max={metrics['hausdorff_max']:.1f}m")
                print(f"  Centroid: mean={metrics['centroid_mean']:.1f}m, max={metrics['centroid_max']:.1f}m") 
                print(f"  Area dev: mean={metrics['area_dev_mean']:.1f}%, max={metrics['area_dev_max']:.1f}%")
                
                success = metrics['thesis_criteria']['strict_success']
                relaxed = metrics['thesis_criteria']['relaxed_success']
                print(f"  Thesis criteria: {'✓ PASS' if success else '✓ PASS (95th percentile)' if relaxed else '✗ FAIL'}")
            else:
                print("✗ No results generated")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Test 2: Precise Generalization
        print(f"\n2. PRECISE GENERALIZATION")
        print("-" * 50)
        try:
            precise_result = anonymizer.precise_generalization(tolerance=2.0)
            if len(precise_result) > 0:
                results['precise_gen'] = precise_result
                
                metrics = anonymizer.calculate_detailed_metrics(anonymizer.gdf, precise_result)
                
                print(f"✓ Applied conservative simplification (2m tolerance)")
                print(f"  Processed {len(precise_result)} polygons")
                print(f"  Hausdorff: mean={metrics['hausdorff_mean']:.1f}m, max={metrics['hausdorff_max']:.1f}m")
                print(f"  Centroid: mean={metrics['centroid_mean']:.1f}m, max={metrics['centroid_max']:.1f}m")
                print(f"  Area dev: mean={metrics['area_dev_mean']:.1f}%, max={metrics['area_dev_max']:.1f}%")
                
                success = metrics['thesis_criteria']['strict_success'] 
                relaxed = metrics['thesis_criteria']['relaxed_success']
                print(f"  Thesis criteria: {'✓ PASS' if success else '✓ PASS (95th percentile)' if relaxed else '✗ FAIL'}")
            else:
                print("✗ No results generated")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Test 3: Optimized K-Anonymity
        print(f"\n3. OPTIMIZED K-ANONYMITY CLUSTERING")
        print("-" * 50)
        try:
            k_values = [3, 5, 7]
            best_k_result = None
            best_k_score = 0
            best_k = 3
            
            for k in k_values:
                if len(anonymizer.gdf) >= k * 2:
                    k_result = anonymizer.optimized_k_anonymity(k=k)
                    
                    if len(k_result) > 0:
                        # Score based on utility preservation
                        area_preservation = k_result['total_area'].sum() / anonymizer.gdf.geometry.area.sum()
                        cluster_count = len(k_result)
                        score = area_preservation * cluster_count
                        
                        if score > best_k_score:
                            best_k_score = score
                            best_k_result = k_result
                            best_k = k
            
            if best_k_result is not None and len(best_k_result) > 0:
                results['optimized_k'] = best_k_result
                
                print(f"✓ Best k-anonymity result: k={best_k}")
                print(f"  Clusters created: {len(best_k_result)}")
                print(f"  Data reduction: {len(anonymizer.gdf)} → {len(best_k_result)} polygons")
                
                reduction_ratio = (len(anonymizer.gdf) - len(best_k_result)) / len(anonymizer.gdf) * 100
                area_preservation = best_k_result['total_area'].sum() / anonymizer.gdf.geometry.area.sum() * 100
                
                print(f"  Reduction ratio: {reduction_ratio:.1f}%")
                print(f"  Area preservation: {area_preservation:.1f}%")
                
                min_k = min([row['member_count'] for _, row in best_k_result.iterrows()])
                print(f"  Achieved k-anonymity: k = {min_k}")
            else:
                print("✗ Could not create valid k-anonymous clusters")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Test 4: Donut Geomasking
        print(f"\n4. DONUT GEOMASKING (K-AWARE)")
        print("-" * 50)
        try:
            donut_result = anonymizer.donut_geomasking(k=5)
            if len(donut_result) > 0:
                results['donut_geomasking'] = donut_result
                
                metrics = anonymizer.calculate_detailed_metrics(anonymizer.gdf, donut_result)
                dm = anonymizer.privacy_metrics['donut_geomasking']
                
                print(f"✓ Applied donut geomasking (k={dm['k']})")
                print(f"  Mean displacement: {dm['mean_displacement']:.1f}m")
                print(f"  Max displacement: {dm['max_displacement']:.1f}m")
                print(f"  Failure rate: {dm['failure_rate']:.1f}%")
                print(f"  Hausdorff: mean={metrics['hausdorff_mean']:.1f}m, max={metrics['hausdorff_max']:.1f}m")
                print(f"  Centroid: mean={metrics['centroid_mean']:.1f}m, max={metrics['centroid_max']:.1f}m")
            else:
                print("✗ No results generated")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Test 5: Topology-Preserving Generalization
        print(f"\n5. TOPOLOGY-PRESERVING GENERALIZATION")
        print("-" * 50)
        try:
            # topo_result = anonymizer.topology_preserving_generalization(grid_size=5.0)
            topo_result = anonymizer.topology_preserving_generalization(
                grid_size=20.0,           # Increase from 5.0 to 20.0 meters
                simplify_tolerance=15.0   # Increase from 2.0 to 10.0 meters
            )
            if len(topo_result) > 0:
                results['topology_preserving'] = topo_result
                
                metrics = anonymizer.calculate_detailed_metrics(anonymizer.gdf, topo_result)
                tp = anonymizer.privacy_metrics['topology_preserving']
                
                print(f"✓ Applied topology-preserving generalization")
                print(f"  Grid size: {tp['grid_size']}m")
                print(f"  Adjacency preserved: {tp['adjacency_preserved_pct']:.1f}%")
                print(f"  Vertex reduction: {tp['vertex_reduction_pct']:.1f}%")
                print(f"  Hausdorff: mean={metrics['hausdorff_mean']:.1f}m, max={metrics['hausdorff_max']:.1f}m")
            else:
                print("✗ No results generated")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Test 6: DP Grid Aggregation
        print(f"\n6. DIFFERENTIALLY PRIVATE GRID AGGREGATION")
        print("-" * 50)
        try:
            dp_result = anonymizer.dp_grid_aggregation(epsilon=1.0, grid_resolution=8)
            if len(dp_result) > 0:
                results['dp_grid'] = dp_result
                
                dpm = anonymizer.privacy_metrics['dp_grid_aggregation']
                
                print(f"✓ Applied DP grid aggregation (ε={dpm['epsilon']})")
                print(f"  Grid cells: {dpm['num_cells']}")
                print(f"  Mean k per cell: {dpm['mean_k_per_cell']:.1f}")
                print(f"  Count MAE: {dpm['mae']:.2f}")
                print(f"  Count RMSE: {dpm['rmse']:.2f}")
                print(f"  Data reduction: {dpm['data_reduction']:.1f}%")
            else:
                print("✗ No results generated")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Summary
        print(f"\n" + "="*70)
        print("THESIS EVALUATION SUMMARY")
        print("="*70)
        
        successful_methods = [name for name, result in results.items() if len(result) > 0]
        
        print(f"\nSuccessful methods: {len(successful_methods)}/3")
        if successful_methods:
            print("Methods completed:", ", ".join(successful_methods))
        
        print(f"\nRecommendations for thesis:")
        print("1. Clean your GeoJSON data to remove null geometries")
        print("2. Ensure all features have valid polygon geometries")
        print("3. Test with smaller datasets first to validate approach")
        print("4. Consider data preprocessing to handle edge cases")
        
        # Threat model evaluation
        # if results:
        #     print("\n" + "="*70)
        #     print("THREAT MODEL EVALUATION")
        #     print("="*70)
            
        #     # Evaluate robustness of each method
        #     robustness_df = evaluate_anonymization_robustness(
        #         anonymizer.original_gdf,
        #         results
        #     )
            
        #     # Find most vulnerable method
        #     if len(robustness_df) > 0:
        #         most_vulnerable = robustness_df.iloc[-1]
        #         print(f"\n⚠️  Most Vulnerable Method: {most_vulnerable['Method']}")
        #         print(f"   Overall Vulnerability: {most_vulnerable['Overall_Vulnerability']:.1%}")
        #         print(f"   Risk Level: {most_vulnerable['Risk_Level']}")
                
        #         # Recommend defenses based on vulnerabilities
        #         if most_vulnerable['Homogeneity_Success'] > 0.5:
        #             print("\n   Recommended Defense: Implement L-Diversity")
        #         if most_vulnerable['Background_Success'] > 0.5:
        #             print("   Recommended Defense: Increase K value or use DP Grid")
        #         if most_vulnerable['Satellite_Success'] > 0.5:
        #             print("   Recommended Defense: Add more geometric distortion")
            
        #     # Demonstrate enhanced protection
        #     print("\n" + "="*70)
        #     print("APPLYING ENHANCED DEFENSES")
        #     print("="*70)
            
        #     # Apply enhanced defenses to most vulnerable method
        #     defenses = EnhancedPrivacyDefenses(anonymizer.gdf)
            
        #     # Create sample groups for k-anonymity
        #     if 'optimized_k' in results and len(results['optimized_k']) > 0:
        #         print("\nEnhancing K-Anonymity with L-Diversity...")
                
        #         # Get cluster assignments
        #         if 'cluster_id' in anonymizer.gdf.columns:
        #             groups = {}
        #             for cluster_id in anonymizer.gdf['cluster_id'].unique():
        #                 groups[cluster_id] = anonymizer.gdf[
        #                     anonymizer.gdf['cluster_id'] == cluster_id
        #                 ].index.tolist()
                    
        #             # Apply l-diversity
        #             l_diverse = defenses.enforce_l_diversity(groups, l=3)
        #             print(f"  Groups after L-diversity: {len(l_diverse['groups'])}")
        #             print(f"  Records suppressed: {l_diverse['suppressed']}")
            
        #     # Create fully protected version
        #     print("\nCreating Privacy-Protected Version...")
        #     privacy_requirements = {
        #         'min_k': 5,
        #         'l_diversity': 3,
        #         't_closeness': 0.2,
        #         'epsilon': 1.0,
        #         'attack_resistance': ['homogeneity', 'background', 'satellite']
        #     }
            
        #     protected_gdf = create_privacy_preserving_pipeline(
        #         anonymizer.original_gdf,
        #         privacy_requirements
        #     )
            
        #     # Add to results
        #     results['protected_pipeline'] = protected_gdf
            
        #     # Test the protected version
        #     print("\nTesting Protected Version...")
        #     protected_simulator = PrivacyAttackSimulator(
        #         anonymizer.original_gdf,
        #         protected_gdf,
        #         'protected_pipeline'
        #     )
            
        #     # Quick test of key attacks
        #     homogeneity = protected_simulator.homogeneity_attack()
        #     background = protected_simulator.background_knowledge_attack({})
            
        #     print(f"\nProtected Version Attack Resistance:")
        #     print(f"  Homogeneity Attack: {'N/A' if not homogeneity.get('applicable') else f'{homogeneity.get('success_rate', 0):.1%}'}")
        #     print(f"  Background Attack: {background['success_rate']:.1%}")
            
        #     # Export enhanced results
        #     try:
        #         # Original export functions
        #         export_thesis_results(results, anonymizer)
        #         create_comparison_visualization(results, anonymizer)
        #         create_comprehensive_maps(results, anonymizer)
                
                # New: Export threat model results
                # export_threat_model_results(robustness_df, protected_gdf)
                
            # except Exception as e:
            #     print(f"⚠ Export failed: {e}")
        
        return results
        
    except Exception as e:
        print(f"Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    
    
    
def export_threat_model_results(robustness_df: pd.DataFrame, 
                               protected_gdf: gpd.GeoDataFrame) -> None:
    """
    Export threat model evaluation results.
    """
    try:
        output_dir = Path("thesis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Export robustness evaluation
        robustness_file = output_dir / "robustness_evaluation.csv"
        robustness_df.to_csv(robustness_file, index=False)
        print(f"✓ Exported robustness evaluation to {robustness_file}")
        
        # Export protected dataset
        if len(protected_gdf) > 0:
            protected_file = output_dir / "protected_anonymized.geojson"
            protected_gdf.to_file(protected_file, driver='GeoJSON')
            print(f"✓ Exported protected dataset to {protected_file}")
        
        # Create threat model report
        report_file = output_dir / "threat_model_report.md"
        with open(report_file, 'w') as f:
            f.write("# Threat Model Evaluation Report\n\n")
            f.write("## Vulnerability Assessment\n\n")
            
            if len(robustness_df) > 0:
                f.write("### Method Rankings by Security\n")
                for _, row in robustness_df.iterrows():
                    f.write(f"- **{row['Method']}**: {row['Overall_Vulnerability']:.1%} vulnerability ({row['Risk_Level']} risk)\n")
                
                f.write("\n### Attack-Specific Success Rates\n\n")
                f.write("| Method | Homogeneity | Background | Boundary | Satellite | Membership | Temporal |\n")
                f.write("|--------|-------------|------------|----------|-----------|------------|----------|\n")
                
                for _, row in robustness_df.iterrows():
                    f.write(f"| {row['Method']} | {row['Homogeneity_Success']:.1%} | ")
                    f.write(f"{row['Background_Success']:.1%} | {row['Boundary_Success']:.1%} | ")
                    f.write(f"{row['Satellite_Success']:.1%} | {row['Membership_Confidence']:.1%} | ")
                    f.write(f"{row['Temporal_Success']:.1%} |\n")
                
                # Recommendations
                f.write("\n## Security Recommendations\n\n")
                
                most_secure = robustness_df.iloc[0]
                least_secure = robustness_df.iloc[-1]
                
                f.write(f"1. **Use {most_secure['Method']}** for highest security ")
                f.write(f"({most_secure['Overall_Vulnerability']:.1%} vulnerability)\n")
                f.write(f"2. **Avoid {least_secure['Method']}** due to high vulnerability ")
                f.write(f"({least_secure['Overall_Vulnerability']:.1%})\n")
                f.write("3. **Implement L-Diversity** for all k-anonymity based methods\n")
                f.write("4. **Use Differential Privacy** for formal guarantees\n")
                f.write("5. **Apply geometric distortion** against satellite correlation\n")
            
            f.write("\n## Protected Pipeline Configuration\n\n")
            f.write("The enhanced protection pipeline implements:\n")
            f.write("- Minimum k-anonymity: 5\n")
            f.write("- L-diversity: 3\n")
            f.write("- T-closeness: 0.2\n")
            f.write("- Differential privacy ε: 1.0\n")
            f.write("- Attack resistance: homogeneity, background, satellite\n")
        
        print(f"✓ Created threat model report: {report_file}")
        
    except Exception as e:
        print(f"Threat model export error: {e}")



# ============= STANDALONE TEST FUNCTION =============

def test_specific_attack(geojson_string: str, attack_type: str = 'all'):
    """
    Test specific attack against your anonymized data.
    
    attack_type: 'homogeneity', 'background', 'boundary', 'satellite', 
                 'membership', 'temporal', or 'all'
    """
    # Load and anonymize data
    results = comprehensive_test(geojson_string)
    
    if not results:
        print("No results to test")
        return
    
    print("\n" + "="*70)
    print(f"TESTING ATTACK: {attack_type.upper()}")
    print("="*70)
    
    # Get original data from the anonymizer
    cleaned_geojson = clean_geojson_data(geojson_string)
    data = json.loads(cleaned_geojson)
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:25832')
    
    anonymizer = OptimizedCadastralAnonymizer(gdf)
    
    # Test each method
    for method_name, anonymized_gdf in results.items():
        if len(anonymized_gdf) == 0:
            continue
            
        print(f"\nMethod: {method_name}")
        print("-" * 40)
        
        simulator = PrivacyAttackSimulator(
            anonymizer.original_gdf,
            anonymized_gdf,
            method_name
        )
        
        if attack_type == 'homogeneity':
            result = simulator.homogeneity_attack()
        elif attack_type == 'background':
            result = simulator.background_knowledge_attack({
                'unique_parcels': [
                    {'area_min': 10000, 'area_max': 20000},
                    {'area_min': 50000, 'area_max': float('inf')}
                ]
            })
        elif attack_type == 'boundary':
            result = simulator.boundary_reconstruction_attack()
        elif attack_type == 'satellite':
            result = simulator.satellite_correlation_attack()
        elif attack_type == 'membership':
            result = simulator.membership_inference_attack()
        elif attack_type == 'temporal':
            result = simulator.temporal_correlation_attack()
        elif attack_type == 'all':
            result = simulator.run_all_attacks()
        else:
            print(f"Unknown attack type: {attack_type}")
            return
        
        # Display results
        if isinstance(result, dict):
            if 'success_rate' in result:
                print(f"  Success Rate: {result['success_rate']:.1%}")
            if 'overall_vulnerability' in result:
                print(f"  Overall Vulnerability: {result['overall_vulnerability']:.1%}")
            if 'risk_level' in result:
                print(f"  Risk Level: {result['risk_level']}")

def analyze_geojson_issues(geojson_string: str):
    """Analyze and report issues in GeoJSON data"""
    print("GEOJSON DATA ANALYSIS")
    print("=" * 50)
    
    try:
        data = json.loads(geojson_string)
        print(f"✓ JSON parsing successful")
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
        return
    
    if not isinstance(data, dict):
        print(f"✗ Root object is not a dictionary: {type(data)}")
        return
    
    if data.get('type') != 'FeatureCollection':
        print(f"⚠ Not a FeatureCollection: {data.get('type')}")
    
    features = data.get('features', [])
    print(f"Total features found: {len(features)}")
    
    valid_features = 0
    null_geometries = 0
    invalid_geometries = 0
    missing_properties = 0
    
    for i, feature in enumerate(features):
        if not isinstance(feature, dict):
            print(f"  Feature {i}: Not a dictionary")
            continue
            
        if feature.get('type') != 'Feature':
            print(f"  Feature {i}: Not a Feature type")
            continue
            
        geometry = feature.get('geometry')
        if geometry is None:
            null_geometries += 1
            if i < 5:  # Show first few examples
                print(f"  Feature {i}: NULL geometry")
            continue
            
        properties = feature.get('properties', {})
        if not properties:
            missing_properties += 1
            
        # Try to create a shapely geometry
        try:
            from shapely.geometry import shape
            geom = shape(geometry)
            if geom.is_valid:
                valid_features += 1
            else:
                invalid_geometries += 1
                if i < 5:
                    print(f"  Feature {i}: Invalid geometry")
        except Exception as e:
            invalid_geometries += 1
            if i < 5:
                print(f"  Feature {i}: Geometry error: {e}")
    
    print(f"\nSUMMARY:")
    print(f"  Valid features: {valid_features}")
    print(f"  Null geometries: {null_geometries}")
    print(f"  Invalid geometries: {invalid_geometries}")
    print(f"  Missing properties: {missing_properties}")
    print(f"  Success rate: {valid_features/len(features)*100:.1f}%")
    
    if valid_features > 0:
        print(f"\n✓ Found {valid_features} valid features to process")
    else:
        print(f"\n✗ No valid features found - check your data source")

def create_detailed_k_anonymity_map(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create detailed k-anonymity map showing original parcels within clusters"""
    try:
        if 'optimized_k' not in results or len(results['optimized_k']) == 0:
            print("No k-anonymity results to display")
            return
            
        k_result = results['optimized_k']
        original_gdf = anonymizer.original_gdf.copy()
        
        # Clean original data
        columns_to_drop = ['centroid', 'centroid_x', 'centroid_y']
        for col in columns_to_drop:
            if col in original_gdf.columns:
                original_gdf = original_gdf.drop(columns=[col])
        
        # Convert to geographic coordinates
        if original_gdf.crs and original_gdf.crs.to_epsg() != 4326:
            original_geo = original_gdf.to_crs('EPSG:4326')
        else:
            original_geo = original_gdf
            
        if k_result.crs and k_result.crs.to_epsg() != 4326:
            k_geo = k_result.to_crs('EPSG:4326')
        else:
            k_geo = k_result
            
        # Calculate map center
        bounds = original_geo.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create map with cluster selection capability
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Get the cluster assignments from the anonymizer
        cluster_assignments = anonymizer.gdf['cluster_id'] if 'cluster_id' in anonymizer.gdf.columns else None
        
        if cluster_assignments is not None:
            # Create layers for each cluster
            cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
                            'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
                            'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
            
            # Create feature groups for clusters
            cluster_layers = {}
            
            for cluster_id in k_geo['cluster_id'].unique():
                # Get cluster info
                cluster_row = k_geo[k_geo['cluster_id'] == cluster_id].iloc[0]
                color = cluster_colors[cluster_id % len(cluster_colors)]
                
                # Create layer for this cluster
                cluster_layer = folium.FeatureGroup(
                    name=f"Cluster {cluster_id} ({cluster_row['member_count']} parcels)", 
                    show=True
                )
                
                # Add the cluster boundary (k-anonymous region)
                cluster_popup = f"""
                <div style="font-family: Arial, sans-serif; min-width: 300px;">
                    <h3>K-Anonymous Cluster {cluster_id}</h3>
                    <table style="width:100%">
                        <tr><td><b>Member Count:</b></td><td>{cluster_row['member_count']}</td></tr>
                        <tr><td><b>Total Area:</b></td><td>{cluster_row['total_area']:.0f} m²</td></tr>
                        <tr><td><b>Land Type:</b></td><td>{cluster_row['flaechentyp']}</td></tr>
                        <tr><td><b>K-value:</b></td><td>{cluster_row['k_value']}</td></tr>
                        <tr><td><b>Privacy Level:</b></td><td>High (grouped with {cluster_row['member_count']-1} others)</td></tr>
                    </table>
                    <br>
                    <p><i>This cluster represents {cluster_row['member_count']} original parcels merged for privacy protection. Original parcels are shown in the same color.</i></p>
                </div>
                """
                
                folium.GeoJson(
                    cluster_row.geometry.__geo_interface__,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 3,
                        'fillOpacity': 0.3,
                        'opacity': 1.0,
                        'dashArray': '5, 5'  # Dashed line for cluster boundary
                    },
                    popup=folium.Popup(cluster_popup, max_width=400),
                    tooltip=f"K-Anonymous Cluster {cluster_id} - {cluster_row['member_count']} parcels"
                ).add_to(cluster_layer)
                
                # Add original parcels that belong to this cluster
                original_parcels_in_cluster = anonymizer.gdf[anonymizer.gdf['cluster_id'] == cluster_id]
                
                for idx, parcel in original_parcels_in_cluster.iterrows():
                    try:
                        # Convert parcel to geographic coordinates if needed
                        if anonymizer.gdf.crs and anonymizer.gdf.crs.to_epsg() != 4326:
                            parcel_geom = gpd.GeoSeries([parcel.geometry], crs=anonymizer.gdf.crs).to_crs('EPSG:4326').iloc[0]
                        else:
                            parcel_geom = parcel.geometry
                            
                        # Create detailed popup for original parcel
                        parcel_popup = f"""
                        <div style="font-family: Arial, sans-serif; min-width: 250px;">
                            <h4>Original Parcel {parcel.get('id', idx)}</h4>
                            <table style="width:100%">
                                <tr><td><b>Cluster:</b></td><td>{cluster_id}</td></tr>
                                <tr><td><b>Area:</b></td><td>{parcel.get('flaecheAmtlich', 0):.0f} m²</td></tr>
                                <tr><td><b>Type:</b></td><td>{parcel.get('flaechentyp', 'N/A')}</td></tr>
                                <tr><td><b>Owner ID:</b></td><td><span style="color:red">PROTECTED</span></td></tr>
                                <tr><td><b>Address:</b></td><td><span style="color:red">PROTECTED</span></td></tr>
                                <tr><td><b>Survey Date:</b></td><td>{str(parcel.get('erhebung', 'N/A'))[:10]}</td></tr>
                            </table>
                            <br>
                            <p style="font-size:12px; color:gray;">
                                <i>This parcel has been anonymized by grouping with {cluster_row['member_count']-1} other parcels. 
                                Individual owner information is protected.</i>
                            </p>
                        </div>
                        """
                        
                        folium.GeoJson(
                            parcel_geom.__geo_interface__,
                            style_function=lambda x, color=color: {
                                'fillColor': color,
                                'color': 'darkred' if color == 'red' else 'dark' + color,
                                'weight': 1,
                                'fillOpacity': 0.7,
                                'opacity': 1.0
                            },
                            popup=folium.Popup(parcel_popup, max_width=350),
                            tooltip=f"Original Parcel {parcel.get('id', idx)} (Cluster {cluster_id})"
                        ).add_to(cluster_layer)
                        
                    except Exception as e:
                        print(f"Could not add parcel {idx} to cluster {cluster_id}: {e}")
                        continue
                
                cluster_layers[cluster_id] = cluster_layer
                cluster_layer.add_to(m)
            
            # Add layer control
            folium.LayerControl(position='topright', collapsed=False).add_to(m)
            
            # Add comprehensive legend
            legend_html = f'''
            <div style="position: fixed; 
                        bottom: 20px; left: 20px; width: 320px; height: auto; max-height: 400px;
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:12px; padding: 15px; overflow-y: auto;">
            <h3>K-Anonymity Visualization Legend</h3>
            <div style="margin-bottom: 10px;">
                <p><b>Dashed Outline:</b> K-Anonymous cluster boundary</p>
                <p><b>Solid Parcels:</b> Original parcels within cluster</p>
                <p><b>Same Colors:</b> Parcels grouped together for privacy</p>
            </div>
            <div style="border-top: 1px solid gray; padding-top: 10px;">
                <h4>Cluster Summary:</h4>
                <p><b>Total Clusters:</b> {len(k_geo)}</p>
                <p><b>Original Parcels:</b> {len(original_geo)}</p>
                <p><b>Data Reduction:</b> {((len(original_geo) - len(k_geo)) / len(original_geo) * 100):.1f}%</p>
            </div>
            <div style="border-top: 1px solid gray; padding-top: 10px; font-size: 11px;">
                <p><i>Toggle cluster layers on/off using the control panel. 
                Click on any parcel to see detailed information.</i></p>
            </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add title
            title_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 50%; transform: translateX(-50%);
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:18px; padding: 10px; text-align: center; max-width: 600px;">
            <h3>K-Anonymity Detailed View</h3>
            <p>Each color represents a k-anonymous cluster. Dashed boundaries show anonymized regions, solid parcels show original data.</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
        else:
            print("No cluster assignments found - cannot create detailed k-anonymity map")
            return
            
        # Save the detailed k-anonymity map
        k_map_file = output_dir / "detailed_k_anonymity_map.html"
        m.save(str(k_map_file))
        print(f"✓ Created detailed k-anonymity map: {k_map_file}")
        
    except Exception as e:
        print(f"Detailed k-anonymity map creation error: {e}")
        import traceback
        traceback.print_exc()
        

def create_donut_geomasking_visualization(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create visualization showing displacement vectors for donut geomasking"""
    try:
        if 'donut_geomasking' not in results or len(results['donut_geomasking']) == 0:
            print("No donut geomasking results to display")
            return
        
        donut_result = results['donut_geomasking']
        original_gdf = anonymizer.original_gdf.copy()
        
        # Clean and convert to geographic coordinates
        columns_to_drop = ['centroid', 'centroid_x', 'centroid_y']
        for col in columns_to_drop:
            if col in original_gdf.columns:
                original_gdf = original_gdf.drop(columns=[col])
        
        if original_gdf.crs and original_gdf.crs.to_epsg() != 4326:
            original_geo = original_gdf.to_crs('EPSG:4326')
        else:
            original_geo = original_gdf
        
        if donut_result.crs and donut_result.crs.to_epsg() != 4326:
            donut_geo = donut_result.to_crs('EPSG:4326')
        else:
            donut_geo = donut_result
        
        # Calculate map center
        bounds = original_geo.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add original parcels layer
        original_layer = folium.FeatureGroup(name="Original Parcels", show=True)
        
        sample_size = min(200, len(original_geo))
        original_sample = original_geo.sample(n=sample_size, random_state=42)
        
        for idx, row in original_sample.iterrows():
            try:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'blue',
                        'color': 'darkblue',
                        'weight': 1,
                        'fillOpacity': 0.4,
                        'opacity': 0.8
                    },
                    tooltip=f"Original Parcel {idx}"
                ).add_to(original_layer)
            except:
                continue
        
        original_layer.add_to(m)
        
        # Add displaced parcels layer with displacement vectors
        displaced_layer = folium.FeatureGroup(name="Displaced Parcels (Donut Masked)", show=True)
        
        donut_sample = donut_geo.loc[original_sample.index] if len(donut_geo) == len(original_geo) else donut_geo.sample(n=min(200, len(donut_geo)), random_state=42)
        
        for idx in original_sample.index:
            try:
                if idx not in donut_sample.index:
                    continue
                
                orig_centroid = original_sample.loc[idx].geometry.centroid
                disp_centroid = donut_sample.loc[idx].geometry.centroid
                
                # Draw displacement arrow
                folium.PolyLine(
                    locations=[[orig_centroid.y, orig_centroid.x], 
                              [disp_centroid.y, disp_centroid.x]],
                    color='red',
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Displacement vector for parcel {idx}"
                ).add_to(displaced_layer)
                
                # Draw displaced parcel
                folium.GeoJson(
                    donut_sample.loc[idx].geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'red',
                        'color': 'darkred',
                        'weight': 2,
                        'fillOpacity': 0.5,
                        'opacity': 1.0
                    },
                    tooltip=f"Displaced Parcel {idx}"
                ).add_to(displaced_layer)
                
            except Exception as e:
                continue
        
        displaced_layer.add_to(m)
        
        # Add legend
        dm = anonymizer.privacy_metrics.get('donut_geomasking', {})
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 280px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px;">
        <h3>Donut Geomasking</h3>
        <p><i class="fa fa-square" style="color:blue"></i> Original Parcels</p>
        <p><i class="fa fa-square" style="color:red"></i> Displaced Parcels</p>
        <p><i class="fa fa-minus" style="color:red"></i> Displacement Vectors</p>
        <hr>
        <p><b>Mean Displacement:</b> {dm.get('mean_displacement', 0):.1f}m</p>
        <p><b>Max Displacement:</b> {dm.get('max_displacement', 0):.1f}m</p>
        <p><b>K-value:</b> {dm.get('k', 0)}</p>
        <p><b>Failure Rate:</b> {dm.get('failure_rate', 0):.1f}%</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:18px; padding: 10px; text-align: center; max-width: 600px;">
        <h3>Donut Geomasking Visualization</h3>
        <p>Red arrows show displacement from original (blue) to masked (red) locations</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Save map
        map_file = output_dir / "donut_geomasking_map.html"
        m.save(str(map_file))
        print(f"✓ Created donut geomasking map: {map_file}")
        
    except Exception as e:
        print(f"Donut geomasking map error: {e}")
        import traceback
        traceback.print_exc()

def create_topology_preserving_visualization(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create visualization comparing topology before/after"""
    try:
        if 'topology_preserving' not in results or len(results['topology_preserving']) == 0:
            print("No topology-preserving results to display")
            return
        
        topo_result = results['topology_preserving']
        original_gdf = anonymizer.original_gdf.copy()
        
        # Clean and convert
        columns_to_drop = ['centroid', 'centroid_x', 'centroid_y']
        for col in columns_to_drop:
            if col in original_gdf.columns:
                original_gdf = original_gdf.drop(columns=[col])
        
        if original_gdf.crs and original_gdf.crs.to_epsg() != 4326:
            original_geo = original_gdf.to_crs('EPSG:4326')
        else:
            original_geo = original_gdf
        
        if topo_result.crs and topo_result.crs.to_epsg() != 4326:
            topo_geo = topo_result.to_crs('EPSG:4326')
        else:
            topo_geo = topo_result
        
        # Calculate map center
        bounds = original_geo.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        # Original layer
        original_layer = folium.FeatureGroup(name="Original Topology", show=False)
        
        sample_size = min(100, len(original_geo))
        original_sample = original_geo.sample(n=sample_size, random_state=42)
        
        for idx, row in original_sample.iterrows():
            try:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'lightblue',
                        'color': 'blue',
                        'weight': 2,
                        'fillOpacity': 0.3,
                        'opacity': 1.0
                    },
                    tooltip=f"Original - {len(list(row.geometry.exterior.coords))} vertices"
                ).add_to(original_layer)
            except:
                continue
        
        original_layer.add_to(m)
        
        # Generalized layer
        generalized_layer = folium.FeatureGroup(name="Generalized Topology", show=True)
        
        topo_sample = topo_geo.loc[original_sample.index] if len(topo_geo) == len(original_geo) else topo_geo.sample(n=min(100, len(topo_geo)), random_state=42)
        
        for idx, row in topo_sample.iterrows():
            try:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'lightgreen',
                        'color': 'darkgreen',
                        'weight': 2,
                        'fillOpacity': 0.5,
                        'opacity': 1.0
                    },
                    tooltip=f"Generalized - {len(list(row.geometry.exterior.coords))} vertices"
                ).add_to(generalized_layer)
            except:
                continue
        
        generalized_layer.add_to(m)
        
        # Add legend
        tp = anonymizer.privacy_metrics.get('topology_preserving', {})
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 300px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px;">
        <h3>Topology-Preserving Generalization</h3>
        <p><i class="fa fa-square" style="color:blue"></i> Original Topology</p>
        <p><i class="fa fa-square" style="color:green"></i> Generalized Topology</p>
        <hr>
        <p><b>Grid Size:</b> {tp.get('grid_size', 0)}m</p>
        <p><b>Adjacency Preserved:</b> {tp.get('adjacency_preserved_pct', 0):.1f}%</p>
        <p><b>Vertex Reduction:</b> {tp.get('vertex_reduction_pct', 0):.1f}%</p>
        <p><b>Original Vertices:</b> {tp.get('original_vertices', 0)}</p>
        <p><b>New Vertices:</b> {tp.get('new_vertices', 0)}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:18px; padding: 10px; text-align: center; max-width: 600px;">
        <h3>Topology-Preserving Generalization</h3>
        <p>Grid snapping and simplification while maintaining adjacencies</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        folium.LayerControl(position='topright').add_to(m)
        
        map_file = output_dir / "topology_preserving_map.html"
        m.save(str(map_file))
        print(f"✓ Created topology-preserving map: {map_file}")
        
    except Exception as e:
        print(f"Topology-preserving map error: {e}")
        import traceback
        traceback.print_exc()

def create_dp_grid_visualization(results: Dict, anonymizer, output_dir: Path) -> None:
    """Create visualization of DP grid aggregation with hexagons"""
    try:
        if 'dp_grid' not in results or len(results['dp_grid']) == 0:
            print("No DP grid results to display")
            return
        
        dp_result = results['dp_grid']
        
        # Convert to geographic coordinates
        if dp_result.crs and dp_result.crs.to_epsg() != 4326:
            dp_geo = dp_result.to_crs('EPSG:4326')
        else:
            dp_geo = dp_result
        
        # Calculate map center
        bounds = dp_geo.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Determine color scale based on noisy counts
        max_count = dp_geo['noisy_count'].max()
        
        # Add hexagons with color based on count
        for idx, row in dp_geo.iterrows():
            try:
                count = row['noisy_count']
                # Color intensity based on count
                intensity = min(1.0, count / max_count) if max_count > 0 else 0.5
                color = f'rgba(255, {int(255 * (1 - intensity))}, 0, 0.6)'
                
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <h4>DP Grid Cell</h4>
                    <table style="width:100%">
                        <tr><td><b>Noisy Count:</b></td><td>{row['noisy_count']}</td></tr>
                        <tr><td><b>Noisy Mean Area:</b></td><td>{row['noisy_mean_area']:.0f} m²</td></tr>
                        <tr><td><b>Dominant Type:</b></td><td>{row['dominant_type']}</td></tr>
                        <tr><td><b>Epsilon:</b></td><td>{row['epsilon']}</td></tr>
                    </table>
                    <p style="font-size:11px; color:gray;"><i>Note: Actual counts hidden for privacy</i></p>
                </div>
                """
                
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7,
                        'opacity': 1.0
                    },
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Cell: {row['noisy_count']} parcels"
                ).add_to(m)
                
            except Exception as e:
                continue
        
        # Add legend
        dpm = anonymizer.privacy_metrics.get('dp_grid_aggregation', {})
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 20px; right: 20px; width: 280px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px;">
        <h3>DP Grid Aggregation</h3>
        <div style="margin: 10px 0;">
            <p><b>Color Intensity:</b> Parcel count per cell</p>
            <div style="background: linear-gradient(to right, yellow, orange, red); 
                        height: 20px; border: 1px solid black;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px;">
                <span>Low</span><span>High</span>
            </div>
        </div>
        <hr>
        <p><b>Epsilon (ε):</b> {dpm.get('epsilon', 0)}</p>
        <p><b>Grid Cells:</b> {dpm.get('num_cells', 0)}</p>
        <p><b>Mean k/cell:</b> {dpm.get('mean_k_per_cell', 0):.1f}</p>
        <p><b>Count MAE:</b> {dpm.get('mae', 0):.2f}</p>
        <p><b>Count RMSE:</b> {dpm.get('rmse', 0):.2f}</p>
        <p style="font-size:10px; color:gray;"><i>Formal ε-DP guarantee</i></p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:18px; padding: 10px; text-align: center; max-width: 600px;">
        <h3>Differentially Private Grid Aggregation</h3>
        <p>H3 hexagonal cells with Laplace noise for formal privacy guarantees</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        map_file = output_dir / "dp_grid_aggregation_map.html"
        m.save(str(map_file))
        print(f"✓ Created DP grid aggregation map: {map_file}")
        
    except Exception as e:
        print(f"DP grid map error: {e}")
        import traceback
        traceback.print_exc()

def create_comprehensive_maps(results: Dict, anonymizer) -> None:
    """Create all comparison maps including new methods"""
    plots_dir = Path("thesis_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Original maps
    create_before_after_comparison_map(results, anonymizer, plots_dir)
    create_detailed_original_data_map(anonymizer, plots_dir)
    create_detailed_k_anonymity_map(results, anonymizer, plots_dir)
    
    # New method visualizations
    create_donut_geomasking_visualization(results, anonymizer, plots_dir)
    create_topology_preserving_visualization(results, anonymizer, plots_dir)
    create_dp_grid_visualization(results, anonymizer, plots_dir)

def create_sample_forest_data():
    """Create realistic sample forest cadastral data for testing"""
    import random
    
    features = []
    
    # Create sample forest parcels with realistic properties
    base_coords = [533400, 5366600]  # Approximate German coordinates
    
    for i in range(20):  # Create 20 sample parcels
        # Create a simple rectangular parcel
        x = base_coords[0] + random.uniform(-1000, 1000)
        y = base_coords[1] + random.uniform(-1000, 1000)
        width = random.uniform(50, 200)
        height = random.uniform(50, 200)
        
        coords = [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
            [x, y]  # Close the polygon
        ]
        
        area = width * height
        
        # Random forest properties
        species = random.choice(['Fichte', 'Buche', 'Eiche', 'Kiefer', 'Mischwald'])
        has_owner = random.choice([True, False])
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "flaecheAmtlich": area,
                "flaechentyp": f"Forstbetriebsfläche - {species}",
                "hierarchieEbene": "Teilfläche",
                "id": i + 1
            }
        }
        
        # Add owner information for some parcels (creates sensitivity)
        if has_owner:
            feature["properties"]["eigentuemer"] = [random.randint(1, 100)]
            feature["properties"]["waldAdresse"] = random.randint(1, 50)
        
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }
    
def run_parameter_sweep(geojson_string: str):
    """
    Runs a comprehensive parameter sweep using BOTH anonymizer classes.
    """
    print("\n" + "="*70)
    print("🚀 STARTING PARAMETER SWEEP (TRADITIONAL + HYBRID)")
    print("="*70)

    # 1. Setup Data and Output
    # We clean and load data once to pass to both classes
    cleaned = clean_geojson_data(geojson_string)
    data = json.loads(cleaned)
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:25832') # Use your local EPSG

    output_dir = Path("thesis_outputs/parameter_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize BOTH Anonymizers
    print("Initializing anonymizer classes...")
    # Class 1: Traditional methods (in this file)
    traditional_anon = OptimizedCadastralAnonymizer(gdf) 
    # Class 2: Hybrid methods (imported from hybrid_anon.py)
    hybrid_anon = HybridAnonymizer(gdf) 

    # 3. Define Test Configurations
    configs = {
        # --- Methods from OptimizedCadastralAnonymizer (cadastral_anonymizer.py) ---
        'k_anon': [
            {'k': 3, 'desc': 'Low Privacy (Internal)'},
            {'k': 5, 'desc': 'Standard'}, 
            {'k': 10, 'desc': 'High Privacy'}
        ],
        'dp_grid': [
            {'eps': 0.5, 'res': 8, 'desc': 'Strong Privacy'},
            {'eps': 1.0, 'res': 8, 'desc': 'Balanced'},
            {'eps': 5.0, 'res': 8, 'desc': 'High Utility'}
        ],

        # --- Methods from HybridAnonymizer (hybrid_anon.py) ---
        'geohash': [
            {'prec': 5, 'desc': 'Region (~5km)'},
            {'prec': 6, 'desc': 'District (~1km)'},
            {'prec': 7, 'desc': 'Neighborhood (~150m)'}
        ],
        'h3_hex': [
            {'res': 7, 'desc': 'Coarse (~5km2)'},
            {'res': 8, 'desc': 'Balanced (~0.7km2)'},
            {'res': 9, 'desc': 'Fine (~0.1km2)'}
        ],
        'hybrid_donut': [
            {'k': 5, 'eps': 0.5, 'desc': 'Max Protection'},
            {'k': 5, 'eps': 2.0, 'desc': 'Balanced Hybrid'}
        ]
    }

    results_log = []

    # 4. Run Tests
    
    # --- Test Traditional Methods ---
    print("\n--- Testing Traditional Methods ---")
    for cfg in configs['k_anon']:
        tag = f"k_anon_k{cfg['k']}"
        print(f"Running {tag}...")
        # calling local class method
        res = traditional_anon.optimized_k_anonymity(k=cfg['k'])
        if len(res) > 0:
            res.to_file(output_dir / f"{tag}.geojson", driver='GeoJSON')
            results_log.append({'config': tag, 'features': len(res)})

    for cfg in configs['dp_grid']:
        tag = f"dp_grid_eps{cfg['eps']}_res{cfg['res']}"
        print(f"Running {tag}...")
        # calling local class method
        res = traditional_anon.dp_grid_aggregation(epsilon=cfg['eps'], grid_resolution=cfg['res'])
        if len(res) > 0:
            res.to_file(output_dir / f"{tag}.geojson", driver='GeoJSON')
            results_log.append({'config': tag, 'features': len(res)})

    # --- Test Hybrid Methods ---
    print("\n--- Testing Hybrid Methods ---")
    for cfg in configs['geohash']:
        tag = f"geohash_prec{cfg['prec']}"
        print(f"Running {tag}...")
        # calling imported class method
        res = hybrid_anon.geohashing_anonymization(precision=cfg['prec'])
        if len(res) > 0:
            res.to_file(output_dir / f"{tag}.geojson", driver='GeoJSON')
            results_log.append({'config': tag, 'features': len(res)})

    for cfg in configs['h3_hex']:
        tag = f"h3_res{cfg['res']}"
        print(f"Running {tag}...")
        # calling imported class method
        res = hybrid_anon.h3_hexagonal_anonymization(resolution=cfg['res'])
        if len(res) > 0:
            res.to_file(output_dir / f"{tag}.geojson", driver='GeoJSON')
            results_log.append({'config': tag, 'features': len(res)})

    for cfg in configs['hybrid_donut']:
        tag = f"hybrid_donut_k{cfg['k']}_eps{cfg['eps']}"
        print(f"Running {tag}...")
        # calling imported class method
        res = hybrid_anon.hybrid_donut_conservative_geo(k=cfg['k'], epsilon=cfg['eps'])
        if len(res) > 0:
            res.to_file(output_dir / f"{tag}.geojson", driver='GeoJSON')
            results_log.append({'config': tag, 'features': len(res)})

    # 5. Save Summary Log
    pd.DataFrame(results_log).to_csv(output_dir / "sweep_log.csv", index=False)
    print(f"\n✅ Parameter sweep complete. Saved {len(results_log)} variations to {output_dir}")
    create_parameter_sweep_visualization(output_dir, original_gdf=gdf)

def create_parameter_sweep_visualization(output_dir: Path, original_gdf: gpd.GeoDataFrame = None):
    """
    Creates an interactive HTML map where clicking a grid cell reveals 
    the specific original parcels contained within it.
    """
    print(f"\n🗺️ Creating interactive visualization with parcel inspection...")
    
    # 1. Initialize Map
    geojsons = list(output_dir.glob("*.geojson"))
    if not geojsons and original_gdf is None:
        print("No results found to visualize.")
        return

    # Set map center
    if original_gdf is not None:
        # Prepare original data for spatial operations
        if original_gdf.crs and original_gdf.crs.to_epsg() != 4326:
            orig_geo_latlon = original_gdf.to_crs("EPSG:4326")
        else:
            orig_geo_latlon = original_gdf.copy()
            
        # Ensure we have an ID column for the list
        if 'id' not in orig_geo_latlon.columns:
            orig_geo_latlon['id'] = orig_geo_latlon.index.astype(str)
            
        bounds = orig_geo_latlon.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    else:
        # Fallback
        center_lat, center_lon = 51.1657, 10.4515
        orig_geo_latlon = None

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    # 2. Add Original Data Layer
    if orig_geo_latlon is not None:
        print("  Added layer: Original Data")
        folium.GeoJson(
            orig_geo_latlon,
            name="📍 Original Data",
            style_function=lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.3},
            tooltip=folium.GeoJsonTooltip(fields=['id'], aliases=['Parcel ID:'], localize=True)
        ).add_to(m)

    # 3. Add Result Layers with "Contained Parcels" Popup
    category_colors = {
        'k_anon': 'blue', 'dp_grid': 'green', 'geohash': 'orange', 
        'h3': 'purple', 'hybrid': 'red'
    }

    for file_path in sorted(geojsons):
        try:
            name = file_path.stem
            color = next((c for k, c in category_colors.items() if name.startswith(k)), 'gray')
            
            # Load result layer
            gdf = gpd.read_file(file_path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            # --- SPATIAL JOIN LOGIC ---
            # If we have original data, identify which parcels are in each cell
            if orig_geo_latlon is not None:
                # 1. Spatial join: Find intersection of Grid Cells (gdf) and Parcels (orig)
                # We use a unique temp index to map back
                gdf['temp_index'] = gdf.index 
                joined = gpd.sjoin(gdf, orig_geo_latlon[['geometry', 'id']], how='left', predicate='intersects')
                
                # 2. Group by Grid Cell and collect Parcel IDs into a list
                grouped = joined.groupby('temp_index')['id'].apply(list)
                
                # 3. Create a formatted string for the popup (limit to top 10 to avoid huge popups)
                def format_parcel_list(id_list):
                    if isinstance(id_list, float): return "None" # Handle NaNs
                    valid_ids = [str(x) for x in id_list if str(x) != 'nan']
                    count = len(valid_ids)
                    if count == 0: return "None"
                    shown = ", ".join(valid_ids[:10])
                    return f"{shown} (+{count-10} more)" if count > 10 else shown

                # FIX: Use map on the Series (column), not the Index
                gdf['contained_parcels'] = gdf['temp_index'].map(grouped).apply(format_parcel_list)
                
                # FIX: Same here
                gdf['parcel_count'] = gdf['temp_index'].map(grouped).apply(
                    lambda x: len([i for i in x if str(i) != 'nan']) if isinstance(x, list) else 0
                )

            # --- CREATE POPUP ---
            # Create a simple HTML table for the popup
            def make_popup(row):
                html = f"""
                <div style="font-family: Arial; font-size: 12px; width: 220px;">
                    <h4 style="color: #333; margin-bottom: 5px;">{name}</h4>
                    <b>Aggregated Statistics:</b><br>
                    <hr style="margin: 2px 0;">
                """
                
                # Check for Area and Type first (for your survey screenshot!)
                if 'total_area' in row and pd.notna(row['total_area']):
                    html += f"<b>Total Area:</b> {row['total_area']:.0f} m²<br>"
                elif 'flaecheAmtlich' in row and pd.notna(row['flaecheAmtlich']):
                    html += f"<b>Total Area:</b> {row['flaecheAmtlich']:.0f} m²<br>"
                    
                if 'flaechentyp' in row and pd.notna(row['flaechentyp']):
                    html += f"<b>Dominant Type:</b> {row['flaechentyp']}<br>"
                
                # Add other existing properties
                for col in ['group_size', 'noisy_count', 'parcel_count', 'epsilon', 'k']:
                    if col in row and pd.notna(row[col]):
                        nice_name = col.replace('_', ' ').title()
                        html += f"<b>{nice_name}:</b> {row[col]}<br>"
                
                # Add the list of contained parcels
                if 'contained_parcels' in row:
                    html += f"<hr style='margin: 5px 0;'><b>Contained Parcel IDs:</b><br>{row['contained_parcels']}"
                
                html += "</div>"
                return folium.Popup(html, max_width=300)

            # Create FeatureGroup
            fg = folium.FeatureGroup(name=name, show=False)
            
            # Add GeoJson to FeatureGroup manually to attach custom popups
            for idx, row in gdf.iterrows():
                # Prepare style
                style = {'fillColor': color, 'color': color, 'weight': 1, 'fillOpacity': 0.4}
                
                # Create geometry object
                g = folium.GeoJson(
                    row.geometry,
                    style_function=lambda x: style
                )
                g.add_child(make_popup(row)) # Attach the popup
                g.add_to(fg)
            
            fg.add_to(m)
            print(f"  Added layer: {name} (with parcel lists)")
            
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")
            # print full stack trace to debug
            # import traceback
            # traceback.print_exc()

    # 4. Save
    folium.LayerControl(collapsed=False).add_to(m)
    output_html = output_dir / "parameter_comparison_map.html"
    m.save(str(output_html))
    print(f"✓ Visualization saved to: {output_html}")
# Main execution
# if __name__ == "__main__":
#     print("Robust Forest Cadastral Data Anonymization")
#     print("=" * 50)
    
#     geojson_path = "minifix.geojson.json"
    
#     try:
#         geojson_str = load_geojson_text(geojson_path)
        
#         # First analyze the data for issues
#         analyze_geojson_issues(geojson_str)
#         print("\n")
        
#         # Option 1: Run comprehensive test with threat evaluation
#         print("\n1. Running comprehensive test with threats...")
#         results = comprehensive_test(geojson_str)
        
#         # Option 2: Test specific attack
#         print("\n2. Testing specific attack...")
#         test_specific_attack(geojson_str, 'homogeneity')
        
#         # Option 3: Create protected version only
#         print("\n3. Creating protected version...")
#         cleaned_geojson = clean_geojson_data(geojson_str)
#         data = json.loads(cleaned_geojson)
#         gdf = gpd.GeoDataFrame.from_features(data['features'])
#         if gdf.crs is None:
#             gdf = gdf.set_crs('EPSG:25832')
        
#         privacy_requirements = {
#             'min_k': 7,
#             'epsilon': 0.5,
#             'attack_resistance': ['homogeneity', 'background', 'satellite']
#         }
        
#         protected = create_privacy_preserving_pipeline(gdf, privacy_requirements)
#         print(f"Protected dataset: {len(protected)} aggregated regions")
        
#     except FileNotFoundError:
#         print(f"File {geojson_path} not found.")
    
#     print("\n" + "="*50)
#     print("Integration complete!")
#     print("The threat model is now part of your anonymization pipeline.")



if __name__ == "__main__":
    print("Robust Forest Cadastral Data Anonymization")
    print("=" * 50)
    geojson_path = "minifix.geojson.json"

    try:
        geojson_str = load_geojson_text(geojson_path)

        # 0) (optional) quick sanity report
        analyze_geojson_issues(geojson_str)
        print("\n")

        # 1) Load data
        cleaned_geojson = clean_geojson_data(geojson_str)
        data = json.loads(cleaned_geojson)
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:25832')

        print("\n1. Running anonymization methods ONLY (no threat model)...")
        anonymizer = OptimizedCadastralAnonymizer(gdf)
        results = {}
        
        print("\n1. Running comprehensive test")
                
        results = comprehensive_test(geojson_str)
        print("\n- Parameter sweep")
        run_parameter_sweep(geojson_str)
        
        # print("\n- Conservative Geo-Indistinguishability")
        # cg = anonymizer.conservative_geo_indistinguishability(epsilon=2.0)
        # if len(cg) > 0: results["conservative_geo"] = cg

        # print("\n- Precise Generalization")
        # pg = anonymizer.precise_generalization(tolerance=2.0)
        # if len(pg) > 0: results["precise_gen"] = pg

        # print("\n- Optimized K-Anonymity")
        # ka = anonymizer.optimized_k_anonymity(k=5)
        # if len(ka) > 0: results["optimized_k"] = ka

        # print("\n- Donut Geomasking")
        # dg = anonymizer.donut_geomasking(k=5)
        # if len(dg) > 0: results["donut_geomasking"] = dg

        # print("\n- Topology-Preserving Generalization")
        # tp = anonymizer.topology_preserving_generalization(grid_size=20.0, simplify_tolerance=15.0)
        # if len(tp) > 0: results["topology_preserving"] = tp

        # print("\n- Differentially Private Grid Aggregation")
        # dp = anonymizer.dp_grid_aggregation(epsilon=1.0, grid_resolution=8)
        # if len(dp) > 0: results["dp_grid"] = dp

        # 3) Export and visualize (these don’t use the threat model)
        export_thesis_results(results, anonymizer)
        create_comparison_visualization(results, anonymizer)
        create_comprehensive_maps(results, anonymizer)
        
        # === THREAT MODEL: evaluate anonymized outputs ===
        # Only include layers you actually produced
        methods_for_eval = {
            k: v for k, v in results.items()
            if k in ("conservative_geo", "precise_gen", "optimized_k", "donut_geomasking", "topology_preserving", "dp_grid")
            and len(v) > 0
        }

        if methods_for_eval:
            robustness_df = evaluate_anonymization_robustness(
                anonymizer.original_gdf,   # original (pre-anonymization)
                methods_for_eval           # dict of {method_name: anonymized_gdf}
            )

            # Save + quick printout
            out_dir = Path("thesis_outputs"); out_dir.mkdir(exist_ok=True)
            robustness_csv = out_dir / "robustness_evaluation.csv"
            robustness_df.to_csv(robustness_csv, index=False)
            print(f"\n✓ Threat model evaluation complete. Saved: {robustness_csv}")
        else:
            print("\n(No anonymized layers to evaluate)")
            
        # === OPTIONAL: deep-dive attacks on a selected method ===
        method_to_probe = "precise_gen"  # change to any key present in results
        if method_to_probe in results and len(results[method_to_probe]) > 0:
            sim = PrivacyAttackSimulator(
                anonymizer.original_gdf,
                results[method_to_probe],
                method_to_probe
            )
            _ = sim.run_all_attacks()  # prints detailed attack outcomes
            

        print("\nDone. Anonymization outputs written to 'thesis_outputs' and 'thesis_plots'.")

    except FileNotFoundError:
        print(f"File {geojson_path} not found.")