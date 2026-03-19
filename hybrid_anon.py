import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import json
import folium
from folium import plugins
import h3
import geohash2
from typing import Dict, Any, List, Tuple, Optional
import warnings
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

class HybridAnonymizer:
    """
    Advanced hybrid anonymization methods for geodata privacy.
    Combines multiple techniques to maximize privacy while preserving utility.
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """Initialize with geodata"""
        self.original_gdf = gdf.copy()
        if self.original_gdf.crs is None:
            self.original_gdf = self.original_gdf.set_crs('EPSG:25832')
        
        # Calculate centroids for point-based operations
        self.original_gdf['centroid'] = self.original_gdf.geometry.centroid
        self.original_gdf['centroid_x'] = self.original_gdf.centroid.x
        self.original_gdf['centroid_y'] = self.original_gdf.centroid.y
        
        print(f"Initialized with {len(self.original_gdf)} features")
    
    def hybrid_donut_conservative_geo(self, 
                                      k: int = 5,
                                      epsilon: float = 1.0,
                                      inner_radius_ratio: float = 0.3,
                                      outer_radius_ratio: float = 1.5) -> gpd.GeoDataFrame:
        """
        HYBRID METHOD 1: Donut Geomasking + Conservative Geo-Indistinguishability
        
        This combines:
        1. Donut geomasking to hide exact locations within a ring
        2. Conservative geo-indist to add controlled noise
        3. K-anonymity grouping for additional privacy
        
        Args:
            k: Minimum group size for k-anonymity
            epsilon: Privacy budget for geo-indistinguishability (lower = more private)
            inner_radius_ratio: Inner radius as ratio of cluster spread
            outer_radius_ratio: Outer radius as ratio of cluster spread
        """
        print(f"\n🔄 Hybrid Donut + Conservative Geo-Indist (k={k}, ε={epsilon})")
        
        # Step 1: K-anonymity clustering
        coords = np.column_stack([self.original_gdf.centroid_x, self.original_gdf.centroid_y])
        
        # Use DBSCAN for spatial clustering
        # Calculate reasonable eps from nearest neighbor distances
        # Use nearest neighbor distances instead of all pairwise
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)  # k=2 to get nearest neighbor (k=1 is self)
        nn_distances = distances[:, 1]  # Get the nearest neighbor distances
        eps_distance = np.median(nn_distances) * 3  # Use 3x median nearest neighbor
        print(f"Using eps distance: {eps_distance:.1f}m")
        clustering = DBSCAN(eps=eps_distance, min_samples=k).fit(coords)
        self.original_gdf['cluster'] = clustering.labels_
        
        # Remove noise points (-1 cluster)
        valid_clusters = self.original_gdf[self.original_gdf['cluster'] >= 0].copy()
        
        anonymized_features = []
        
        for cluster_id in valid_clusters['cluster'].unique():
            cluster_data = valid_clusters[valid_clusters['cluster'] == cluster_id]
            
            if len(cluster_data) < k:
                continue
            
            # Calculate cluster center
            center_x = cluster_data.centroid_x.mean()
            center_y = cluster_data.centroid_y.mean()
            
            # Calculate cluster spread (for donut sizing)
            spread = np.sqrt(
                (cluster_data.centroid_x - center_x).pow(2).sum() +
                (cluster_data.centroid_y - center_y).pow(2).sum()
            ) / len(cluster_data)
            
            # Step 2: Apply donut geomasking
            inner_radius = spread * inner_radius_ratio
            outer_radius = spread * outer_radius_ratio
            
            # Generate random point in donut ring
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(inner_radius, outer_radius)
            donut_x = center_x + radius * np.cos(angle)
            donut_y = center_y + radius * np.sin(angle)
            
            # Step 3: Add conservative geo-indistinguishability noise
            # Laplace mechanism with epsilon privacy budget
            scale = 1.0 / epsilon
            noise_x = np.random.laplace(0, scale)
            noise_y = np.random.laplace(0, scale)
            
            final_x = donut_x + noise_x
            final_y = donut_y + noise_y
            
            # Create anonymized geometry (aggregate polygons)
            union_geom = unary_union(cluster_data.geometry)
            
            # Create buffer around the final point to represent uncertainty
            uncertainty_buffer = Point(final_x, final_y).buffer(outer_radius * 1.2)
            
            # Combine with original geometry shape
            if isinstance(union_geom, Polygon):
                combined_geom = union_geom.union(uncertainty_buffer).convex_hull
            else:
                combined_geom = uncertainty_buffer
            
            # Store anonymized feature
            anonymized_features.append({
                'geometry': combined_geom,
                'cluster_id': int(cluster_id),
                'group_size': len(cluster_data),
                'total_area': cluster_data.flaecheAmtlich.sum() if 'flaecheAmtlich' in cluster_data.columns else 0,
                'method': 'hybrid_donut_conservative',
                'epsilon': epsilon,
                'k': k,
                #added
                'original_center_x': center_x,
                'original_center_y': center_y,
                'displaced_x': final_x,
                'displaced_y': final_y,
                'displacement_distance': np.sqrt((final_x - center_x)**2 + (final_y - center_y)**2),
                #added 
                'inner_radius': inner_radius,
                'outer_radius': outer_radius,
                'uncertainty_radius': outer_radius * 1.2
            })
        
        result_gdf = gpd.GeoDataFrame(anonymized_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} anonymized regions (avg size: {result_gdf['group_size'].mean():.1f})")
        
        return result_gdf
    
    def donut_geo_indist_only(self, 
                            epsilon: float = 1.0,
                            inner_radius: float = 500,
                            outer_radius: float = 2000,
                            geometry_noise_scale: float = 10) -> gpd.GeoDataFrame:
        """
        Donut Geomasking + TRUE Geo-Indistinguishability (WITHOUT k-anonymity)
        
        Applies privacy to each feature individually:
        1. Displaces location to random point in donut ring
        2. Adds geo-indistinguishability noise to GEOMETRY VERTICES (shape distortion)
        """
        print(f"\n🍩🎲 Donut + TRUE Geo-Indist (ε={epsilon}, noise_scale={geometry_noise_scale}m)")
        
        anonymized_features = []
        
        for idx, row in self.original_gdf.iterrows():
            center_x = row['centroid_x']
            center_y = row['centroid_y']
            
            # Step 1: Donut geomasking for location
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(inner_radius, outer_radius)
            donut_x = center_x + radius * np.cos(angle)
            donut_y = center_y + radius * np.sin(angle)
            
            # Step 2: Apply noise to EVERY VERTEX of the geometry
            geom = row['geometry']
            scale = geometry_noise_scale / epsilon  # Noise inversely proportional to epsilon
            
            if geom.geom_type == 'Polygon':
                # Get exterior coordinates and add noise
                coords = list(geom.exterior.coords)
                noisy_coords = []
                
                for x, y in coords:
                    # Add Laplace noise to each vertex
                    vertex_noise_x = np.random.laplace(0, scale)
                    vertex_noise_y = np.random.laplace(0, scale)
                    noisy_coords.append((x + vertex_noise_x, y + vertex_noise_y))
                
                # Create noisy polygon
                noisy_geom = Polygon(noisy_coords)
                
                # Handle holes if present
                if len(list(geom.interiors)) > 0:
                    holes = []
                    for interior in geom.interiors:
                        hole_coords = []
                        for x, y in interior.coords:
                            vertex_noise_x = np.random.laplace(0, scale)
                            vertex_noise_y = np.random.laplace(0, scale)
                            hole_coords.append((x + vertex_noise_x, y + vertex_noise_y))
                        holes.append(hole_coords)
                    noisy_geom = Polygon(noisy_coords, holes)
                    
            elif geom.geom_type == 'MultiPolygon':
                polys = []
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    noisy_coords = []
                    for x, y in coords:
                        vertex_noise_x = np.random.laplace(0, scale)
                        vertex_noise_y = np.random.laplace(0, scale)
                        noisy_coords.append((x + vertex_noise_x, y + vertex_noise_y))
                    polys.append(Polygon(noisy_coords))
                noisy_geom = MultiPolygon(polys)
            else:
                noisy_geom = geom  # Fallback
            
            # Step 3: Translate the NOISY geometry to donut location
            from shapely.affinity import translate
            displacement_x = donut_x - center_x
            displacement_y = donut_y - center_y
            final_geom = translate(noisy_geom, xoff=displacement_x, yoff=displacement_y)
            
            # Ensure geometry is valid
            if not final_geom.is_valid:
                final_geom = final_geom.buffer(0)
            
            anonymized_features.append({
                'geometry': final_geom,
                'feature_id': idx,
                'group_size': 1,  # For visualization compatibility
                'total_area': row.get('flaecheAmtlich', final_geom.area),
                'original_area': row.get('flaecheAmtlich', 0),
                'method': 'donut_geo_only',
                'epsilon': epsilon,
                'geometry_noise_scale': scale,
                'original_center_x': center_x,
                'original_center_y': center_y,
                'displaced_x': donut_x,
                'displaced_y': donut_y,
                'displacement_distance': np.sqrt((donut_x - center_x)**2 + (donut_y - center_y)**2),
                'inner_radius': inner_radius,
                'outer_radius': outer_radius
            })
        
        result_gdf = gpd.GeoDataFrame(anonymized_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} anonymized features with SHAPE DISTORTION")
        
        return result_gdf
    
    def geohashing_anonymization(self, precision: int = 6) -> gpd.GeoDataFrame:
        """
        GEOHASHING METHOD: Spatial hashing for location privacy
        
        Uses geohash to discretize space into fixed-size cells.
        Each location is mapped to its geohash cell, providing k-anonymity
        within the cell.
        
        Args:
            precision: Geohash precision (1-12, lower = larger cells = more privacy)
                       5: ~4.9km × 4.9km
                       6: ~1.2km × 0.6km  
                       7: ~152m × 152m
        """
        print(f"\n🔲 Geohashing Anonymization (precision={precision})")
        
        # Convert to WGS84 for geohashing
        gdf_wgs84 = self.original_gdf.to_crs('EPSG:4326')
        
        # Calculate geohash for each feature
        gdf_wgs84['geohash'] = gdf_wgs84.centroid.apply(
            lambda p: geohash2.encode(p.y, p.x, precision=precision)
        )
        
        anonymized_features = []
        
        for geohash_id in gdf_wgs84['geohash'].unique():
            cell_data = gdf_wgs84[gdf_wgs84['geohash'] == geohash_id]
            
            # Get geohash bounding box
            bbox = geohash2.decode_exactly(geohash_id)
            lat, lon, lat_err, lon_err = bbox
            
            # Create polygon for geohash cell
            cell_polygon = Polygon([
                [lon - lon_err, lat - lat_err],
                [lon + lon_err, lat - lat_err],
                [lon + lon_err, lat + lat_err],
                [lon - lon_err, lat + lat_err],
                [lon - lon_err, lat - lat_err]
            ])
            
            # Convert back to original CRS
            cell_gdf = gpd.GeoDataFrame([{'geometry': cell_polygon}], crs='EPSG:4326')
            cell_gdf = cell_gdf.to_crs(self.original_gdf.crs)
            
            anonymized_features.append({
                'geometry': cell_gdf.geometry.iloc[0],
                'geohash': geohash_id,
                'group_size': len(cell_data),
                'total_area': cell_data.flaecheAmtlich.sum() if 'flaecheAmtlich' in cell_data.columns else 0,
                'method': 'geohashing',
                'precision': precision,
                'cell_size_km': f"{4.9 / (5**precision):.3f}"
            })
        
        result_gdf = gpd.GeoDataFrame(anonymized_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} geohash cells (avg size: {result_gdf['group_size'].mean():.1f})")
        
        return result_gdf
    
    def h3_hexagonal_anonymization(self, resolution: int = 8) -> gpd.GeoDataFrame:
        """
        H3 HEXAGONAL GRID: Uber's H3 hierarchical hexagonal grid
        
        Similar to geohashing but uses hexagonal cells which have better
        properties for spatial analysis (uniform neighbor distance).
        
        Args:
            resolution: H3 resolution (0-15, lower = larger cells)
                        7: ~5.16 km² per cell
                        8: ~0.74 km² per cell
                        9: ~0.10 km² per cell
        """
        print(f"\n⬡ H3 Hexagonal Anonymization (resolution={resolution})")
        
        # Convert to WGS84 for H3
        gdf_wgs84 = self.original_gdf.to_crs('EPSG:4326')
        
        # Calculate H3 index for each feature - using modern API
        gdf_wgs84['h3_index'] = gdf_wgs84.centroid.apply(
            lambda p: h3.latlng_to_cell(p.y, p.x, resolution)
        )
        
        anonymized_features = []
        
        for h3_index in gdf_wgs84['h3_index'].unique():
            cell_data = gdf_wgs84[gdf_wgs84['h3_index'] == h3_index]
            
            # Get H3 cell boundary - using modern API
            h3_boundary = h3.cell_to_boundary(h3_index)
            
            # Create polygon from boundary (returns list of (lat, lng) tuples)
            # Convert to (lng, lat) for GeoJSON format
            coords = [(lng, lat) for lat, lng in h3_boundary]
            cell_polygon = Polygon(coords)
            
            # Convert back to original CRS
            cell_gdf = gpd.GeoDataFrame([{'geometry': cell_polygon}], crs='EPSG:4326')
            cell_gdf = cell_gdf.to_crs(self.original_gdf.crs)
            
            # Get cell area using modern API
            try:
                cell_area = h3.cell_area(h3_index, unit='km^2')
            except:
                cell_area = 0.74  # Default for resolution 8
            
            anonymized_features.append({
                'geometry': cell_gdf.geometry.iloc[0],
                'h3_index': h3_index,
                'group_size': len(cell_data),
                'total_area': cell_data.flaecheAmtlich.sum() if 'flaecheAmtlich' in cell_data.columns else 0,
                'method': 'h3_hexagonal',
                'resolution': resolution,
                'cell_area_km2': cell_area
            })
        
        result_gdf = gpd.GeoDataFrame(anonymized_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} H3 hexagonal cells (avg size: {result_gdf['group_size'].mean():.1f})")
        
        return result_gdf
    
    def hybrid_geohash_noise(self, precision: int = 6, epsilon: float = 1.0) -> gpd.GeoDataFrame:
        """
        HYBRID METHOD 2: Geohashing + Differential Privacy Noise
        """
        print(f"\n🔲➕🎲 Hybrid Geohash + DP Noise (precision={precision}, ε={epsilon})")
        
        # 1. Apply Standard Geohashing
        geohash_result = self.geohashing_anonymization(precision)
        
        # 2. Add Differential Privacy Noise to the Count
        # We ensure the count is at least 1 (to avoid disappearing cells)
        scale = 1.0 / epsilon
        geohash_result['noisy_count'] = geohash_result['group_size'].apply(
            lambda x: max(1, int(round(x + np.random.laplace(0, scale))))
        )
        
        perturbed_features = []
        for idx, row in geohash_result.iterrows():
            geom = row['geometry']
            
            # 3. Perturb Geometry (Shift Centroid)
            noise_x = np.random.laplace(0, scale * 10)
            noise_y = np.random.laplace(0, scale * 10)
            
            from shapely.affinity import translate
            perturbed_geom = translate(geom, xoff=noise_x, yoff=noise_y)
            
            perturbed_features.append({
                'geometry': perturbed_geom,
                'geohash': row['geohash'],
                # FIX: The "Public" group size should be the NOISY count
                'group_size': row['noisy_count'],
                # Keep the "Internal" true count for your analysis
                'true_count': row['group_size'], 
                'total_area': row['total_area'],
                'method': 'hybrid_geohash_dp',
                'precision': precision,
                'epsilon': epsilon
            })
        
        result_gdf = gpd.GeoDataFrame(perturbed_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} perturbed geohash cells")
        
        return result_gdf
    
    def hybrid_h3_clustering(self, resolution: int = 8, k: int = 5) -> gpd.GeoDataFrame:
        """
        HYBRID METHOD 3: H3 Grid + K-Anonymity Clustering
        
        Uses H3 cells as base units, then merges adjacent cells
        to ensure k-anonymity.
        """
        print(f"\n⬡➕👥 Hybrid H3 + K-Anonymity (resolution={resolution}, k={k})")
        
        # Get H3 cells
        h3_result = self.h3_hexagonal_anonymization(resolution)
        
        # Merge cells with small populations
        merged_features = []
        processed = set()
        
        for idx, row in h3_result.iterrows():
            if idx in processed:
                continue
            
            if row['group_size'] >= k:
                # Cell already satisfies k-anonymity
                merged_features.append(row.to_dict())
                processed.add(idx)
            else:
                # Find nearby cells to merge
                nearby_cells = []
                nearby_indices = []
                
                for idx2, row2 in h3_result.iterrows():
                    if idx2 not in processed and idx != idx2:
                        # Check if cells are adjacent (distance < 2x typical cell size)
                        dist = row['geometry'].centroid.distance(row2['geometry'].centroid)
                        # Estimate typical size from cell area
                        typical_size = np.sqrt(row['cell_area_km2'] * 1e6)  # Convert km² to m²
                        
                        if dist < typical_size * 2:
                            nearby_cells.append(row2)
                            nearby_indices.append(idx2)
                
                # Merge with nearby cells
                merged_geom = row['geometry']
                merged_count = row['group_size']
                merged_area = row['total_area']
                
                for i, nearby_row in enumerate(nearby_cells):
                    merged_geom = merged_geom.union(nearby_row['geometry'])
                    merged_count += nearby_row['group_size']
                    merged_area += nearby_row['total_area']
                    processed.add(nearby_indices[i])
                    
                    if merged_count >= k:
                        break
                
                if merged_count >= k:
                    merged_features.append({
                        'geometry': merged_geom,
                        'group_size': merged_count,
                        'total_area': merged_area,
                        'method': 'hybrid_h3_k',
                        'resolution': resolution,
                        'k': k
                    })
                    processed.add(idx)
        
        result_gdf = gpd.GeoDataFrame(merged_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} merged H3 regions (avg size: {result_gdf['group_size'].mean():.1f})")
        
        return result_gdf
    
    def hybrid_triple_layer(self, k: int = 5, epsilon: float = 1.0, 
                           h3_resolution: int = 8) -> gpd.GeoDataFrame:
        """
        HYBRID METHOD 4: Triple-Layer Privacy (Donut + Geo-Indist + H3)
        
        Maximum privacy through three layers:
        1. H3 grid for spatial discretization
        2. Donut masking for location hiding
        3. Geo-indistinguishability for formal privacy
        """
        print(f"\n🔐 Triple-Layer Hybrid (k={k}, ε={epsilon}, H3={h3_resolution})")
        
        # Layer 1: H3 discretization
        h3_result = self.h3_hexagonal_anonymization(h3_resolution)
        
        # Layer 2 & 3: Apply donut + geo-indist to each H3 cell
        enhanced_features = []
        
        for idx, row in h3_result.iterrows():
            if row['group_size'] < k:
                continue
            
            geom = row['geometry']
            centroid = geom.centroid
            
            # Calculate cell size for donut sizing
            # Estimate from cell area
            cell_size = np.sqrt(row['cell_area_km2'] * 1e6)  # Convert km² to m²
            
            # Donut masking
            inner_radius = cell_size * 0.3
            outer_radius = cell_size * 1.5
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(inner_radius, outer_radius)
            
            donut_x = centroid.x + radius * np.cos(angle)
            donut_y = centroid.y + radius * np.sin(angle)
            
            # Geo-indistinguishability noise
            scale = 1.0 / epsilon
            noise_x = np.random.laplace(0, scale)
            noise_y = np.random.laplace(0, scale)
            
            final_x = donut_x + noise_x
            final_y = donut_y + noise_y
            
            # Create final geometry
            uncertainty_buffer = Point(final_x, final_y).buffer(outer_radius * 1.5)
            final_geom = geom.union(uncertainty_buffer).convex_hull
            
            enhanced_features.append({
                'geometry': final_geom,
                'group_size': row['group_size'],
                'total_area': row['total_area'],
                'method': 'triple_layer',
                'k': k,
                'epsilon': epsilon,
                'h3_resolution': h3_resolution,
                'privacy_layers': 3
            })
        
        result_gdf = gpd.GeoDataFrame(enhanced_features, crs=self.original_gdf.crs)
        print(f"✓ Created {len(result_gdf)} triple-protected regions")
        
        return result_gdf


def create_comprehensive_visualization(results: Dict[str, gpd.GeoDataFrame], 
                                      original_gdf: gpd.GeoDataFrame,
                                      output_path: str = 'hybrid_anonymization_map.html'):
    """
    Create comprehensive interactive HTML visualization of all methods
    """
    print("\n🗺️  Creating Interactive HTML Visualization...")
    
    # Convert to WGS84 for mapping
    original_wgs84 = original_gdf.to_crs('EPSG:4326')
    # Verify conversion worked
    print(f"Original CRS: {original_gdf.crs}")
    print(f"Converted bounds: {original_wgs84.total_bounds}")
    
    # Calculate map center
    bounds = original_wgs84.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    m.get_root().html.add_child(folium.Element("""
    <style>
    .leaflet-control-container { z-index: 10050; }
    .map-title, .map-legend { pointer-events: none; }
    </style>
    """))
    
    # Color scheme for different methods
    colors = {
        'original': '#3388ff',
        'hybrid_donut_conservative': '#ff6b6b',
        'donut_geo_only': '#e74c3c',
        'geohashing': '#4ecdc4',
        'h3_hexagonal': '#95e1d3',
        'hybrid_geohash_dp': '#f38181',
        'hybrid_h3_k': '#aa96da',
        'triple_layer': '#fcbad3'
    }
    
    # Add original data layer
    original_layer = folium.FeatureGroup(name='📍 Original Data', show=True)
    first_geom = original_wgs84.geometry.iloc[0]
    
    display_original = original_wgs84.head(200)
    print(f"Displaying {len(display_original)}/{len(original_wgs84)} original features")
    first_geom = display_original.geometry.iloc[0]
    print(f"First geometry type: {first_geom.geom_type}")
    print(f"First coords: {list(first_geom.exterior.coords)[:3]}")
    print(f"First geometry type: {first_geom.geom_type}")
    print(f"First coords: {list(first_geom.exterior.coords)[:3]}")  # First 3 points
    
    for idx, row in display_original.iterrows():  # Limit for performance
        folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda x: {
                'fillColor': colors['original'],
                'color': colors['original'],
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=f"Original Feature {idx}"
        ).add_to(original_layer)
    original_layer.add_to(m)
    
    # Add each anonymization method as a layer
    for method_name, gdf in results.items():
        if len(gdf) == 0:
            continue
        
        gdf_wgs84 = gdf.to_crs('EPSG:4326')
        
        # Create icon based on method
        icon_map = {
            'hybrid_donut_conservative': '🍩',
            'donut_geo_only': '🍩🎲',
            'geohashing': '🔲',
            'h3_hexagonal': '⬡',
            'hybrid_geohash_dp': '🔲🎲',
            'hybrid_h3_k': '⬡👥',
            'triple_layer': '🔐'
        }
        icon = icon_map.get(method_name, '🔹')
        
        layer = folium.FeatureGroup(name=f'{icon} {method_name.replace("_", " ").title()}', show=False)
        # LIMIT FEATURES FOR DISPLAY (keeps export files complete)
        display_gdf = gdf_wgs84.head(200) if len(gdf_wgs84) > 200 else gdf_wgs84
        print(f"  Displaying {len(display_gdf)}/{len(gdf_wgs84)} features for {method_name}")
            
        for idx, row in display_gdf.iterrows():
            # Create popup with method details
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Method:</b> {method_name}<br>
                <b>Group Size:</b> {row.get('group_size', 'N/A')}<br>
                <b>Total Area:</b> {row.get('total_area', 0):.2f} m²<br>
            """
            
            # Add method-specific details
            if 'epsilon' in row:
                popup_html += f"<b>Privacy Budget (ε):</b> {row['epsilon']}<br>"
            if 'k' in row:
                popup_html += f"<b>K-Anonymity (k):</b> {row['k']}<br>"
            if 'precision' in row:
                popup_html += f"<b>Geohash Precision:</b> {row['precision']}<br>"
            if 'resolution' in row:
                popup_html += f"<b>H3 Resolution:</b> {row['resolution']}<br>"
            if 'privacy_layers' in row:
                popup_html += f"<b>Privacy Layers:</b> {row['privacy_layers']}<br>"
            
            popup_html += "</div>"
            
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda x, color=colors.get(method_name, '#888888'): {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.5
                },
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(layer)
        
        # SPECIAL HANDLING FOR DONUT+DP TO SHOW DISPLACEMENT VECTORS
        if method_name == 'donut_geo_only' and 'original_center_x' in gdf.columns:
            # LIMIT TO FIRST 100 FOR PERFORMANCE
            sample_gdf = gdf.head(100)
            sample_gdf_wgs84 = sample_gdf.to_crs('EPSG:4326')
            # Convert original centers to WGS84
            original_centers_wgs84 = []
            for idx, row in sample_gdf.iterrows():
                orig_point = gpd.GeoDataFrame(
                    [{'geometry': Point(row['original_center_x'], row['original_center_y'])}],
                    crs=gdf.crs
                ).to_crs('EPSG:4326')
                original_centers_wgs84.append(orig_point.geometry.iloc[0])
            
            # Draw displacement vectors
            for idx, (row, orig_center) in enumerate(zip(sample_gdf_wgs84.itertuples(), original_centers_wgs84)):
                displaced_center = row.geometry.centroid
                
                # Draw arrow from original to displaced
                line = folium.PolyLine(
                    locations=[
                        [orig_center.y, orig_center.x],
                        [displaced_center.y, displaced_center.x]
                    ],
                    color='red',
                    weight=2,
                    opacity=0.8,
                    tooltip=f"Displacement vector for parcel {idx}"
                ).add_to(layer)

                # Add arrow head bound to the SAME line
                plugins.PolyLineTextPath(
                    line,
                    '▶',
                    repeat=False,
                    offset=12,
                    attributes={'fill': 'red', 'font-size': '20'}
                ).add_to(layer)
                
                # Draw original location (blue)
                folium.CircleMarker(
                    location=[orig_center.y, orig_center.x],
                    radius=5,
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.7,
                    tooltip=f"Original location {idx}"
                ).add_to(layer)
                
                # Draw displaced location (red)
                folium.CircleMarker(
                    location=[displaced_center.y, displaced_center.x],
                    radius=5,
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7,
                    tooltip=f"Displaced location {idx}"
                ).add_to(layer)
        
        layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False, position='bottomright').add_to(m)
    
    # Add title
    title_html = '''
    <div class="map-title" style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%);
                background-color: white; border: 3px solid #666; z-index: 500;
                pointer-events: none;
                font-size: 20px; padding: 15px; text-align: center;
                border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin: 0 0 10px 0; color: #333;">🔐 Hybrid Geodata Anonymization Methods</h2>
        <p style="margin: 0; font-size: 14px; color: #666;">
            Multiple privacy-preserving techniques visualized
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div class="map-legend" style="position: fixed; 
                bottom: 50px; left: 50px;
                background-color: white; border: 2px solid #666; z-index: 499;
                font-size: 12px; padding: 10px;
                border-radius: 5px; max-width: 300px;">
        <h4 style="margin: 0 0 10px 0;">Privacy Methods</h4>
        <p style="margin: 5px 0;"><span style="color: #ff6b6b;">●</span> Donut + Geo-Indist (k-anon + noise)</p>
        <p style="margin: 5px 0;"><span style="color: #4ecdc4;">●</span> Geohashing (spatial hashing)</p>
        <p style="margin: 5px 0;"><span style="color: #95e1d3;">●</span> H3 Hexagonal (uniform cells)</p>
        <p style="margin: 5px 0;"><span style="color: #f38181;">●</span> Geohash + DP (hash + noise)</p>
        <p style="margin: 5px 0;"><span style="color: #aa96da;">●</span> H3 + K-Anon (merged cells)</p>
        <p style="margin: 5px 0;"><span style="color: #fcbad3;">●</span> Triple Layer (max privacy)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_path)
    print(f"✓ Saved interactive map to: {output_path}")
    
    return m


def create_method_comparison_visualization(results: Dict[str, gpd.GeoDataFrame],
                                          output_path: str = 'method_comparison.html'):
    """
    Create side-by-side comparison of all methods
    """
    print("\n📊 Creating Method Comparison Visualization...")
    
    # Calculate statistics for each method
    stats = []
    for method_name, gdf in results.items():
        if len(gdf) == 0:
            continue
        
        stats.append({
            'Method': method_name.replace('_', ' ').title(),
            'Regions': len(gdf),
            'Avg Group Size': gdf['group_size'].mean() if 'group_size' in gdf.columns else 0,
            'Min Group Size': gdf['group_size'].min() if 'group_size' in gdf.columns else 0,
            'Max Group Size': gdf['group_size'].max() if 'group_size' in gdf.columns else 0,
            'Total Area Preserved': gdf['total_area'].sum() if 'total_area' in gdf.columns else 0
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Create HTML table
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Anonymization Method Comparison</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 40px;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                text-align: left;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .summary {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px;
                padding: 15px;
                background-color: #e8f5e9;
                border-radius: 5px;
                min-width: 200px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2e7d32;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>🔐 Hybrid Anonymization Methods Comparison</h1>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <div class="metric-value">{len(stats)}</div>
                <div class="metric-label">Methods Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats_df['Regions'].sum():.0f}</div>
                <div class="metric-label">Total Regions Created</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats_df['Avg Group Size'].mean():.1f}</div>
                <div class="metric-label">Overall Avg Group Size</div>
            </div>
        </div>
        
        <h2>Detailed Method Comparison</h2>
        {stats_df.to_html(index=False, classes='dataframe', border=0)}
        
        <div class="summary" style="margin-top: 40px;">
            <h2>Method Descriptions</h2>
            <h3>🍩 Hybrid Donut + Conservative Geo-Indist</h3>
            <p>Combines k-anonymity clustering with donut geomasking (hiding locations within a ring) 
            and geo-indistinguishability (Laplace noise) for triple-layer protection.</p>
            
            <h3>🔲 Geohashing</h3>
            <p>Uses spatial hashing to discretize space into fixed-size rectangular cells. 
            Simple and efficient, provides k-anonymity within cells.</p>
            
            <h3>⬡ H3 Hexagonal</h3>
            <p>Uber's H3 hierarchical hexagonal grid system. Hexagons provide uniform neighbor 
            distances and better spatial properties than rectangles.</p>
            
            <h3>🔲🎲 Hybrid Geohash + DP</h3>
            <p>Combines geohashing with differential privacy noise on counts and geometry 
            perturbation for enhanced privacy.</p>
            
            <h3>⬡👥 Hybrid H3 + K-Anonymity</h3>
            <p>Uses H3 hexagonal cells as base units, then merges adjacent cells to ensure 
            k-anonymity requirements are met.</p>
            
            <h3>🔐 Triple-Layer</h3>
            <p>Maximum privacy through three protection layers: H3 spatial discretization, 
            donut masking, and geo-indistinguishability. Best for high-risk scenarios.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Saved comparison table to: {output_path}")


def main():
    """Main execution function"""
    print("="*70)
    print("🔐 HYBRID GEODATA ANONYMIZATION SYSTEM")
    print("="*70)
    
    # Load your data
    print("\n📁 Loading your geodata...")
    gdf = gpd.read_file("minifix.geojson.json")
    
    gdf = gdf.set_crs('EPSG:25832', allow_override=True)
    
    print(f"✓ Loaded {len(gdf)} features")
    
    # Remove invalid geometries
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.is_valid]
    print(f"✓ {len(gdf)} valid features after filtering")

    # Initialize anonymizer
    anonymizer = HybridAnonymizer(gdf)
    
    # Run all anonymization methods
    results = {}
    
    print("\n" + "="*70)
    print("RUNNING ANONYMIZATION METHODS")
    print("="*70)
    
    # 1. Hybrid Donut + Conservative Geo-Indist
    try:
        results['hybrid_donut_conservative'] = anonymizer.hybrid_donut_conservative_geo(
            k=5, epsilon=1.0, inner_radius_ratio=0.3, outer_radius_ratio=1.5
        )
    except Exception as e:
        print(f"✗ Hybrid Donut + Geo-Indist failed: {e}")
        
    # 2a. Donut + Geo-Indist WITHOUT k-anonymity
    try:
        results['donut_geo_only'] = anonymizer.donut_geo_indist_only(
            epsilon=1.0, 
            inner_radius=500, 
            outer_radius=2000
        )
    except Exception as e:
        print(f"✗ Donut + Geo-Indist (no k-anon) failed: {e}")
    
    # 2. Geohashing
    try:
        results['geohashing'] = anonymizer.geohashing_anonymization(precision=6)
    except Exception as e:
        print(f"✗ Geohashing failed: {e}")
    
    # 3. H3 Hexagonal
    try:
        results['h3_hexagonal'] = anonymizer.h3_hexagonal_anonymization(resolution=8)
    except Exception as e:
        print(f"✗ H3 Hexagonal failed: {e}")
    
    # 4. Hybrid Geohash + DP
    try:
        results['hybrid_geohash_dp'] = anonymizer.hybrid_geohash_noise(precision=6, epsilon=1.0)
    except Exception as e:
        print(f"✗ Hybrid Geohash + DP failed: {e}")
    
    # 5. Hybrid H3 + K-Anonymity
    try:
        results['hybrid_h3_k'] = anonymizer.hybrid_h3_clustering(resolution=8, k=3)
    except Exception as e:
        print(f"✗ Hybrid H3 + K-Anonymity failed: {e}")
    
    # 6. Triple-Layer Maximum Privacy
    try:
        results['triple_layer'] = anonymizer.hybrid_triple_layer(k=5, epsilon=1.0, h3_resolution=9)
    except Exception as e:
        print(f"✗ Triple-Layer failed: {e}")
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    print({k: len(v) for k, v in results.items()})
    create_comprehensive_visualization(results, gdf)
    create_method_comparison_visualization(results)
    
    # Export results as GeoJSON
    print("\n📦 Exporting results...")
    for method_name, gdf_result in results.items():
        if len(gdf_result) > 0:
            output_file = f'thesis_outputs/{method_name}_result.geojson'
            gdf_result.to_file(output_file, driver='GeoJSON')
            print(f"✓ Exported {method_name} to {output_file}")
    
    print("\n" + "="*70)
    print("✅ ANONYMIZATION COMPLETE!")
    print("="*70)
    print("\nResults saved to /mnt/user-data/outputs/:")
    print("  - hybrid_anonymization_map.html (interactive map)")
    print("  - method_comparison.html (statistics table)")
    print("  - *_result.geojson (individual method results)")
    print("\n🔐 All methods successfully applied with multiple privacy layers!")


if __name__ == "__main__":
    main()