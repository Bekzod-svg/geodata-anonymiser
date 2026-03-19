# Geodata Anonymisation Framework for Cadastral Forest Data

A Python framework for evaluating and benchmarking privacy-preserving anonymisation techniques on cadastral forest parcel data. Developed as part of a Master's thesis on anonymisation of geodata spaces (FH Dortmund, 2025–2026).

---

## Overview

This framework implements and evaluates **12 anonymisation methods** across three categories on real-world forest parcel data from Baden-Württemberg, Germany (2,147 parcels, EPSG:25832). Each method is assessed against a structured **threat model** simulating three realistic adversarial attacks.

---

## Project Structure

```
.
├── cadastral_anonymizer.py   # Traditional & DP-based anonymisation methods
├── hybrid_anon.py            # Hybrid anonymisation methods (Geohash+DP, H3+k, etc.)
├── threat_model.py           # Attack simulator & comprehensive evaluation pipeline
├── minifix.geojson.json      # Input dataset (forest parcels, Baden-Württemberg)
└── thesis_outputs/
    └── full_parameter_threat_analysis.csv   # Evaluation results
```

---

## Anonymisation Methods

### Traditional Methods (`cadastral_anonymizer.py`)
| Method | Key Parameter |
|---|---|
| K-Anonymity (spatial clustering) | k ∈ {3, 5, 10} |
| Differential Privacy Grid Aggregation | ε ∈ {0.5, 1.0, 5.0} |
| Geo-Indistinguishability (planar Laplace) | ε ∈ {1.0, 2.0, 5.0} |
| Topology-Preserving Generalisation | tolerance ∈ {1m, 2m, 5m} |
| Donut Geomasking | k ∈ {5, 50} |

### Hybrid Methods (`hybrid_anon.py`)
| Method | Description |
|---|---|
| Geohash + DP Noise | Grid-based encoding with Laplace noise |
| H3 Hexagonal + K-Clustering | Uber H3 spatial indexing with k-anonymity |
| Donut + Conservative Geo-Indistinguishability | Displacement masking with bounded DP noise |
| Donut + Vertex Perturbation | Shape-level noise injection |

---

## Threat Model

Three adversarial attacks are simulated in `threat_model.py`:

| Attack | Description | Weight |
|---|---|---|
| **Homogeneity Attack** | Exploits uniform sensitive attributes within spatial clusters | 30% |
| **Background Knowledge Attack** | Re-identifies parcels using approximate location + cadastral area | 40% |
| **Satellite Correlation Attack** | Matches parcel shapes via Hausdorff distance against satellite imagery | 30% |

**Overall Vulnerability Score:**
```
Vuln = 0.3 × Homogeneity + 0.4 × Background + 0.3 × Satellite
```

---

## Key Findings

- **Optimal configuration:** `hybrid_geohash_dp_prec5` — **5.2% overall vulnerability**
- **Lowest vulnerability:** `dp_grid_eps0.5` — **0.1%** (high utility trade-off)
- **Critical insight:** Shape-preserving methods retain **100% satellite correlation vulnerability** regardless of parameter tuning (*Geometry-Privacy Paradox*)
- **Privacy cliff:** Geohash precision 7 produces a sharp vulnerability spike due to near-original spatial resolution

---

## Installation & Usage

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install numpy pandas geopandas shapely matplotlib folium scikit-learn h3 scipy geohash2

# 3. Run the anonymiser
python cadastral_anonymizer.py
```

To run the full threat model evaluation:

```bash
python threat_model.py
```

Results are saved to `thesis_outputs/full_parameter_threat_analysis.csv`.

To use individual anonymisers:

```python
from cadastral_anonymizer import OptimizedCadastralAnonymizer
from hybrid_anon import HybridAnonymizer

anon = OptimizedCadastralAnonymizer(gdf)
result = anon.dp_grid_aggregation(epsilon=1.0, grid_resolution=8)

hybrid = HybridAnonymizer(gdf)
result = hybrid.hybrid_geohash_noise(precision=5, epsilon=1.0)
```

---

## Research Context

- **Thesis:** *Anonymisation of Cadastral Forest Data: Privacy Techniques for Geodata Spaces*
- **Institution:** Fachhochschule Dortmund, Digital Transformation (M.Sc.)
- **Supervisor:** Prof. Dr. Jan Cirullies
- **Industry Partners:** wetransform GmbH, M.O.S.S. GmbH

---

## License

For academic use only. Data sourced from public cadastral records (Baden-Württemberg, Germany).