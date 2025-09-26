# Longitudinal ECHO Learning Network Dataset

## Dataset Overview
This synthetic dataset captures the growth and evolution of a Project ECHO telementoring network over 5 years (2019-2023), focusing on opioid use disorder (OUD) treatment in rural communities. It demonstrates typical ECHO network expansion patterns: starting with a small pilot, adding new spoke sites, incorporating subject matter experts, and developing peer learning connections as the network matures.

## Temporal Coverage
- **Time Period**: January 2019 - December 2023 (60 months)
- **Observation Points**: Quarterly snapshots (20 time periods)
- **Growth Pattern**: Organic expansion from 5 initial nodes to 42 nodes

## Network Evolution Summary
- **Q1 2019**: Launch with 1 hub + 4 pilot spokes
- **2019-2020**: Early growth, adding rural clinics
- **2021**: Expert integration and first peer connections
- **2022**: Rapid expansion with state funding
- **2023**: Network maturation with strong peer learning

## File Descriptions

### 1. echo_longitudinal_edges.csv
Contains all network relationships with temporal information.

**Variables:**
- `edge_id`: Unique identifier for each edge
- `from`: Source node of the relationship
- `to`: Target node of the relationship
- `weight`: Strength of connection (1-20 scale, evolves over time)
- `relationship_type`: Type of connection
  - `teaching`: Hub teaches spokes (core ECHO model)
  - `expertise`: Experts provide specialized knowledge
  - `peer_learning`: Spoke-to-spoke knowledge exchange
  - `collaboration`: Joint projects or initiatives
  - `mentorship`: Experienced spokes mentor newer ones
- `start_quarter`: When relationship began (e.g., "2019_Q1")
- `end_quarter`: When relationship ended (NULL if ongoing)
- `interaction_mode`: How interaction occurs
  - `synchronous`: Live video sessions
  - `asynchronous`: Email, forums, recorded content
  - `hybrid`: Both synchronous and asynchronous
- `knowledge_domain`: Primary content area
  - `clinical_protocols`: Treatment guidelines
  - `medication_management`: MAT/buprenorphine
  - `behavioral_health`: Counseling approaches
  - `regulatory_compliance`: Legal/policy issues
  - `case_consultation`: Patient case discussions

### 2. echo_longitudinal_nodes.csv
Contains node attributes with time-varying characteristics.

**Variables:**
- `node_id`: Unique identifier for each node
- `node_name`: Organization or expert name
- `node_type`: Role in the network
  - `hub`: Central ECHO hub
  - `spoke_clinic`: Primary care clinic
  - `spoke_hospital`: Hospital or health system
  - `spoke_fqhc`: Federally Qualified Health Center
  - `expert_clinical`: Clinical specialist
  - `expert_policy`: Policy/regulatory expert
  - `expert_behavioral`: Behavioral health specialist
- `join_quarter`: When joined network (e.g., "2019_Q1")
- `leave_quarter`: When left network (NULL if still active)
- `state`: Geographic location (NM, AZ, CO, TX, UT, NV)
- `rurality_score`: RUCC code (1=urban to 9=most rural)
- `initial_capacity`: Starting capacity metrics
  - `staff_fte`: Clinical FTEs at joining
  - `oud_experience`: Prior OUD treatment experience (0-10 scale)
  - `tech_readiness`: Technology infrastructure (1-5 scale)
- `growth_trajectory`: How node evolved
  - `stable`: Consistent participation
  - `growing`: Increasing engagement
  - `declining`: Decreasing participation
  - `variable`: Fluctuating engagement

### 3. echo_longitudinal_metrics.csv
Quarterly network-level and node-level metrics.

**Variables:**
- `quarter`: Time period (e.g., "2019_Q1")
- `node_id`: Node identifier (NULL for network-level metrics)
- `metric_type`: Category of metric
  - `network_size`: Total active nodes
  - `network_density`: Connection density
  - `participation_rate`: Session attendance %
  - `knowledge_score`: Self-reported competency (1-10)
  - `patient_volume`: Patients treated for OUD
  - `implementation_score`: Protocol adoption (1-10)
- `metric_value`: Numerical value
- `data_quality`: Completeness of data
  - `complete`: All data available
  - `partial`: Some missing elements
  - `estimated`: Imputed or projected

## Key Temporal Patterns

### Growth Phases
1. **Pilot Phase (2019 Q1-Q3)**: Small, tight-knit group establishing protocols
2. **Early Expansion (2019 Q4-2020 Q4)**: Adding nearby clinics, building trust
3. **Expert Integration (2021 Q1-Q2)**: Bringing in specialists, diversifying content
4. **Rapid Growth (2021 Q3-2022 Q4)**: State funding enables broader recruitment
5. **Maturation (2023)**: Stable network with robust peer learning

### Evolution Indicators
- **Edge Persistence**: 78% of teaching edges last >8 quarters
- **Peer Learning Emergence**: First appears Q3 2020, accelerates after Q1 2021
- **Geographic Clustering**: Regional sub-networks emerge by 2022
- **Expertise Diversification**: Clinical-only → multidisciplinary by 2021

## Usage Examples

### Loading in R
```r
library(tidyverse)
library(igraph)

# Load data
edges <- read_csv("echo_longitudinal_edges.csv")
nodes <- read_csv("echo_longitudinal_nodes.csv")
metrics <- read_csv("echo_longitudinal_metrics.csv")

# Create network for specific quarter
q_2021_Q4 <- edges %>%
  filter(start_quarter <= "2021_Q4", 
         is.na(end_quarter) | end_quarter > "2021_Q4")

g_2021_Q4 <- graph_from_data_frame(
  d = q_2021_Q4,
  vertices = nodes %>% filter(join_quarter <= "2021_Q4"),
  directed = TRUE
)
```

### Loading in Python
```python
import pandas as pd
import networkx as nx

# Load data
edges = pd.read_csv("echo_longitudinal_edges.csv")
nodes = pd.read_csv("echo_longitudinal_nodes.csv")
metrics = pd.read_csv("echo_longitudinal_metrics.csv")

# Create network for specific quarter
def get_network_at_time(quarter):
    active_edges = edges[
        (edges['start_quarter'] <= quarter) & 
        (edges['end_quarter'].isna() | (edges['end_quarter'] > quarter))
    ]
    
    active_nodes = nodes[nodes['join_quarter'] <= quarter]
    
    G = nx.from_pandas_edgelist(
        active_edges, 
        source='from', 
        target='to',
        edge_attr=True,
        create_using=nx.DiGraph()
    )
    
    return G
```

## Research Questions This Dataset Can Address

1. **Network Growth**: How do ECHO networks expand over time?
2. **Peer Learning Evolution**: When and how do peer connections emerge?
3. **Expertise Integration**: Impact of adding subject matter experts
4. **Geographic Patterns**: Do regional clusters form naturally?
5. **Sustainability**: What predicts node retention vs. attrition?
6. **Knowledge Diffusion**: How does expertise spread through the network?
7. **Capacity Building**: How do node capabilities change over time?

## Data Generation Notes
- Based on real ECHO network growth patterns from literature
- Incorporates realistic delays in peer connection formation
- Includes natural attrition (5-10% annually)
- Weights evolve to show strengthening relationships
- Geographic expansion follows funding and policy changes