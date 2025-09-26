# Medium Collaboration Network Dataset

## Overview
This synthetic dataset represents a multi-regional healthcare collaboration network with hierarchical structure and diverse relationship types. It's designed to demonstrate network analysis techniques at medium scale with more complex patterns than simple hub-and-spoke models.

## Network Structure
- **Total Nodes**: 57
  - 1 Central hub (University Medical Center)
  - 5 Regional hubs
  - 8 Hospitals
  - 33 Clinics
  - 10 Support organizations
  
- **Total Edges**: 146
  - Multi-level hierarchy
  - Regional clustering
  - Peer-to-peer connections
  - Cross-regional collaborations

## File Descriptions

### 1. medium_network_edges.csv
Edge list containing 146 directed relationships.

**Variables:**
- `from`: Source node
- `to`: Target node
- `weight`: Strength of collaboration (1-25 scale)
- `collaboration_type`: Type of relationship
  - `mentoring`: Hub-to-clinic knowledge transfer
  - `partnership`: Formal organizational partnerships
  - `coordination`: Administrative/governance links
  - `peer_learning`: Clinic-to-clinic knowledge sharing
  - `referral`: Patient/case referrals
  - `research`: Research collaborations
  - `data_sharing`: Information system connections
  - `consultation`: Specialist services
  - `training`: Educational programs
  - `quality_review`: Quality improvement activities
  - And others...
- `start_year`: When collaboration began (2018-2023)
- `active`: Whether relationship is currently active (1=yes, 0=no)

### 2. medium_network_nodes.csv
Node attributes for all 57 actors.

**Variables:**
- `node_id`: Unique identifier
- `node_type`: Role in network
  - `central_hub`: Main coordinating center
  - `regional_hub`: Regional coordinators
  - `hospital`: Hospital facilities
  - `clinic`: Primary care clinics
  - `specialist_group`: Specialist networks
  - `support`: Supporting organizations
  - `community`: Community partners
- `region`: Geographic region (West, East, Central, North, South, Multi)
- `organization_type`: Type of organization
- `years_active`: How long in the network
- `size_category`: Organization size (small, medium, large)
- `specialty_area`: Clinical focus area
- `annual_volume`: Patient volume (0 for non-clinical entities)
- `quality_score`: Quality metric (0-100)
- `engagement_level`: Participation rate (0-100)
- `urban_rural`: Geographic setting
- `funding_type`: Primary funding source

## Key Network Features

### 1. **Multi-Level Hierarchy**
- University Medical Center at the top
- Regional hubs as intermediaries
- Local clinics and hospitals at the periphery

### 2. **Regional Clustering**
- Five distinct geographic regions
- Within-region connections stronger than between-region
- Regional hubs coordinate local activities

### 3. **Support Infrastructure**
- Research Institute for evidence generation
- Data Center for information management
- Training Center for capacity building
- Quality Improvement Team for standards
- Policy Advisory Board for governance

### 4. **Diverse Relationship Types**
- Vertical: mentoring, coordination
- Horizontal: peer learning, collaboration
- Service: referral, consultation
- Infrastructure: data sharing, technology

### 5. **Temporal Evolution**
- Network started 2018-2019 with core infrastructure
- Expanded regionally 2020-2021
- Added new clinics 2022-2023

## Analysis Opportunities

This dataset is ideal for demonstrating:
- **Community detection**: Finding regional clusters
- **Multi-level analysis**: Examining hierarchical structure
- **Brokerage analysis**: Identifying bridging organizations
- **Resilience testing**: Impact of removing regional hubs
- **Diffusion modeling**: How innovations spread through regions
- **Homophily analysis**: Similar organizations connecting
- **Temporal patterns**: Network growth over time
- **Core-periphery structure**: Multiple cores with peripheries

## Loading the Data

### In R:
```r
# Load data
edges <- read.csv("medium_network_edges.csv")
nodes <- read.csv("medium_network_nodes.csv")

# Using igraph
library(igraph)
g <- graph_from_data_frame(edges, directed = TRUE, vertices = nodes)

# Using netify
library(netify)
# Convert to adjacency matrix first
actors <- unique(c(edges$from, edges$to))
adj_matrix <- matrix(0, length(actors), length(actors), 
                     dimnames = list(actors, actors))
for(i in 1:nrow(edges)) {
  adj_matrix[edges$from[i], edges$to[i]] <- edges$weight[i]
}
net <- netify(adj_matrix, directed = TRUE)
```

### In Python:
```python
import pandas as pd
import networkx as nx

# Load data
edges = pd.read_csv("medium_network_edges.csv")
nodes = pd.read_csv("medium_network_nodes.csv")

# Create network
G = nx.from_pandas_edgelist(edges, 
                            source='from', 
                            target='to',
                            edge_attr=['weight', 'collaboration_type'],
                            create_using=nx.DiGraph())

# Add node attributes
node_attrs = nodes.set_index('node_id').to_dict('index')
nx.set_node_attributes(G, node_attrs)
```

## Comparison to Small Dataset

| Feature | Small Network | Medium Network |
|---------|--------------|----------------|
| Nodes | 17 | 57 |
| Edges | 27 | 146 |
| Structure | Simple hub-spoke | Multi-level hierarchy |
| Regions | 4 states | 5 regions |
| Hubs | 1 central | 1 central + 5 regional |
| Density | ~0.10 | ~0.05 |
| Complexity | Basic | Moderate |

## Research Questions

This dataset can address questions like:
1. How does regional structure affect collaboration patterns?
2. Which organizations bridge different regions?
3. How resilient is the network to regional hub failure?
4. Do similar organizations preferentially connect?
5. How has the network evolved over time?
6. What's the optimal structure for knowledge diffusion?
7. Which nodes are critical for network connectivity?

## Notes
- All data is synthetic but reflects realistic collaboration patterns
- Weights represent interaction frequency/intensity
- Network shows typical healthcare system hierarchies
- Regional variation intentionally included for analysis