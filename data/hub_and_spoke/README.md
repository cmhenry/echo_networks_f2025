# Small Hub-and-Spoke ECHO Learning Network Dataset

## Dataset Overview
This synthetic dataset represents a Project ECHO telementoring network focused on diabetes care in rural/underserved areas. It captures the structure typical of ECHO programs where a central hub provides expertise to multiple spoke sites.

## Network Structure
- **Total Nodes**: 17
  - 1 Hub (UNM ECHO Hub)
  - 12 Spoke sites (clinics/hospitals)
  - 4 Expert consultants
- **Total Edges**: 27
  - Hub-to-spoke teaching connections
  - Expert-to-hub knowledge connections
  - Spoke-to-spoke peer learning connections

## File Descriptions

### 1. echo_edges.csv
Contains the network relationships (edges) between participants.

**Variables:**
- `from`: Source node of the relationship
- `to`: Target node of the relationship
- `weight`: Strength of connection (1-20 scale, higher = stronger)
- `relationship_type`: Type of connection
  - `teaching`: Hub teaches spokes (main ECHO model)
  - `expertise`: Experts provide knowledge to hub
  - `peer_learning`: Spoke-to-spoke knowledge exchange
  - `collaboration`: Active project collaboration
  - `referral`: Patient/case referral relationship
- `interaction_frequency`: How often interaction occurs
  - `weekly`: Regular ECHO sessions
  - `bi-weekly`: Every other week
  - `monthly`: Monthly interaction
  - `occasional`: Irregular/as-needed

### 2. echo_nodes.csv
Contains attributes for each node in the network.

**Variables:**
- `node_id`: Unique identifier for each node
- `node_type`: Role in the network
  - `hub`: Central ECHO hub
  - `spoke`: Participating site
  - `expert`: Subject matter expert
- `state`: Geographic location (NM, TX, CO, AZ)
- `years_in_echo`: How long participating in ECHO (0-15)
- `specialty_focus`: Clinical specialty or focus area
- `staff_size`: Number of clinical staff at site
- `rural_urban`: Geographic classification
- `patient_volume`: Approximate patients served monthly (0 for experts/hub)
- `participation_rate`: Percentage attendance at ECHO sessions (0-100)

## Key Network Features

1. **Hub-and-Spoke Pattern**: Classic ECHO structure with central hub connected to all spokes
2. **Expert Layer**: Specialists feed expertise into the hub
3. **Peer Connections**: Some spokes connect directly (representing informal learning networks)
4. **Regional Clusters**: Some geographic proximity effects in peer connections
5. **Heterogeneous Participation**: Varying levels of engagement (see participation_rate)
