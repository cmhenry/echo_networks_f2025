#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2: Basic Network Descriptives
Network Analysis for Project ECHO - Interactive Learning Module (Python Version)
"""

# %% [markdown]
# # Introduction: Understanding Network Structure in Healthcare Learning
#
# ## Why Network Analysis for Project ECHO?
#
# How do we understand if knowledge-sharing networks are working effectively? 
# How do we identify which partners might need more support, or which ones could serve as peer mentors?
#
# Network analysis provides the mathematical and visual tools to answer these questions.
#
# ### The Power of Network Thinking
#
# Consider two rural clinics with identical resources and staff qualifications. 
# Traditional analysis might predict similar outcomes. But what if one clinic is 
# well-connected to peers and experts while the other is isolated? Network analysis 
# reveals these hidden structural advantages that can make the difference between 
# thriving and struggling.
#
# ### Learning Objectives
#
# By the end of this interactive notebook, you will be able to:
# 1. **Transform** relational data from edge lists to network formats
# 2. **Visualize** network structure to identify patterns and anomalies
# 3. **Calculate** centrality measures to find key actors
# 4. **Interpret** network-level statistics for system insights
# 5. **Apply** these techniques to real healthcare networks

# %% [markdown]
# ### Setting Up Your Environment

# %%
# ============================================
# NETWORK ANALYSIS TOOLKIT
# ============================================

# Core network analysis packages
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set default plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Check package versions
print("Package versions:")
print(f"- networkx: {nx.__version__}")
print(f"- pandas: {pd.__version__}")
print(f"- numpy: {np.__version__}")
print(f"- matplotlib: {plt.matplotlib.__version__}\n")

# %% [markdown]
# ---
# # Part 1: Data Transformation - From Lists to Networks
#
# ## Understanding Network Data Structures
#
# Network data is fundamentally different from traditional tabular data. Instead of rows 
# representing independent observations, we have **actors** (nodes) and **relationships** 
# (edges) that form an interconnected system.
#
# ### The Two Fundamental Representations
#
# There are two primary ways to represent network data, each with distinct advantages:
#
# 1. **Edge List Format**: A table where each row represents one relationship
#    - Intuitive and human-readable
#    - Efficient for sparse networks
#    - Easy to collect from surveys or observations
#    
# 2. **Adjacency Matrix Format**: A square matrix where cells indicate connections
#    - Computationally efficient for algorithms
#    - Shows both presence and absence of ties
#    - Required input for many network functions

# %%
# ---------------------------------------------
# LOADING THE ECHO NETWORK DATA
# ---------------------------------------------

# Load edge list and node attributes
edges = pd.read_csv("data/hub_and_spoke/echo_edges.csv")
nodes = pd.read_csv("data/hub_and_spoke/echo_nodes.csv")

# Examine the edge list structure
print("EDGE LIST STRUCTURE:")
print("===================")
print(edges.head(3).to_string())

print("\n\nWhat do these columns mean?")
print("- 'from': The initiating actor (who reaches out)")
print("- 'to': The receiving actor (who is contacted)")
print("- 'weight': Strength of the relationship (1-20 scale)")
print("- 'relationship_type': Nature of the connection")
print("- 'interaction_frequency': How often they interact")

# %% [markdown]
# ### Understanding Our Actors
#
# Before diving into network structure, let's understand who's in our network:

# %%
# Examine node attributes
print("\nNODE ATTRIBUTES:")
print("===============")
print(nodes.head(3).to_string())

# Summarize network composition
print("\n\nNETWORK COMPOSITION:")
print("====================\n")

node_summary = nodes.groupby('node_type').agg({
    'node_id': 'count',
    'participation_rate': 'mean',
    'state': 'nunique'
}).rename(columns={'node_id': 'count', 'participation_rate': 'avg_participation', 'state': 'states'})

print(node_summary.to_string())

print("\nKey observations:")
print("- One central hub (academic medical center)")
print("- Multiple spoke sites (community clinics)")
print("- Expert consultants providing specialized knowledge")
print("- Geographic distribution across multiple states")

# %% [markdown]
# ## Creating the Adjacency Matrix
#
# Now let's transform our edge list into an adjacency matrix - the computational workhorse of network analysis:

# %%
# ---------------------------------------------
# EDGE LIST TO ADJACENCY MATRIX TRANSFORMATION
# ---------------------------------------------

# Extract unique actors
actors = list(set(edges['from'].tolist() + edges['to'].tolist()))
n = len(actors)

print("Network dimensions:")
print(f"- Total actors: {n}")
print(f"- Total relationships: {len(edges)}")
print(f"- Matrix size: {n} x {n} = {n*n} cells\n")

# Create adjacency matrix using pandas for easier indexing
adjacency_matrix = pd.DataFrame(0, index=actors, columns=actors)

# Populate with edge weights
for _, edge in edges.iterrows():
    adjacency_matrix.loc[edge['from'], edge['to']] = edge['weight']

# Display a portion of the matrix
print("Adjacency matrix (first 5x5 subset):")
print(adjacency_matrix.iloc[:5, :5].to_string())

# Calculate sparsity
A = adjacency_matrix.values
sparsity = np.sum(A == 0) / (n * n)
print(f"\nMatrix sparsity: {sparsity * 100:.1f}%")
print(f"This means {sparsity * 100:.1f}% of possible connections don't exist.")
print("This is typical for real-world networks!")

# %% [markdown]
# ## Creating a NetworkX Graph Object
#
# NetworkX is Python's premier network analysis library. Let's create our network object:

# %%
# Create NetworkX directed graph from edge list
G = nx.from_pandas_edgelist(edges, 
                             source='from', 
                             target='to', 
                             edge_attr=['weight', 'relationship_type', 'interaction_frequency'],
                             create_using=nx.DiGraph())

# Add node attributes
node_attrs = nodes.set_index('node_id').to_dict('index')
nx.set_node_attributes(G, node_attrs)

# Display network summary
print(f"Network type: {type(G).__name__}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is directed: {G.is_directed()}")
print(f"Is weighted: {'weight' in next(iter(G.edges(data=True)))[2]}")

print("\nWhat NetworkX tells us:")
print("- Network type: directed (connections have direction)")
print("- Observation period: single time point (cross-sectional)")
print("- Ready for analysis with NetworkX functions")

# %% [markdown]
# ---
# # Part 2: Network Visualization - Seeing the Structure
#
# ## The Art and Science of Network Visualization
#
# A good network visualization can reveal patterns that would take hours to discover in data tables. 
# But creating effective visualizations requires thoughtful choices about layout, color, and visual encoding.

# %%
# ---------------------------------------------
# NETWORK VISUALIZATION SETUP
# ---------------------------------------------

# Create visual encodings based on node attributes
color_map = {'hub': 'red', 'spoke': 'lightblue', 'expert': 'gold'}
node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]

# Size nodes by participation rate (normalized for visibility)
node_sizes = [G.nodes[node]['participation_rate'] * 10 + 100 for node in G.nodes()]

# Edge widths based on weight
edge_widths = [G[u][v]['weight'] / 4 for u, v in G.edges()]

print("Visual encoding scheme:")
print("- Color: Node type (red=hub, blue=spoke, gold=expert)")
print("- Size: Participation rate (larger = more active)")
print("- Edge width: Interaction strength")

# %% [markdown]
# ### Creating the Network Visualization

# %%
# Set up the plot
plt.figure(figsize=(12, 8))

# Use Kamada-Kawai layout for better visualization
pos = nx.kamada_kawai_layout(G)

# Draw the network
nx.draw_networkx_nodes(G, pos, 
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.9)

nx.draw_networkx_edges(G, pos,
                       width=edge_widths,
                       alpha=0.5,
                       edge_color='gray',
                       arrows=True,
                       arrowsize=10,
                       arrowstyle='->')

nx.draw_networkx_labels(G, pos, 
                        font_size=8,
                        font_family='sans-serif')

plt.title("ECHO Learning Network Structure", fontsize=16, fontweight='bold')
plt.suptitle("Node size = participation rate, Edge width = interaction strength", 
             fontsize=10, y=0.02)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='Hub'),
                  Patch(facecolor='lightblue', label='Spoke'),
                  Patch(facecolor='gold', label='Expert')]
plt.legend(handles=legend_elements, loc='upper right')

plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpreting the Visualization

# %%
print("\nWHAT THE VISUALIZATION REVEALS:")
print("================================\n")

print("1. HUB-AND-SPOKE ARCHITECTURE:")
print("   - Central red node connects to most blue spokes")
print("   - Classic telementoring structure clearly visible")
print("   - Information flows primarily through the hub\n")

print("2. PEER LEARNING PATTERNS:")
print("   - Some blue nodes connect directly to each other")
print("   - These peer connections may indicate:")
print("     * Geographic proximity")
print("     * Shared specialties")
print("     * Informal knowledge exchange\n")

print("3. EXPERT POSITIONING:")
print("   - Gold nodes (experts) connect primarily to hub")
print("   - Expertise flows through hub to spokes")
print("   - No direct expert-to-spoke connections\n")

print("4. PARTICIPATION PATTERNS:")
print("   - Node sizes vary significantly")
print("   - Some spokes are highly active (large nodes)")
print("   - Others may need engagement support (small nodes)")

# %% [markdown]
# ---
# # Part 3: Centrality Analysis - Finding Key Players
#
# ## Multiple Dimensions of Importance
#
# In a network, "importance" isn't one-dimensional. Different actors play different roles:
# - **Connectors**: Have many relationships (high degree)
# - **Brokers**: Bridge different groups (high betweenness)
# - **Influencers**: Connected to other important actors (high eigenvector)

# %%
# ---------------------------------------------
# CENTRALITY MEASURES CALCULATION
# ---------------------------------------------

print("Calculating centrality measures...\n")

# Calculate various centrality measures
degree_in = dict(G.in_degree())
degree_out = dict(G.out_degree())
degree_total = dict(G.degree())

# Betweenness centrality (brokerage position)
betweenness = nx.betweenness_centrality(G, normalized=False)

# Eigenvector centrality (connection to well-connected others)
try:
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
except:
    # If eigenvector fails, use PageRank as alternative
    eigenvector = nx.pagerank(G)

# Compile results into a comprehensive dataframe
centrality_df = pd.DataFrame({
    'actor': list(G.nodes()),
    'type': [G.nodes[node]['node_type'] for node in G.nodes()],
    'in_degree': [degree_in[node] for node in G.nodes()],
    'out_degree': [degree_out[node] for node in G.nodes()],
    'total_degree': [degree_total[node] for node in G.nodes()],
    'betweenness': [round(betweenness[node], 2) for node in G.nodes()],
    'eigenvector': [round(eigenvector[node], 3) for node in G.nodes()]
})

print("Centrality measures calculated successfully!")

# %% [markdown]
# ### Identifying Network Brokers
#
# Brokers control information flow. Let's find them:

# %%
# Identify top brokers by betweenness centrality
print("TOP 5 BROKERS (by betweenness centrality):")
print("==========================================\n")

top_brokers = centrality_df.nlargest(5, 'betweenness')[['actor', 'type', 'betweenness', 'total_degree']]
print(top_brokers.to_string(index=False))

print("\nInterpretation:")
print("- High betweenness indicates control over information flow")
print("- These actors bridge otherwise disconnected groups")
print("- Removing them would increase network fragmentation")
print("- Consider them for leadership or coordination roles")

# %% [markdown]
# ### Comparing Centrality by Node Type
#
# How do different types of actors compare in terms of centrality?

# %%
# Analyze centrality patterns by node type
print("\nAVERAGE CENTRALITY BY NODE TYPE:")
print("=================================\n")

type_comparison = centrality_df.groupby('type').agg({
    'actor': 'count',
    'total_degree': ['mean', 'max'],
    'betweenness': ['mean', 'max'],
    'eigenvector': 'mean'
}).round(2)

type_comparison.columns = ['n', 'avg_degree', 'max_degree', 'avg_betweenness', 
                           'max_betweenness', 'avg_eigenvector']
print(type_comparison.to_string())

print("\nKey insights:")
print("- Hub dominates all centrality measures (as expected)")
print("- Spokes vary widely in connectivity")
print("- Experts have specialized roles (moderate centrality)")
print("- Some spokes may serve as secondary hubs")

# %% [markdown]
# ---
# # Part 4: Network-Level Measures - System Properties
#
# ## Beyond Individual Actors
#
# While centrality tells us about individuals, network-level measures reveal system properties 
# that affect how the entire network functions:

# %%
# ---------------------------------------------
# NETWORK-LEVEL STATISTICS
# ---------------------------------------------

print("=== NETWORK-LEVEL STATISTICS ===")
print("================================\n")

# Calculate key network metrics
# Density: proportion of possible edges that exist
density = nx.density(G)
print(f"Network Density: {density:.3f} ({density * 100:.1f}% of possible connections exist)")
print("   → Interpretation: Low density indicates sparse network")
print("   → Typical for hub-and-spoke architectures\n")

# Centralization: how hierarchical is the network?
# Calculate degree centralization manually
max_degree = max(dict(G.degree()).values())
centralization = sum(max_degree - d for d in dict(G.degree()).values()) / ((n - 1) * (n - 2))
print(f"Degree Centralization: {centralization:.3f}")
print("   → 1 = perfect star (all connections through one node)")
print("   → 0 = everyone equally connected")
print("   → High value confirms hub-dominated structure\n")

# Clustering: do friends of friends know each other?
clustering = nx.transitivity(G)
print(f"Clustering Coefficient: {clustering:.3f}")
print("   → Probability that two of my contacts know each other")
print("   → Low value suggests limited peer-to-peer learning\n")

# Average path length: degrees of separation
if nx.is_strongly_connected(G):
    avg_path = nx.average_shortest_path_length(G)
else:
    # Use largest strongly connected component
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    avg_path = nx.average_shortest_path_length(G_sub)
print(f"Average Path Length: {avg_path:.2f}")
print("   → Average 'degrees of separation' in network")
print("   → Short paths enable rapid information diffusion\n")

# Network diameter: maximum separation
if nx.is_strongly_connected(G):
    diameter = nx.diameter(G)
else:
    diameter = nx.diameter(G_sub)
print(f"Network Diameter: {diameter}")
print("   → Maximum degrees of separation")
print("   → Longest 'shortest path' between any two nodes")

# %% [markdown]
# ### Understanding Network Vulnerability

# %%
print("\nNETWORK RESILIENCE ANALYSIS:")
print("============================\n")

# What happens if the hub fails?
hub_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'hub']
if hub_nodes:
    hub_name = hub_nodes[0]
    
    # Count components before removing hub
    components_with_hub = nx.number_weakly_connected_components(G)
    
    # Create copy and remove hub
    G_no_hub = G.copy()
    G_no_hub.remove_node(hub_name)
    
    # Count components after removing hub
    components_without_hub = nx.number_weakly_connected_components(G_no_hub)
    
    print(f"Components with hub: {components_with_hub}")
    print(f"Components without hub: {components_without_hub}\n")
    
    print("VULNERABILITY ASSESSMENT:")
    print(f"- Removing hub would fragment network into {components_without_hub} disconnected groups")
    print("- This represents a critical vulnerability")
    print("- Recommendation: Develop regional sub-hubs for resilience")

# %% [markdown]
# ---
# # Part 5: Visualizing Centrality Distributions
#
# ## Understanding Variation in Network Position
#
# Let's create visualizations to understand how centrality is distributed across the network:

# %%
# ---------------------------------------------
# VISUALIZATION OF CENTRALITY DISTRIBUTIONS
# ---------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Degree distribution
axes[0, 0].hist(list(degree_total.values()), bins=10, color='lightblue', edgecolor='black')
axes[0, 0].axvline(np.mean(list(degree_total.values())), color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('Degree Distribution')
axes[0, 0].set_xlabel('Number of Connections')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].text(np.mean(list(degree_total.values())) + 0.5, 
                axes[0, 0].get_ylim()[1] * 0.9, 
                f'Mean = {np.mean(list(degree_total.values())):.1f}', color='red')

# 2. Betweenness distribution
axes[0, 1].hist(list(betweenness.values()), bins=10, color='lightgreen', edgecolor='black')
axes[0, 1].axvline(np.mean(list(betweenness.values())), color='red', linestyle='--', linewidth=2)
axes[0, 1].set_title('Betweenness Distribution')
axes[0, 1].set_xlabel('Betweenness Centrality')
axes[0, 1].set_ylabel('Frequency')

# 3. Centrality by node type
type_data = centrality_df.groupby('type')['total_degree'].apply(list).to_dict()
box_data = [type_data.get('expert', []), type_data.get('hub', []), type_data.get('spoke', [])]
bp = axes[1, 0].boxplot(box_data, labels=['Expert', 'Hub', 'Spoke'], 
                         patch_artist=True)
colors = ['gold', 'red', 'lightblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[1, 0].set_title('Degree by Node Type')
axes[1, 0].set_ylabel('Total Degree')

# 4. Participation vs Centrality
participation = [G.nodes[node].get('participation_rate', 0) for node in G.nodes()]
degrees = [degree_total[node] for node in G.nodes()]
node_colors_scatter = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]

axes[1, 1].scatter(participation, degrees, c=node_colors_scatter, s=50, alpha=0.7)
z = np.polyfit(participation, degrees, 1)
p = np.poly1d(z)
axes[1, 1].plot(participation, p(participation), "gray", linestyle='--', alpha=0.5)
axes[1, 1].set_title('Participation vs Connections')
axes[1, 1].set_xlabel('Participation Rate (%)')
axes[1, 1].set_ylabel('Total Degree')

# Calculate and display correlation
cor_value = np.corrcoef(participation, degrees)[0, 1]
axes[1, 1].text(0.7, 0.9, f'r = {cor_value:.2f}', 
                transform=axes[1, 1].transAxes, fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpreting the Distributions

# %%
print("\nDISTRIBUTION INSIGHTS:")
print("======================\n")

# Analyze degree distribution
from scipy import stats
degree_values = list(degree_total.values())
degree_skew = stats.skew(degree_values)

print("1. DEGREE DISTRIBUTION:")
print(f"   - Skewness: {degree_skew:.2f}")
if degree_skew > 1:
    print("   - Highly skewed: Few actors have many connections")
    print("   - Most actors have few connections")
    print("   - Typical of scale-free networks")
else:
    print("   - Relatively even distribution")

# Analyze participation-centrality relationship
print("\n2. PARTICIPATION-CENTRALITY RELATIONSHIP:")
print(f"   - Correlation: {cor_value:.2f}")
if abs(cor_value) > 0.5:
    print("   - Strong relationship detected")
    print("   - Well-connected sites tend to be more engaged")
    print("   - Consider: Does connection drive participation or vice versa?")
else:
    print("   - Weak to moderate relationship")
    print("   - Other factors may be more important for engagement")

# Identify outliers
mean_degree = np.mean(degree_values)
std_degree = np.std(degree_values)
outliers = centrality_df[centrality_df['total_degree'] > mean_degree + 2*std_degree]

if len(outliers) > 0:
    print("\n3. NETWORK OUTLIERS (unusually high connectivity):")
    print(outliers[['actor', 'type', 'total_degree']].to_string(index=False))

# %% [markdown]
# ---
# # Part 6: Key Insights and Recommendations
#
# ## Strategic Insights for Project ECHO
#
# Based on our comprehensive network analysis, here are actionable insights:

# %%
# ---------------------------------------------
# KEY INSIGHTS FOR PRESENTATION
# ---------------------------------------------

print("\n=== KEY INSIGHTS ===")
print("===================\n")

insights = {
    "Network Architecture": "Clear hub-and-spoke structure with limited peer connections",
    "Central Dependency": "High reliance on hub creates vulnerability",
    "Peer Learning": "Some organic peer connections exist but could be strengthened",
    "Expert Integration": "Experts feed knowledge through hub, not directly to spokes",
    "Geographic Patterns": "Evidence of regional clustering in peer connections",
    "Engagement Variation": "Wide range in participation rates across spokes"
}

for i, (key, value) in enumerate(insights.items(), 1):
    print(f"{i}. {key}:\n   {value}\n")

# %% [markdown]
# ## Practical Recommendations

# %%
print("\n=== RECOMMENDATIONS FOR NETWORK IMPROVEMENT ===")
print("==============================================\n")

# Identify specific intervention opportunities
# 1. Find isolated spokes
isolated_spokes = centrality_df[(centrality_df['type'] == 'spoke') & 
                                (centrality_df['total_degree'] <= 2)]

if len(isolated_spokes) > 0:
    print("1. SUPPORT ISOLATED SPOKES:")
    print("   These sites need additional connections:")
    print(isolated_spokes[['actor', 'total_degree']].to_string(index=False))
    print("   → Action: Pair with peer mentors or create buddy systems\n")

# 2. Identify potential peer leaders
spoke_df = centrality_df[centrality_df['type'] == 'spoke']
median_betweenness = spoke_df['betweenness'].median()
peer_leaders = spoke_df[spoke_df['betweenness'] > median_betweenness].nlargest(3, 'betweenness')

print("2. LEVERAGE NATURAL BROKERS:")
print("   These spokes already bridge different groups:")
print(peer_leaders[['actor', 'betweenness', 'total_degree']].to_string(index=False))
print("   → Action: Formalize their role as peer coordinators\n")

# 3. Geographic opportunities
print("3. STRENGTHEN REGIONAL CONNECTIONS:")
print("   → Create state-based special interest groups")
print("   → Facilitate regional peer learning sessions")
print("   → Consider regional sub-hubs for resilience\n")

# 4. Reduce vulnerability
print("4. BUILD NETWORK RESILIENCE:")
print(f"   → Current centralization: {centralization * 100:.1f}%")
print("   → Target: Reduce to below 50% through peer connections")
print("   → Develop backup communication channels")

# %% [markdown]
# ## Network Report Card

# %%
print("\n=== NETWORK REPORT CARD ===")
print("===========================\n")

# Create comprehensive assessment
report_card = pd.DataFrame({
    'Dimension': [
        'Connectivity',
        'Resilience',
        'Efficiency',
        'Peer Learning',
        'Expert Integration',
        'Geographic Coverage'
    ],
    'Grade': ['B+', 'C', 'A', 'C+', 'B', 'B'],
    'Assessment': [
        'Good hub-spoke connectivity, some isolated sites',
        'High vulnerability due to hub dependence',
        'Short paths enable rapid information flow',
        'Limited peer connections (opportunity for growth)',
        'Experts well-integrated through hub',
        'Present across states but uneven distribution'
    ],
    'Recommendation': [
        'Connect isolated spokes to peer mentors',
        'Develop regional sub-hubs for redundancy',
        'Maintain current efficient structure',
        'Foster spoke-to-spoke relationships',
        'Consider direct expert-spoke connections for specialized topics',
        'Balance network growth across regions'
    ]
})

print(report_card.to_string(index=False))

print("\n\nOVERALL ASSESSMENT:")
print("The ECHO network successfully delivers expertise from hub to spokes")
print("but shows opportunities for improvement in peer learning and resilience.")
print("Strategic development of peer connections and regional structures")
print("would enhance both learning outcomes and network sustainability.")

# %% [markdown]
# ---
# # Part 7: Saving Results for Further Analysis
#
# ## Preparing Data for Statistical Modeling
#
# Let's save our network measures for use in Part 3, where we'll incorporate them into statistical models:

# %%
# Save results for further analysis
import pickle

# Create results dictionary
results = {
    'centrality_df': centrality_df,
    'G': G,
    'adjacency_matrix': adjacency_matrix,
    'edges': edges,
    'nodes': nodes,
    'network_metrics': {
        'density': density,
        'centralization': centralization,
        'clustering': clustering,
        'avg_path_length': avg_path
    }
}

# Save to pickle file
with open('part2_complete_analysis.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Complete analysis saved to 'part2_complete_analysis.pkl'\n")

# Also save centrality measures to CSV for easier access
centrality_df.to_csv('network_centrality_measures.csv', index=False)
print("✓ Centrality measures saved to 'network_centrality_measures.csv'\n")

print("Files ready for Part 3: Statistical Network Models")

# %% [markdown]
# ---
# # Summary and Next Steps
#
# ## What We've Accomplished
#
# In this comprehensive network analysis, we've:
#
# ✓ **Transformed** edge list data into network format  
# ✓ **Visualized** the hub-and-spoke architecture  
# ✓ **Calculated** multiple centrality measures  
# ✓ **Analyzed** network-level properties  
# ✓ **Identified** vulnerable points and opportunities  
# ✓ **Generated** specific, actionable recommendations
#
# ## Key Takeaways
#
# 1. **Structure Matters**: The hub-and-spoke architecture efficiently distributes expertise but creates vulnerability
#
# 2. **Multiple Perspectives**: Different centrality measures reveal different aspects of importance
#
# 3. **System Properties**: Network-level measures like density and centralization characterize the entire system
#
# 4. **Actionable Insights**: Network analysis identifies specific interventions to improve the learning network
#
# ## Preview of Part 3
#
# In the next session, we'll explore:
# - Using network measures as predictors in regression models
# - Testing hypotheses about network formation
# - Network regression models
# - Distinguishing selection from influence effects
#
# ---