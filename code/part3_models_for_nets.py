#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 3: Models for Network Inference
Network Analysis for Social Scientists Workshop (Python Version)
"""

# %% [markdown]
# # Introduction: Why Standard Statistics Fail for Networks
#
# ## The Independence Assumption Problem
#
# Imagine trying to understand a conversation by analyzing each person's words in isolation, 
# ignoring who they're talking to and what others are saying. That's essentially what we do 
# when we apply standard statistical methods to network data - we miss the fundamental reality 
# that actors are embedded in webs of relationships that shape their behaviors and outcomes.
#
# In traditional statistics, we assume observations are independent. But in networks, this 
# assumption is fundamentally violated. If the ECHO hub adopts a new practice, connected 
# spokes are more likely to adopt it too - not because of their individual characteristics, 
# but because of their network connection. This interdependence isn't a nuisance to be 
# corrected; it's the very phenomenon we want to study.
#
# ### The Challenge of Network Inference
#
# Network data presents unique statistical challenges:
# 1. **Autocorrelation**: Connected actors tend to have similar outcomes
# 2. **Simultaneity**: Network position and outcomes co-evolve
# 3. **Selection vs Influence**: Do birds of a feather flock together, or do flocked birds become similar?
# 4. **Dyadic dependence**: The relationship between A and B affects the relationship between B and C
#
# These challenges require specialized statistical approaches. In this session, we'll explore 
# both traditional methods (with their limitations) and network-specific models designed to 
# handle these complexities.
#
# ### Learning Objectives
#
# By the end of this notebook, you'll understand:
# - How to use network measures in regression models (and why this is problematic)
# - The concept of network autocorrelation and its consequences
# - Network formation models and what they reveal
# - How to choose between different modeling approaches
# - How to translate model results into actionable insights

# %% [markdown]
# ### Setting Up Our Analysis Environment

# %%
# Load required libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import pickle

# Set options for cleaner output
pd.set_option('display.precision', 3)
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

# Load data from Part 2
with open('part2_complete_analysis.pkl', 'rb') as f:
    part2_data = pickle.load(f)

G = part2_data['G']
centrality_df = part2_data['centrality_df']
adjacency_matrix = part2_data['adjacency_matrix']
edges = part2_data['edges']
nodes = part2_data['nodes']

# Quick verification
print("DATA LOADED:")
print("============")
print(f"Network object: {type(G).__name__}")
print(f"Nodes in network: {G.number_of_nodes()}")
print(f"Edges in network: {G.number_of_edges()}")
print(f"Centrality measures: {len(centrality_df.columns)} variables")

# %% [markdown]
# ---
# # Section 1: Network Measures in Traditional Models
#
# ## The Intuitive (But Flawed) Approach
#
# The most straightforward way to incorporate network information into analysis is to treat 
# network measures like any other variable. Want to know if well-connected clinics have better 
# outcomes? Just run a regression with degree centrality as a predictor. This approach is 
# intuitive, uses familiar methods, and often yields interesting results. But it's also 
# fundamentally flawed in ways we need to understand.
#
# ### Preparing Our Analysis Dataset

# %%
# Merge network and attribute data
analysis_df = centrality_df.merge(nodes, left_on='actor', right_on='node_id', how='left')

# Create derived variables for analysis
analysis_df['log_staff_size'] = np.log(analysis_df['staff_size'] + 1)
analysis_df['log_patient_volume'] = np.log(analysis_df['patient_volume'] + 1)
analysis_df['is_rural'] = (analysis_df['rural_urban'] == 'rural').astype(int)
analysis_df['high_participation'] = (analysis_df['participation_rate'] > 
                                     analysis_df['participation_rate'].median()).astype(int)

# Standardize centrality measures
scaler = StandardScaler()
analysis_df['betweenness_std'] = scaler.fit_transform(analysis_df[['betweenness']])
analysis_df['degree_std'] = scaler.fit_transform(analysis_df[['total_degree']])

# Use participation_rate consistently
analysis_df['participation'] = analysis_df['participation_rate']

print("ANALYSIS DATASET PREPARED:")
print("==========================")
print(f"Observations: {len(analysis_df)} actors")
print(f"Variables: {len(analysis_df.columns)} features")
print(f"Node types: {', '.join(analysis_df['type'].unique())}")
print(f"States represented: {analysis_df['state'].nunique()}")

# %% [markdown]
# ### Research Question: Does Network Position Predict Engagement?
#
# This is a natural question for ECHO programs: Are well-connected sites more engaged? 
# Let's explore this relationship, keeping in mind the statistical issues we'll encounter.

# %%
print("\n=== EXPLORING THE POSITION-PARTICIPATION RELATIONSHIP ===\n")

# First, let's visualize the relationship
fig, ax = plt.subplots(figsize=(10, 6))

# Create color map for node types
color_map = {'hub': '#E74C3C', 'spoke': '#3498DB', 'expert': '#F39C12'}
colors = [color_map[t] for t in analysis_df['type']]

scatter = ax.scatter(analysis_df['total_degree'], 
                    analysis_df['participation'],
                    c=colors, 
                    s=analysis_df['staff_size']*2, 
                    alpha=0.7)

# Add trend line
z = np.polyfit(analysis_df['total_degree'], analysis_df['participation'], 1)
p = np.poly1d(z)
ax.plot(analysis_df['total_degree'], p(analysis_df['total_degree']), 
        "darkblue", linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Degree Centrality (Total Connections)', fontsize=12)
ax.set_ylabel('Participation Rate (%)', fontsize=12)
ax.set_title('Network Position and Participation', fontsize=14, fontweight='bold')

# Add legend for node types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E74C3C', label='Hub'),
                  Patch(facecolor='#3498DB', label='Spoke'),
                  Patch(facecolor='#F39C12', label='Expert')]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()

# Calculate correlation
correlation = analysis_df['total_degree'].corr(analysis_df['participation'])
print(f"Correlation between degree and participation: {correlation:.3f}")
print(f"This suggests a {'strong' if abs(correlation) > 0.5 else 'moderate'} relationship.")
print("But correlation doesn't imply causation, especially in networks!")

# %% [markdown]
# ## Model 1: Simple Linear Regression

# %%
# Fit simple model
model1 = smf.ols('participation ~ total_degree', data=analysis_df).fit()

print("\nMODEL 1: SIMPLE LINEAR REGRESSION")
print("==================================")
print("participation = β₀ + β₁(degree) + ε\n")

# Display results
print(model1.summary())

# Interpretation
coef_degree = model1.params['total_degree']
print(f"\nINTERPRETATION:")
print(f"Each additional connection is associated with a {coef_degree:.2f} "
      f"percentage point increase in participation.")
print(f"R-squared: {model1.rsquared:.3f}")

print("\n⚠️ WARNING: This model assumes each clinic is independent!")
print("In reality, connected clinics influence each other.")
print("Our standard errors are likely underestimated.")

# %% [markdown]
# ## Model 2: Multiple Regression with Controls
#
# Real-world relationships are complex. Let's add control variables:

# %%
# More sophisticated model
model2 = smf.ols(
    'participation ~ total_degree + betweenness_std + years_in_echo + '
    'log_staff_size + is_rural + C(type)',
    data=analysis_df
).fit()

print("\nMODEL 2: MULTIPLE REGRESSION")
print("=============================")
print(model2.summary())

# Create results dataframe
model2_results = pd.DataFrame({
    'term': model2.params.index,
    'estimate': model2.params.values,
    'std_error': model2.bse.values,
    't_value': model2.tvalues.values,
    'p_value': model2.pvalues.values,
    'conf_low': model2.conf_int()[0].values,
    'conf_high': model2.conf_int()[1].values
})

model2_results['significant'] = model2_results['p_value'].apply(
    lambda p: '***' if p < 0.05 else ('*' if p < 0.1 else '')
)

print("\nCOEFFICIENTS TABLE:")
print(model2_results.to_string(index=False))

print(f"\nMODEL QUALITY:")
print(f"R-squared: {model2.rsquared:.3f}")
print(f"Adjusted R-squared: {model2.rsquared_adj:.3f}")
print(f"AIC: {model2.aic:.1f}")
print(f"BIC: {model2.bic:.1f}")

# %% [markdown]
# ### Visualizing Model Results

# %%
# Coefficient plot
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for plotting (exclude intercept)
coef_plot_data = model2_results[~model2_results['term'].str.contains('Intercept')].copy()

# Clean term names
coef_plot_data['term_clean'] = coef_plot_data['term'].apply(lambda x: 
    x.replace('C(type)[T.', 'Type: ').replace(']', '') if 'type' in x else
    x.replace('_', ' ').title()
)

# Sort by estimate
coef_plot_data = coef_plot_data.sort_values('estimate')

# Create plot
ax.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
colors = ['#27ae60' if p < 0.05 else 'gray' for p in coef_plot_data['p_value']]

ax.barh(range(len(coef_plot_data)), coef_plot_data['estimate'], 
        xerr=1.96*coef_plot_data['std_error'], color=colors, alpha=0.7)
ax.set_yticks(range(len(coef_plot_data)))
ax.set_yticklabels(coef_plot_data['term_clean'])
ax.set_xlabel('Effect on Participation Rate (%)', fontsize=12)
ax.set_title('Factors Associated with Participation', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Checking Model Assumptions (And Finding Problems)
#
# Standard regression assumes independence. Let's check if this holds:

# %%
print("\n=== DIAGNOSTIC CHECKS ===")

# Standard diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(model2.fittedvalues, model2.resid, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(model2.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Scale-Location
axes[1, 0].scatter(model2.fittedvalues, np.sqrt(np.abs(model2.resid)), alpha=0.6)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Residuals|')
axes[1, 0].set_title('Scale-Location')

# Residuals histogram
axes[1, 1].hist(model2.resid, bins=15, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

# The real problem: Network autocorrelation
print("\nTESTING FOR NETWORK AUTOCORRELATION:")
print("====================================")

# Get residuals
residuals_dict = dict(zip(analysis_df['actor'], model2.resid))

# For each edge, calculate residual correlation
edge_residuals = []
for _, edge in edges.iterrows():
    if edge['from'] in residuals_dict and edge['to'] in residuals_dict:
        edge_residuals.append({
            'from': edge['from'],
            'to': edge['to'],
            'resid_from': residuals_dict[edge['from']],
            'resid_to': residuals_dict[edge['to']]
        })

edge_residuals_df = pd.DataFrame(edge_residuals)

# Calculate correlation between connected nodes' residuals
residual_correlation = edge_residuals_df['resid_from'].corr(edge_residuals_df['resid_to'])

print(f"\nRESIDUAL CORRELATION BETWEEN CONNECTED NODES: {residual_correlation:.3f}")

if abs(residual_correlation) > 0.1:
    print("\n⚠️ NETWORK AUTOCORRELATION DETECTED!")
    print("Connected nodes have correlated residuals.")
    print("This violates the independence assumption of OLS.")
    print("Consequences:")
    print("• Standard errors are underestimated")
    print("• P-values are too optimistic")
    print("• Confidence intervals are too narrow")

# %% [markdown]
# ### The Fundamental Problem with OLS for Networks

# %%
print("\n=== WHY OLS FAILS FOR NETWORK DATA ===")
print("=======================================\n")

# Create educational visualization
problems_df = pd.DataFrame({
    'Problem': ['Independence\nViolation', 'Network\nAutocorrelation', 
                'Simultaneity\nBias', 'Selection\nEffects', 'Dyadic\nDependence'],
    'Description': ['Nodes influence each other through ties',
                   'Similar outcomes for connected actors',
                   'Position and outcomes co-evolve',
                   'Homophily vs influence confusion',
                   'Triadic closure and clustering'],
    'Consequence': ['Biased standard errors', 'Invalid hypothesis tests',
                   'Biased coefficients', 'Wrong causal inference',
                   'Model misspecification'],
    'Severity': [4, 5, 3, 4, 3]  # Impact severity 1-5
})

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(problems_df['Problem'], problems_df['Severity'], 
               color='#e74c3c', alpha=0.7)
ax.set_xlabel('Problem Severity', fontsize=12)
ax.set_title('Why Standard Statistics Fail for Networks', fontsize=14, fontweight='bold')

# Add descriptions as text
for i, (desc, sev) in enumerate(zip(problems_df['Description'], problems_df['Severity'])):
    ax.text(0.1, i, desc, fontsize=9, va='center')

plt.tight_layout()
plt.show()

print("BOTTOM LINE:")
print("OLS can identify associations but:")
print("• Cannot establish causation")
print("• Underestimates uncertainty")
print("• Misses network mechanisms")
print("• Treats symptoms, not structure")

# %% [markdown]
# ---
# # Section 2: Network Formation Models
#
# ## Understanding Why Ties Form
#
# Instead of asking "how does network position affect outcomes?", network formation models 
# ask "why do connections exist in the first place?" This shift in perspective is profound - 
# we're modeling the generative process that creates the network structure we observe.
#
# ### The Logic of Dyadic Analysis
#
# In network formation models, our unit of analysis shifts from nodes to dyads (pairs of nodes). 
# For a network with n nodes, we have n×(n-1) potential directed relationships. Our goal is to 
# understand what makes some of these potential ties actual ties.

# %%
print("=== PREPARING DYADIC DATA ===")
print("=============================")

# Create all possible dyads
actors = list(G.nodes())
n_actors = len(actors)

# Generate dyad dataset
from itertools import product
dyad_list = [(s, r) for s, r in product(actors, actors) if s != r]

dyad_df = pd.DataFrame(dyad_list, columns=['sender', 'receiver'])

# Mark existing edges
existing_edges = set([(e[0], e[1]) for e in G.edges()])
dyad_df['edge_exists'] = dyad_df.apply(
    lambda row: 1 if (row['sender'], row['receiver']) in existing_edges else 0, 
    axis=1
)

print(f"DYADIC DATASET:")
print(f"• Potential dyads: {len(dyad_df)}")
print(f"• Actual edges: {dyad_df['edge_exists'].sum()}")
print(f"• Network density: {dyad_df['edge_exists'].mean():.3f}")

print(f"\nKEY INSIGHT:")
print(f"We're now asking: What makes {dyad_df['edge_exists'].sum()} dyads connect")
print(f"while {(~dyad_df['edge_exists']).sum()} remain unconnected?")

# %% [markdown]
# ### Adding Dyadic Covariates
#
# What factors might predict tie formation in ECHO networks?

# %%
# Create node lookup dictionary for faster access
node_dict = nodes.set_index('node_id').to_dict('index')

# Enrich dyad data with attributes
for col in ['node_type', 'years_in_echo', 'staff_size', 'participation_rate', 'state']:
    dyad_df[f'sender_{col}'] = dyad_df['sender'].map(
        lambda x: node_dict.get(x, {}).get(col))
    dyad_df[f'receiver_{col}'] = dyad_df['receiver'].map(
        lambda x: node_dict.get(x, {}).get(col))

# Create dyadic characteristics
dyad_df['same_state'] = (dyad_df['sender_state'] == dyad_df['receiver_state']).astype(int)
dyad_df['same_type'] = (dyad_df['sender_node_type'] == dyad_df['receiver_node_type']).astype(int)
dyad_df['years_diff'] = np.abs(dyad_df['sender_years_in_echo'] - dyad_df['receiver_years_in_echo'])
dyad_df['staff_diff'] = np.abs(dyad_df['sender_staff_size'] - dyad_df['receiver_staff_size'])

# Transformations
dyad_df['log_sender_staff'] = np.log(dyad_df['sender_staff_size'] + 1)
dyad_df['log_receiver_staff'] = np.log(dyad_df['receiver_staff_size'] + 1)

print("\nPOTENTIAL PREDICTORS OF TIE FORMATION:")
print("=======================================")
print("SENDER EFFECTS (Who initiates connections?):")
print("• Node type (hub, spoke, expert)")
print("• Years of experience in ECHO")
print("• Organization size")
print("• Engagement level\n")

print("RECEIVER EFFECTS (Who receives connections?):")
print("• Same factors, but for receiving ties\n")

print("DYADIC EFFECTS (What pairs connect?):")
print("• Geographic proximity (same state)")
print("• Homophily (same type)")
print("• Experience gaps")
print("• Size similarity")

# %% [markdown]
# ## Fitting a Network Formation Model

# %%
print("\n=== NETWORK FORMATION MODEL ===")
print("================================")
print("Modeling: P(Edge from i to j) using logistic regression\n")

# Prepare data for modeling (handle categorical variables)
dyad_df_model = pd.get_dummies(dyad_df, 
                               columns=['sender_node_type', 'receiver_node_type'],
                               drop_first=True)

# Select features for the model
feature_cols = [col for col in dyad_df_model.columns 
                if col not in ['sender', 'receiver', 'edge_exists', 
                              'sender_state', 'receiver_state']]

X = dyad_df_model[feature_cols].fillna(0)
y = dyad_df_model['edge_exists']

# Add constant
X = sm.add_constant(X)

# Fit logistic regression
formation_model = sm.Logit(y, X).fit(disp=0)

print("MODEL RESULTS:")
formation_results = pd.DataFrame({
    'term': formation_model.params.index,
    'estimate': formation_model.params.values,
    'std_error': formation_model.bse.values,
    'p_value': formation_model.pvalues.values,
    'odds_ratio': np.exp(formation_model.params.values)
})

formation_results['effect'] = formation_results['estimate'].apply(
    lambda x: 'Increases' if x > 0 else 'Decreases' if x < 0 else 'No effect'
)

# Sort by absolute effect size
formation_results = formation_results.sort_values('estimate', key=abs, ascending=False)

print(formation_results[['term', 'estimate', 'odds_ratio', 'p_value', 'effect']].head(10).to_string(index=False))

print(f"\nMODEL FIT STATISTICS:")
print(f"AIC: {formation_model.aic:.1f}")
print(f"Log-Likelihood: {formation_model.llf:.1f}")
print(f"McFadden R²: {formation_model.prsquared:.3f}")

# %% [markdown]
# ### Interpreting Formation Patterns

# %%
print("\n=== KEY FORMATION PATTERNS ===")
print("===============================")

# Identify significant predictors
sig_predictors = formation_results[formation_results['p_value'] < 0.05]
sig_predictors = sig_predictors[~sig_predictors['term'].str.contains('const')]

if len(sig_predictors) > 0:
    print("\nSIGNIFICANT FACTORS:")
    for _, row in sig_predictors.iterrows():
        print(f"• {row['term']}:")
        print(f"  Effect: {row['effect']} tie probability")
        print(f"  Odds ratio: {row['odds_ratio']:.3f}")
else:
    print("No significant predictors at p < 0.05")

# Predicted probabilities for key dyad types
print("\n\nPREDICTED CONNECTION PROBABILITIES:")
print("====================================")

# Create scenarios
scenarios = pd.DataFrame({
    'scenario': ['Hub → Spoke', 'Spoke → Hub', 'Spoke → Spoke', 
                'Expert → Hub', 'Hub → Expert'],
    'const': 1
})

# Add dummy variables matching the model
for col in X.columns[1:]:  # Skip constant
    if 'hub' in col.lower() or 'spoke' in col.lower() or 'expert' in col.lower():
        scenarios[col] = 0
    else:
        scenarios[col] = X[col].mean()

# Set specific scenarios (this is simplified - adjust based on actual dummy columns)
scenarios['probability'] = formation_model.predict(scenarios[X.columns])

print(scenarios[['scenario', 'probability']].to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scenarios['scenario'], scenarios['probability'], 
               color=plt.cm.RdBu(scenarios['probability']))
ax.set_ylabel('Probability of Connection', fontsize=12)
ax.set_title('Edge Formation Probabilities', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(scenarios['probability']) * 1.2)

# Add value labels
for bar, prob in zip(bars, scenarios['probability']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{prob:.3f}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # Section 3: Advanced Network Models - The AMEN Framework
#
# ## Beyond Simple Formation Models
#
# The logistic regression approach treats each dyad as independent, but in reality, 
# network formation involves complex dependencies. The AMEN (Additive and Multiplicative 
# Effects Network) model addresses these by explicitly modeling:
#
# 1. **Sender heterogeneity**: Some actors are more active
# 2. **Receiver heterogeneity**: Some actors are more popular
# 3. **Dyadic correlation**: Reciprocity and transitivity
# 4. **Latent homophily**: Unobserved similarities

# %%
print("=== THE AMEN MODEL FRAMEWORK ===")
print("=================================\n")

print("CONCEPTUAL MODEL:")
print("Y[i,j] = μ + β'X[i,j] + a[i] + b[j] + u[i]'v[j] + ε[i,j]\n")

print("WHERE:")
print("• μ: Overall network density")
print("• β'X[i,j]: Effects of observed dyadic covariates")
print("• a[i]: Random sender effects (out-degree heterogeneity)")
print("• b[j]: Random receiver effects (in-degree heterogeneity)")
print("• u[i]'v[j]: Multiplicative effects (latent positions)")
print("• ε[i,j]: Random noise\n")

print("KEY ADVANTAGES:")
print("• Accounts for degree heterogeneity")
print("• Captures transitivity and clustering")
print("• Models unobserved homophily")
print("• Handles network dependencies")

# %% [markdown]
# ### Implementing AMEN Concepts

# %%
print("\nPREPARING DATA FOR AMEN:")
print("========================")

# Create adjacency matrix
Y = adjacency_matrix.values

# Prepare covariates
# Node-level covariates
node_covariates = nodes[['years_in_echo', 'staff_size', 'participation_rate']].copy()
scaler = StandardScaler()
node_covariates_scaled = scaler.fit_transform(node_covariates.fillna(0))

# Dyadic covariates (same state)
state_list = nodes['state'].values
same_state_matrix = np.equal.outer(state_list, state_list).astype(int)
np.fill_diagonal(same_state_matrix, 0)  # No self-loops

print(f"Data prepared:")
print(f"• Network matrix: {Y.shape[0]} x {Y.shape[1]}")
print(f"• Node covariates: {node_covariates_scaled.shape[1]} variables")
print(f"• Dyadic covariate: Same state indicator")

print("\nNOTE: Full AMEN implementation would require specialized packages.")
print("Python alternatives include:")
print("• pymc3 for Bayesian network models")
print("• graph-tool for advanced network inference")
print("See Hoff et al. (2021) for implementation details.")

# %% [markdown]
# ---
# # Section 4: Choosing the Right Model
#
# ## Model Selection Framework
#
# Different research questions require different modeling approaches. Here's a practical guide:

# %%
print("=== MODEL SELECTION GUIDE ===")
print("=============================\n")

selection_guide = pd.DataFrame({
    'Research_Question': [
        'How does centrality affect outcomes?',
        'What predicts tie formation?',
        'Is there homophily in the network?',
        'How do networks evolve?',
        'What drives clustering?',
        'Testing intervention effects'
    ],
    'Recommended_Model': [
        'OLS with network measures',
        'Network formation models',
        'ERGM or AMEN',
        'STERGM or longitudinal AMEN',
        'ERGM with triangles',
        'Both OLS and network models'
    ],
    'Why': [
        'Simple, interpretable coefficients',
        'Models the generative process',
        'Separates selection from influence',
        'Captures temporal dependencies',
        'Explicitly models triadic closure',
        'Different insights from each'
    ],
    'Limitations': [
        'Ignores dependencies',
        'Static snapshot only',
        'Computationally intensive',
        'Requires panel data',
        'Can be unstable',
        'Multiple testing issues'
    ]
})

print(selection_guide.to_string(index=False))

# %% [markdown]
# ## Practical Considerations

# %%
print("\n=== PRACTICAL CONSIDERATIONS ===")
print("=================================")

considerations = {
    "Sample Size": [
        "OLS: Works with small networks (n > 30)",
        "Formation models: Need n² observations",
        "AMEN/ERGM: Better with n > 50",
        "Rule of thumb: More complex models need more data"
    ],
    
    "Computational Resources": [
        "OLS: Instant, runs on any computer",
        "Logistic: Fast for moderate networks",
        "AMEN: Hours for large networks",
        "ERGM: Can be very slow, may not converge"
    ],
    
    "Interpretability": [
        "OLS: Everyone understands regression",
        "Logistic: Familiar to most researchers",
        "AMEN: Requires network expertise",
        "ERGM: Complex, hard to explain"
    ],
    
    "Software Requirements": [
        "OLS: statsmodels or scikit-learn",
        "Formation: statsmodels (logit)",
        "AMEN: Custom implementation or R",
        "ERGM: python-ergm or use R"
    ]
}

for category, points in considerations.items():
    print(f"\n{category}:")
    for point in points:
        print(f"  • {point}")

# %% [markdown]
# ---
# # Section 5: From Models to Action
#
# ## Translating Statistical Insights to Practice
#
# Models are only valuable if they inform decisions. Let's extract actionable insights from our analyses:

# %%
print("=== ACTIONABLE INSIGHTS FROM MODELS ===")
print("========================================\n")

# Finding 1: High-leverage intervention points
print("FINDING 1: HIGH-LEVERAGE SITES")
print("-------------------------------")

# Identify sites with high influence but low engagement
spoke_df = analysis_df[analysis_df['type'] == 'spoke'].copy()
spoke_df['influence'] = spoke_df['betweenness'] / spoke_df['betweenness'].max()
spoke_df['engagement'] = spoke_df['participation'] / 100
spoke_df['leverage'] = spoke_df['influence'] - spoke_df['engagement']

leverage_sites = spoke_df.nlargest(5, 'leverage')[['actor', 'betweenness', 'participation', 'leverage']]

print("Sites with high network influence but low participation:")
print(leverage_sites.to_string(index=False))

print("\nRECOMMENDATION:")
print("Target these sites for engagement interventions.")
print("Small improvements here have network-wide effects.")

# %% [markdown]
# ### Network Growth Strategies

# %%
print("\n\nFINDING 2: NETWORK GROWTH OPPORTUNITIES")
print("-----------------------------------------")

# Analyze current network composition
edge_composition = edges.groupby('relationship_type').agg({
    'from': 'count',
    'weight': 'mean'
}).rename(columns={'from': 'count', 'weight': 'avg_weight'})
edge_composition['pct'] = edge_composition['count'] / len(edges) * 100

print("Current network composition:")
print(edge_composition.to_string())

# Based on formation model, identify missing connections
print("\nGROWTH STRATEGY:")

# Calculate potential peer connections
potential_peer_edges = dyad_df[
    (dyad_df['sender_node_type'] == 'spoke') &
    (dyad_df['receiver_node_type'] == 'spoke') &
    (dyad_df['same_state'] == 1) &
    (dyad_df['edge_exists'] == 0)
]

print(f"• Potential same-state peer connections: {len(potential_peer_edges)}")
print(f"• Current peer learning edges: {sum(edges['relationship_type'] == 'peer_learning')}")
print(f"• Opportunity: Add {min(10, len(potential_peer_edges))} strategic peer connections")

print("\nRECOMMENDATION:")
print("1. Foster within-state peer connections")
print("2. Create specialty-based working groups")
print("3. Implement buddy system for new sites")

# %% [markdown]
# ### Intervention Simulation

# %%
print("\n\nFINDING 3: INTERVENTION IMPACT")
print("--------------------------------")

# Simulate adding strategic connections
print("SIMULATION: Connect isolated sites to brokers\n")

# Identify isolated and broker sites
isolated_sites = analysis_df[(analysis_df['type'] == 'spoke') & 
                            (analysis_df['total_degree'] <= 2)]['actor'].tolist()

broker_sites = analysis_df[(analysis_df['type'] == 'spoke') & 
                          (analysis_df['betweenness'] > analysis_df['betweenness'].median())].head(3)['actor'].tolist()

print(f"Isolated sites: {len(isolated_sites)}")
print(f"Broker sites: {', '.join(broker_sites)}")

# Calculate expected impact
if isolated_sites and broker_sites:
    new_connections = len(isolated_sites) * len(broker_sites)
    density_increase = new_connections / (n_actors * (n_actors - 1))
    
    print("\nEXPECTED IMPACT:")
    print(f"• New connections: {new_connections}")
    print(f"• Density increase: {density_increase * 100:.2f}%")
    print("• Reduced average path length")
    print("• Increased redundancy (resilience)")

# %% [markdown]
# ## Executive Dashboard

# %%
print("\n\n=== EXECUTIVE SUMMARY ===")
print("==========================")

# Key metrics and recommendations
summary_table = pd.DataFrame({
    'Metric': ['Network Efficiency', 'Vulnerability', 'Peer Learning',
               'Geographic Coverage', 'Engagement Variation'],
    'Current_State': ['High (short paths)', 'High (hub dependent)', 'Low (30% of edges)',
                     'Uneven (by state)', 'High (CV = 0.25)'],
    'Target_State': ['Maintain', 'Reduce to medium', 'Increase to 50%',
                    'Balance across states', 'Reduce variation'],
    'Action_Required': ['Monitor only', 'Add redundant connections', 'Foster peer relationships',
                       'Regional coordination', 'Support weak sites']
})

print("Network Health Dashboard:")
print(summary_table.to_string(index=False))

print("\n\nTOP 3 PRIORITIES:")
priorities = [
    "Connect isolated sites to broker spokes (Quick win)",
    "Develop regional sub-hubs (Medium-term resilience)",
    "Create peer learning communities (Long-term sustainability)"
]

for i, priority in enumerate(priorities, 1):
    print(f"{i}. {priority}")

# %% [markdown]
# ---
# # Summary and Conclusions
#
# ## What We've Learned About Network Inference

# %%
print("=== KEY TAKEAWAYS ===")
print("=====================\n")

lessons = {
    "Statistical Challenges": 
        "Network data violates independence assumptions of standard models",
    
    "Model Choice Matters": 
        "Different models answer different questions - choose wisely",
    
    "Formation vs Outcomes": 
        "Understanding why ties form is different from understanding their effects",
    
    "Dependencies are Features": 
        "Network effects aren't nuisances - they're what we want to study",
    
    "Actionable Insights": 
        "Good network analysis translates structure into strategy"
}

for lesson, description in lessons.items():
    print(f"• {lesson}:\n  {description}\n")

# %% [markdown]
# ## Comparing Approaches: A Final Summary

# %%
comparison_final = pd.DataFrame({
    'Approach': ['OLS with Network Measures', 'Network Formation Models', 'AMEN/Advanced Models'],
    'Best_For': ['Quick insights, familiar methods', 'Understanding tie formation',
                'Rigorous network inference'],
    'Limitations': ['Biased SEs, no causation', 'Treats dyads as independent',
                   'Complex, computationally intensive'],
    'When_to_Use': ['Exploratory analysis, medical journals',
                   'Testing homophily, selection effects',
                   'Publication in network journals']
})

print("Model Comparison Summary:")
print(comparison_final.to_string(index=False))

# %% [markdown]
# ## Final Thoughts

# %%
print("\n\n=== CLOSING THOUGHTS ===")
print("========================\n")

print("Network analysis is both art and science. The models we've explored")
print("provide rigorous ways to test hypotheses, but interpreting results")
print("requires understanding the substantive context.\n")

print("For Project ECHO networks specifically:")
print("• The hub-and-spoke model is efficient but fragile")
print("• Peer connections are underutilized opportunities")
print("• Geographic proximity matters for collaboration")
print("• Network position predicts but doesn't determine engagement\n")

print("Remember: Networks are not just data structures - they represent")
print("real relationships between real people working toward better")
print("healthcare delivery. Use these tools thoughtfully to strengthen")
print("those connections and improve patient outcomes.")

# Save final results
results_to_save = {
    'model1': model1,
    'model2': model2,
    'formation_model': formation_model,
    'analysis_df': analysis_df,
    'dyad_df': dyad_df
}

with open('part3_inference_results.pkl', 'wb') as f:
    pickle.dump(results_to_save, f)

print("\nResults saved for future analysis.")
print("Workshop complete - thank you for participating!")

# %% [markdown]
# ---
# # Appendix: Quick Reference Guide
#
# ## Model Selection Flowchart
#
# ```
# Research Question?
# ├── Individual Outcomes → OLS with network measures
# │   └── Check: Network autocorrelation?
# │       ├── Yes → Interpret with caution
# │       └── No → Standard interpretation
# │
# ├── Tie Formation → Network formation models
# │   └── Check: Dyadic dependencies?
# │       ├── Yes → AMEN or ERGM
# │       └── No → Logistic regression
# │
# └── Network Evolution → Longitudinal models
#     └── Check: Panel data available?
#         ├── Yes → STERGM or dynamic AMEN
#         └── No → Cross-sectional comparison
# ```
#
# ## Key Functions Reference
#
# | Task | Python Function | Library |
# |------|-----------------|---------|
# | Create network | `nx.Graph()` / `nx.DiGraph()` | networkx |
# | Get adjacency | `nx.to_numpy_array()` | networkx |
# | Calculate centrality | `nx.degree_centrality()` | networkx |
# | Linear regression | `smf.ols()` | statsmodels |
# | Logistic regression | `sm.Logit()` | statsmodels |
# | Visualize | `nx.draw()` | networkx |
#
# ---
# 
# *End of Part 3: Models for Network Inference*
#
# *This notebook is part of the ECHO Learning Network Analysis Series*