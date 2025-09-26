# Network Analysis for Healthcare Networks
## Project ECHO Workshop Materials

### Overview

This repository contains comprehensive materials for learning network analysis with applications to healthcare learning networks, specifically Project ECHO. The workshop moves from foundational network theory through hands-on analysis to practical applications.

### 🎯 Learning Objectives

By completing this workshop, you will be able to:

1. **Think relationally** - Understand how network perspective differs from traditional statistical approaches
2. **Measure networks** - Calculate and interpret centrality measures and network-level properties
3. **Visualize structure** - Create meaningful network visualizations that reveal hidden patterns
4. **Model network data** - Apply appropriate statistical models that account for network dependencies
5. **Drive action** - Translate network insights into strategic interventions

### 📚 Prerequisites

- **R knowledge**: Basic familiarity with R and RStudio
- **Statistics**: Understanding of linear regression and basic statistical concepts
- **No network experience required**: We build from fundamentals

### 💻 Setup Instructions

#### Option 1: Local RStudio Setup

```

1. Clone this repository:
```bash
git clone https://github.com/[username]/network-analysis-workshop.git
cd network-analysis-workshop
```

2. Install required packages:
```r
# Core packages
install.packages(c("tidyverse", "knitr", "rmarkdown"))

# Network analysis packages
install.packages("igraph")
install.packages("devtools")
devtools::install_github("netify-dev/netify")

# Additional packages for Part 3
install.packages(c("broom", "car", "gridExtra"))
```

3. Open RStudio and set working directory to the repository root

4. Open the R Markdown notebooks in the `code/` folder

#### Option 2: Google Colab Setup

1. Upload the notebook files to Google Colab
2. Install packages in first cell:
```r
install.packages(c("netify", "igraph", "tidyverse"))
```
3. Upload data files when prompted or mount Google Drive

### 📖 Workshop Modules

#### **Part 1: Network Theory & Foundations** 
*Format: Presentation slides*

- Basic network terminology (nodes, edges, paths)
- Types of ties and network structures
- Theoretical frameworks (weak ties, structural holes, small worlds)
- The paradigm shift from individual to relational thinking

#### **Part 2: Describing Networks** 
*Format: Interactive R notebook (`part2_basic_descriptives.Rmd`)*

- **Data Transformation**: Converting edge lists to adjacency matrices
- **Visualization**: Creating meaningful network plots with `igraph`
- **Centrality Measures**: 
  - Degree (connectivity)
  - Betweenness (brokerage)
  - Closeness (efficiency)
  - Eigenvector (influence)
- **Network Properties**: Density, centralization, clustering, path lengths
- **Practical Application**: Identifying vulnerable points and intervention opportunities

Key functions introduced:
```r
netify()           # Create network object
node_degree()      # Calculate degree centrality
node_betweenness() # Measure brokerage
plot()            # Visualize network
```

#### **Part 3: Statistical Models for Networks**
*Format: Interactive R notebook (`part3_models_for_net_inference.Rmd`)*

- **The Independence Problem**: Why standard regression fails for networks
- **Network Autocorrelation**: Testing and implications
- **Modeling Approaches**:
  - OLS with network measures (simple but flawed)
  - Network formation models (dyadic analysis)
  - AMEN framework (advanced network models)
- **Model Selection**: Choosing the right approach for your question
- **From Models to Action**: Translating findings to interventions

Key concepts covered:
- Residual correlation between connected nodes
- Dyadic data preparation
- Sender and receiver effects
- Latent homophily


### 🔑 Key Concepts & Terminology

| Term | Definition | ECHO Example |
|------|------------|--------------|
| **Node** | An entity in the network | A clinic or hub |
| **Edge** | A connection between nodes | Teaching relationship |
| **Degree** | Number of connections | How many partners a clinic has |
| **Betweenness** | Brokerage position | Clinics that bridge groups |
| **Centralization** | Network hierarchy | Hub-and-spoke structure |
| **Homophily** | Similar nodes connecting | Rural clinics partnering |

### 📊 Sample Analysis Workflow

```r
# 1. Load and transform data
edges <- read.csv("data/hub_and_spoke/echo_edges.csv")
nodes <- read.csv("data/hub_and_spoke/echo_nodes.csv")

# 2. Create network object
library(netify)
net <- netify(adjacency_matrix)

# 3. Calculate centrality
centrality <- node_degree(net)

# 4. Visualize
library(igraph)
plot(net, vertex.size = centrality)

# 5. Statistical model
model <- lm(participation ~ degree + controls, data = network_df)

# 6. Identify interventions
high_leverage <- nodes %>%
  filter(betweenness > median(betweenness),
         participation < median(participation))
```

### 🎓 Learning Paths

#### For Beginners:
1. Start with Part 1 slides for conceptual foundation
2. Work through Part 2 notebook step-by-step
3. Spend some time thinking about susbtantive applications before you dive into Part 3

#### For Those with Stats Background:
1. Skim Part 1, focus on network-specific concepts
2. Complete Part 2 for hands-on experience
3. Deep dive into Part 3 statistical models
4. Apply to your own data

#### For Quick Reference:
- Jump to specific sections in notebooks
- Use the function reference in appendices
- Review the model selection guide in Part 3

### 🤝 Contributing & Support

- **Issues**: Report bugs or suggest improvements via GitHub Issues
- **Questions**: Use Discussions tab for Q&A with other learners
- **Contributions**: Pull requests welcome for improvements

### 📚 Additional Resources

#### Essential Reading:
- Wasserman & Faust (1994) - *Social Network Analysis*
- Newman (2018) - *Networks: An Introduction*
- Luke & Harris (2007) - *Network Analysis in Public Health*

#### R Package Documentation:
- [netify documentation](https://netify-dev.github.io/netify/)
- [igraph documentation](https://igraph.org/r/)
- [Network visualization with R](http://kateto.net/network-visualization)

#### Online Courses:
- [Network Analysis in R (DataCamp)](https://www.datacamp.com/courses/network-analysis-in-r)
- [Social Network Analysis (Coursera)](https://www.coursera.org/learn/social-network-analysis)


### 📄 License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

### 🙏 Acknowledgments

- Project ECHO and Jessica Jones for inspiring this application
- The `netify` development team for modern network analysis tools
- Workshop participants for valuable feedback and questions
