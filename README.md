# DishFlow Optimization Network — Snooze-Style Brunch Edition v3

### Flow-Aware Machine Learning for Restaurant Throughput Protection

A neurodivergent-first teaching notebook that demonstrates how machine learning, operations research, and human-centered behavior design can be combined to protect restaurant flow during high-pressure brunch service.

DishFlow models the dish pit not as a cleaning station, but as a critical operational control point whose decisions directly affect plating speed, food quality, runner efficiency, and customer wait times.

The notebook is written from the perspective of a **Head Dishwasher** responsible for maintaining clean-dish availability during an overloaded brunch rush.

---

## Core Idea

Traditional dishwashing metrics focus on cleanliness.

DishFlow focuses on flow.

The central hypothesis is:

> Poor scraping creates downstream delay.

That delay propagates through the restaurant:

```text
Poor Scraping
      ↓
Longer Spray Time
      ↓
Dish-Pit Backlog
      ↓
Plate Shortage
      ↓
Cooked Food Waits
      ↓
Food Quality Drops
      ↓
Runners Wait
      ↓
Guests Wait Longer
```

The objective is not simply to classify dirty dishes.

The objective is to learn how to protect restaurant throughput.

---

# Business Problem

During peak brunch service, restaurants frequently experience:

- Plate shortages
- Rewash overload
- Dish-pit congestion
- Food-ready bottlenecks
- Delayed table turns
- Increased guest wait times

DishFlow investigates:

> How can scraping, spraying, stacking, racking, and running dishes be optimized to reduce downstream food wait?

---

# Learning Objectives

This notebook teaches:

- CRISP-DM methodology
- Synthetic data generation
- Operations-aware feature engineering
- Neural-network regression
- Neural-network classification
- Human-in-the-loop system design
- Restaurant throughput analysis
- Flow-state optimization

The notebook is designed for learners who understand restaurant operations better than machine learning and uses operational language throughout.

---

# System Architecture

DishFlow consists of two neural networks and one behavioral feedback layer.

## 1. Scrape Delay Network

Predicts:

- Scrape Quality Score
- Expected Downstream Delay
- Dish Scrape Olympics Score

Purpose:

```text
Dirty Dish
      ↓
Feature Extraction
      ↓
Scrape Delay Network
      ↓
Predicted Operational Cost
```

---

## 2. Dish Flow Action Network

Predicts the next best operational action.

Possible actions:

| Action | Description |
|----------|-------------|
| Run Plates | Move clean plates to plating line immediately |
| Rescrape | Improve scrape quality before spraying |
| Spray/Rack | Continue normal dish flow |
| Batch Stack | Group similar items under rack pressure |
| Defer Cups | Prioritize plates during shortages |
| Protect Rewash Capacity | Prevent rewash overload |

Purpose:

```text
Current Dish-Pit State
           ↓
Dish Flow Action Network
           ↓
Recommended Action
```

---

## 3. Dish Scrape Olympics

A behavioral reinforcement layer designed to improve human performance.

Rather than punishment-based monitoring, DishFlow explores:

- Team recognition
- Skill visibility
- Positive reinforcement
- Friendly competition
- Flow protection awareness

Measured through:

- Scrape Quality
- Rewash Risk
- Downstream Delay Reduction
- Team Contribution

---

# CRISP-DM Framework

The notebook follows the CRISP-DM lifecycle.

## 1. Business Understanding

Defines the restaurant bottleneck:

```text
Clean plate availability
```

rather than:

```text
Dish cleanliness alone
```

---

## 2. Data Understanding

Models operational variables including:

- Dish type
- Residue type
- Residue coverage
- Stuck-food severity
- Scrape completeness
- Hot-food demand
- Dish-pit backlog
- Rack availability
- Worker fatigue
- Rush intensity

---

## 3. Data Preparation

Generates synthetic restaurant data representing:

- High-volume brunch operations
- Plate-heavy service
- Weekend pressure
- Mixed residue conditions
- Resource constraints

---

## 4. Modeling

Two neural-network models are trained:

### Regression Network

Predicts:

- Scrape Quality
- Downstream Extra Seconds
- Olympics Score

Architecture:

```text
MLPRegressor
128 → 96 → 48
ReLU
Adam Optimizer
Early Stopping
```

### Classification Network

Predicts:

- Optimal Dish-Flow Action

Architecture:

```text
MLPClassifier
128 → 96 → 48
ReLU
Adam Optimizer
Early Stopping
```

---

## 5. Evaluation

### Scrape Delay Network

| Metric | Result |
|----------|----------|
| Scrape Quality MAE | 3.06 |
| Extra Seconds MAE | 1.90 |
| Olympics Score MAE | 2.05 |
| Scrape Quality R² | 0.921 |
| Extra Seconds R² | 0.975 |
| Olympics Score R² | 0.972 |

### Dish Flow Action Network

| Metric | Result |
|----------|----------|
| Action Accuracy | 95.63% |

These results demonstrate successful learning of the synthetic operational policy.

They do **not** demonstrate real-world restaurant performance.

---

## 6. Deployment Thinking

A production implementation would likely include:

### Perception Layer

- Camera vision
- Residue detection
- Scrape assessment

### Flow-State Layer

- Dish-pit backlog
- Clean plate inventory
- Hot-food queue
- Rack capacity
- Staffing conditions

### Recommendation Layer

- Action suggestions
- Flow warnings
- Team scoreboards
- Coaching feedback

The system is intended to augment the Head Dishwasher, not replace them.

---

# Synthetic Dataset

The notebook generates:

```text
12,000 synthetic dish-flow events
30 engineered features
3 regression targets
1 action-class target
```

Dish distribution emphasizes brunch operations:

- Large Entree Plates
- Pancake Plates
- Bowls
- Ramekins
- Coffee Cups
- Silverware Batches

Residue distribution includes:

- Egg Yolk
- Syrup
- Starch
- Hollandaise
- Grease
- Mixed Brunch Residue
- Light Crumbs

---

# Why Brunch?

Brunch creates unusual dish-flow challenges.

Examples:

- Egg residue hardens quickly
- Syrup creates sticky films
- Mixed brunch plates require more processing
- Hot breakfast food loses quality rapidly while waiting

These dynamics make dish-flow decisions operationally significant.

---

# Real-World Data Requirements

This notebook intentionally uses synthetic data.

To build a production system, real observations would be required.

Minimum inputs:

- Plate images or video
- Dish type labels
- Residue labels
- Scrape quality ratings
- Spray time measurements
- Rewash outcomes
- Inventory timestamps
- Ticket timestamps
- Food-ready timestamps
- Food-plated timestamps
- Food-run timestamps
- Staffing information

Suggested starting dataset:

| Dataset Size | Capability |
|-------------|------------|
| 1,000–2,000 Events | Early Prototype |
| 5,000–10,000 Events | Useful Supervised Model |
| 20,000+ Events | Strong Operational Model |

---

# Ethical Design Principles

DishFlow explicitly rejects punitive worker surveillance.

Recommended safeguards:

- Face blurring
- Customer privacy protection
- Team-level feedback
- Consent-based collection
- Limited retention
- Separation from disciplinary systems

The goal is operational learning and flow improvement, not worker monitoring.

---

# Repository Contents

```text
dishflow_snooze_golden_v3/
│
├── README_DISHFLOW_SNOOZE_GOLDEN_V3.md
└── dishflow_snooze_golden_neurodivergent_first_v3.ipynb
```

---

# Running the Notebook

## Requirements

```bash
Python 3.10+
Jupyter Notebook
```

## Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Launch

```bash
jupyter notebook
```

Open:

```text
dishflow_snooze_golden_neurodivergent_first_v3.ipynb
```

Run all cells sequentially.

Do not skip cells.

---

# Truth Boundary

## Exists

- Synthetic Snooze-style brunch data generator
- Scrape Delay Network
- Dish Flow Action Network
- Dish Scrape Olympics reward system
- Evaluation metrics
- Recommendation engine
- Real-data implementation roadmap

## Does Not Exist

- Real Snooze operational data
- POS integrations
- Production deployment
- Live camera feeds
- Restaurant telemetry

## Unverified

- Real-world wait-time reduction
- Food quality improvements
- Throughput gains in live service
- Worker adoption outcomes

---

# Research Positioning

DishFlow sits at the intersection of:

- Machine Learning
- Operations Research
- Restaurant Systems Engineering
- Human Factors
- Behavioral Design
- Human-AI Collaboration

It demonstrates how machine learning can move beyond prediction toward protecting flow within real operational systems.

---

## Author

**Corey Alejandro**

AI Safety Research Engineer • Instructional Systems Designer • Operations Systems Researcher

GitHub: https://github.com/coreyalejandro

Portfolio: https://coreyalejandro.com
