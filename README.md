# RESPOND

Official implementation of **RESPOND**:  
**Risk-Enhanced Structured Pattern for LLM-Driven Online Node-level Decision-making**

---

## Overview

RESPOND is a structured decision-making framework for **LLM-driven autonomous driving agents**
that tightly integrates **quantitative risk modeling** with **large language model (LLM) reasoning**.

While most existing LLM-based driving agents rely on unstructured plain-text memory, RESPOND
grounds high-level reasoning in a **quantified Driver Risk Field (DRF)** and a structured
risk-pattern representation. This design enables precise scene abstraction, reliable memory
retrieval, and efficient reflection.

At the core of RESPOND is a unified **5×3 ego-centric risk pattern matrix**, constructed from
continuous DRF signals. The matrix encodes spatial topology, road constraints, and risk intensity
in a compact and consistent form, allowing LLMs to reason over **explicit, numeric risk-aware
structures** rather than ambiguous textual descriptions.

RESPOND bridges low-level quantitative risk fields and high-level language-based decision-making,
providing a principled interface between classical risk modeling and modern LLM agents.

---

## Paper

**RESPOND: Risk-Enhanced Structured Pattern for LLM-Driven Online Node-level Decision-making**  
Dan Chen, *et al.*

📄 *Paper link*: (to be added)

If you find this work useful, please consider citing our paper.

---

## Code Availability

This repository contains the **official implementation** of RESPOND.

The codebase is currently being organized and will be **fully released shortly**.
The final release will include:

- Quantitative Driver Risk Field (DRF) computation and normalization
- Structured risk pattern construction (5×3 ego-centric matrix)
- Hybrid Rule + LLM decision-making pipeline
- Dual-layer risk-pattern memory and retrieval
- Pattern-aware reflection and structured memory updates
- Experiment scripts corresponding to all sections of the paper

---

## Framework Overview

RESPOND consists of three tightly coupled layers:

### 1. Quantitative Risk Field (DRF)

RESPOND builds upon a continuous **Driver Risk Field (DRF)** that quantifies spatial risk
around the ego vehicle using physically grounded and interaction-aware metrics.
The DRF provides a numeric risk landscape that captures proximity, relative motion,
and collision likelihood.

### 2. Structured Risk Pattern Abstraction

The continuous DRF is discretized into a **5×3 ego-centric risk pattern matrix**, which:

- Preserves spatial topology and road constraints
- Encodes risk intensity in a compact, symbolic form
- Serves as a stable interface between numeric risk modeling and LLM reasoning

This structured representation enables consistent scene matching and avoids the ambiguity
of pure text-based memory.

### 3. Hybrid Rule + LLM Decision and Reflection

RESPOND employs a hybrid pipeline:

- **Exact pattern matching** enables rapid and safe action reuse in high-risk scenarios
- **Sub-pattern abstraction** supports flexibility and personalized driving styles under low risk
- **Pattern-aware reflection** abstracts tactical corrections from crash frames and updates
  structured memory, enabling *one-crash-to-generalize* learning

---

## Relationship to DiLu

This work is developed based on the **DiLu** framework  
(https://github.com/PJLab-ADG/DiLu), which provides a closed-loop LLM-driven autonomous
driving simulation environment.

RESPOND extends DiLu in the following key aspects:

- **Integrates quantitative Driver Risk Fields (DRF) with LLM reasoning**, grounding
  language-based decisions in explicit numeric risk representations
- Introduces **structured risk pattern representations** derived from DRF, replacing
  unstructured plain-text memory
- Adds a **dual-layer risk memory mechanism** supporting exact-pattern reuse and
  sub-pattern generalization
- Implements **pattern-aware reflection** with structured memory updates, enabling
  efficient *one-crash-to-generalize* learning
- Supports **personalized driving styles** through controlled sub-pattern abstraction
  under low-risk conditions

Parts of the code are adapted or modified from DiLu in accordance with its
Apache License 2.0.

We sincerely thank the DiLu authors for their open-source contribution.

---

## Experiments

The final release will include experiment scripts corresponding to the following sections
of the paper:

- Closed-loop simulation in `highway-env`
- Reflection efficiency and one-crash-to-generalize analysis
- Personalized driving under low-risk conditions
- Real-world validation on the highD dataset
- Ablation studies and discussions

Each experiment directory will contain runnable scripts and configuration files
to facilitate reproducibility.

---

## License

This project is released under the **Apache License 2.0**.

It includes code adapted from the DiLu project, which is also licensed under
the Apache License 2.0.

---

## Contact

**Author**: Dan Chen  
For questions or discussions, please open an issue or contact the author directly.
