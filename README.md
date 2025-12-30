# RESPOND

**RESPOND: Risk-Enhanced Structured Pattern for LLM-Driven Online Node-level Decision-making**

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

## 🔥 Quick Start (RESPOND)

The commands below allow you to **run RESPOND out of the box** with the default configuration
and provided structured risk-pattern memory. 
**Note:** It is strongly recommended to use GPU & CUDA to achieve higher efficiency in underlying DRF computations. For PyTorch installation, please follow the official instructions if you need a specific CUDA version.

```bash
git clone https://github.com/gisgrid/RESPOND.git
cd RESPOND

pip install -r requirements.txt

cp configs/respond.example.yaml configs/respond.yaml
# then edit configs/respond.yaml to set your OPENAI_API_KEY
python run_RESPOND.py --config configs/respond.yaml
```

This will:

- Load the default structured risk-pattern memory from `data/default_memory/`
- Run RESPOND in the `highway-env` closed-loop simulation
- Perform online decision-making using the Risk-Enhanced Structured Pattern framework

### Configuration Options

RESPOND behavior is controlled by `configs/respond.yaml`.

The configuration file contains **four predefined runtime options**, corresponding to the
major experimental settings evaluated in the paper.  
Users can switch between options by changing a single field in the YAML file,
**without modifying any code**.

```yaml
############### run_RESPOND options ############
respond_demo_option: 'option1'
```

The available options are:


- **option1: RESPOND (L1 + L2 memory)**  
  Uses the full RESPOND framework with **dual-layer memory lookup**.  
  Exact risk-pattern matches (L1) enable fast action reuse in high-risk scenarios,  
  while sub-pattern memory (L2) supports abstract level driving strategies reuse.

- **option2: RESPOND (L1 memory only)**  
  Uses **exact risk-pattern memory (L1)** without sub-pattern abstraction.  
  This setting evaluates the contribution of L1 exact pattern reuse.

- **option3: No memory (zero-shot with risk encoding)**  
  Disables all memory lookup.  
  The LLM operates in a **pure zero-shot manner**, reasoning only from the current  
  scene description augmented with quantified risk values.

- **option4: RESPOND (L1 + sporty L2 memory)**  
  Uses full dual-layer memory with a **preloaded “Sporty” L2 sub-pattern memory**.  
  This option demonstrates how RESPOND enables **personalized driving styles**
  through sub-pattern abstraction.



Please refer to the inline comments in `respond.yaml` for details on each option.

---

## ▶️ Quick Start (DiLu Baseline)

For reference and comparison, we include the original **DiLu** implementation as a baseline.

To run DiLu:

```bash
cp third_party/dilu/config.yaml .
python third_party/dilu/run_dilu.py
```

⚠️ **Important Notes**

- The DiLu codebase assumes that `config.yaml` is located at the repository root.
  Please copy `third_party/dilu/config.yaml` to the root directory before execution.
- The default DiLu configuration uses an older model setting:
  ```yaml
  OPENAI_CHAT_MODEL: 'gpt-4-1106-preview'
  ```
  For fair comparison with RESPOND, we recommend updating this field in the DiLu
  `config.yaml` to match RESPOND:
  ```yaml
  OPENAI_CHAT_MODEL: 'gpt-4o-mini'
  ```

---

## Paper

**RESPOND: Risk-Enhanced Structured Pattern for LLM-Driven Online Node-level Decision-making**  
Dan Chen, *et al.*

📄 *Paper link*: https://arxiv.org/abs/2512.20179

If you find this work useful, please consider citing our paper.

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{respond2025,
  title   = {RESPOND: Risk-Enhanced Structured Pattern for LLM-Driven Online Node-level Decision-making},
  author  = {Chen, Dan},
  year    = {2025}
}
```

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

## License

This project is released under the **Apache License 2.0**.

It includes code adapted from the DiLu project, which is also licensed under
the Apache License 2.0. See `third_party/dilu/LICENSE` for details.

---

## Contact

**Author**: Dan Chen  
For questions or discussions, please open an issue or contact the author directly.