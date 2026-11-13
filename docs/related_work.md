# Related Work

This document summarizes the papers currently most relevant to the project. The purpose is not to claim that laundry AI is unexplored, but to identify how existing work motivates our research question.

## 1. Care Label Recognition

### Kralicek, Matas, and Busta — *Care Label Recognition* (ICDAR 2019)
**What it does:**  
Introduces care-label recognition as a document-analysis problem. The work addresses both textual and pictogram information on garment care labels and discusses challenges such as small text, fading labels, and non-standard appearance.

**Why it matters for us:**  
It establishes that automatically extracting maintenance information from care labels is a non-trivial perception task.

**Relevant limitation for our project:**  
The paper focuses on recognizing care-label information itself rather than studying how uncertainty in those predictions affects downstream laundry-planning decisions.

**Reference:**  
DOI: https://doi.org/10.1109/ICDAR.2019.00158

---

## 2. Large-Scale Multimodal Fashion Care Recognition

### Su and Wang — *Large Scale Multimodal Fashion Care Recognition* (Multimedia Systems, 2024)
**What it does:**  
Studies multimodal fashion-care recognition using both visual care symbols and textual information. The work introduces the FashionCare dataset and investigates fine-grained and few-shot recognition settings.

**Why it matters for us:**  
This is a strong perception-side anchor paper because it shows that deep learning can be used to extract structured garment-care information from care labels at scale.

**Relevant limitation for our project:**  
The main goal is recognition performance. Our project asks a different downstream question: how should a planner behave when those recognition outputs are uncertain?

**Reference:**  
DOI: https://doi.org/10.1007/s00530-024-01461-2

---

## 3. AI-Assisted Laundry Sorting

### Rabethge and Kummert — *Developing a Human-centred AI-based System to Assist Sorting Laundry* (2024)
**What it does:**  
Develops a human-centered AI system for industrial laundry sorting. Camera images and AI models are used to classify individual laundry items into washing categories. The system also considers human-machine interaction and active learning.

**Why it matters for us:**  
It shows that AI-assisted laundry sorting is already an established application area. Therefore, our project should not claim that using AI to sort laundry is itself novel.

**Relevant limitation for our project:**  
The work focuses primarily on recognition/classification into washing categories. Our project focuses on uncertainty-aware downstream grouping and the safety-efficiency tradeoff that arises when predictions are imperfect.

**Reference:**  
DOI: https://doi.org/10.1007/978-3-658-43705-3_3

---

## 4. Laundry Optimization

### Geng et al. — *Strategies for Optimizing Resource Efficiency in the Laundry Process* (Water Supply, 2025)
**What it does:**  
Develops an optimization framework for resource-efficient mixed washing. The study considers factors including garment material, stain information, detergent solubility, and washing frequency, and applies depth-first search to identify efficient mixed-washing schemes.

**Why it matters for us:**  
It demonstrates that laundry grouping and resource-efficient mixed washing have already been studied as optimization problems.

**Relevant limitation for our project:**  
The optimization setting works with garment properties and washing constraints as inputs. Our project instead studies what happens when those inputs come from an imperfect AI perception system and carry uncertainty.

**Reference:**  
DOI: https://doi.org/10.2166/ws.2025.075

---

## 5. Uncertainty / Selective Prediction

### Geifman and El-Yaniv — *Selective Classification for Deep Neural Networks* (NeurIPS 2017)
**What it does:**  
Studies selective classification, where a model may reject or abstain from uncertain predictions rather than always forcing a decision.

**Why it matters for us:**  
The paper provides a general machine-learning motivation for treating low-confidence predictions differently from high-confidence predictions.

**How we use the idea:**  
Our planner may behave more conservatively when garment attributes are uncertain instead of blindly using the top-1 prediction.

**Reference:**  
https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

---

## 6. Current Research Gap

The papers above suggest two established research directions:

### Perception
AI systems can recognize:
- care-label text and symbols;
- garment attributes;
- laundry categories.

### Planning / Optimization
Optimization methods can use garment properties and constraints to produce more efficient washing strategies.

### Our Question
These two stages are often studied separately. In a real AI system, however, the output of the perception stage may be uncertain or wrong.

Our project therefore investigates:

> **When garment attributes and care requirements are predicted imperfectly by AI, can a confidence-aware planner produce safer laundry loads than a planner that treats every prediction as certain, while keeping the number of loads low?**

We will avoid claiming that no prior work has ever considered uncertainty in laundry planning. Instead, our contribution is to explicitly investigate this perception-to-decision interface and evaluate it experimentally.

## 7. Related-Work Story for the Final Presentation
A concise presentation narrative could be:

1. Care-label recognition is already possible with modern computer vision and multimodal methods.
2. AI-assisted laundry sorting and laundry optimization have also been studied.
3. Recognition is imperfect, yet downstream decision systems may rely on predicted garment attributes.
4. We study how recognition uncertainty affects laundry-planning safety.
5. We compare deterministic planning with an uncertainty-aware planner and an oracle.
