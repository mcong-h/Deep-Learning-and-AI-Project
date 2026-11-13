# Related Work

This document summarizes the papers currently most relevant to our project. The purpose is **not** to claim that AI-based laundry recognition, sorting, or optimization is unexplored. Instead, we use prior work to identify how existing perception and planning systems motivate our research question.

Our project studies the interface between these stages:

> **How should uncertainty from an imperfect garment-care recognition model affect a downstream laundry load-planning decision?**

---

## 1. Care Label Recognition

### Kralicek, Matas, and Busta — *Care Label Recognition* (ICDAR 2019)

**What it does:**  
Kralicek et al. introduce care-label recognition as a document-analysis problem. Their system combines multilingual text recognition, symbol recognition, and a care-specific vocabulary to extract information such as fiber composition and washing, bleaching, ironing, and tumble-drying instructions.

The authors also introduce the CL2018 dataset, which contains care-label images captured using handheld devices under real-world conditions.

Recognition difficulty varies substantially across care attributes. Their reported recognition rates include:

- fiber composition: 96.8%;
- bleaching: 67.7%;
- ironing: 64.5%;
- tumble drying: 74.2%;
- washing: 58.1%.

**Why it matters for us:**  
This paper establishes that automatically extracting maintenance information from care labels is a non-trivial perception task. More importantly, different care attributes have substantially different levels of recognition reliability.

**Relevant limitation for our project:**  
The paper focuses on extracting care-label information itself rather than studying how uncertainty or errors in those predictions affect downstream laundry-planning decisions.

**Reference:**  
DOI: https://doi.org/10.1109/ICDAR.2019.00158

---

## 2. Large-Scale Multimodal Fashion Care Recognition

### Su and Wang — *Large Scale Multimodal Fashion Care Recognition* (Multimedia Systems, 2024)

**What it does:**  
Su and Wang introduce **FashionCare**, a large-scale multimodal care-label dataset containing:

- 30,477 care-label images;
- 157,907 care symbols;
- six major care-operation categories;
- 66 fine-grained subcategories;
- corresponding textual descriptions.

They propose MMFC, a multimodal multi-task framework combining visual care-symbol information with textual information.

Reported performance on FashionCare includes:

- image-only EfficientNet: 81.12% mAP;
- FashionCLIP: 84.52% mAP;
- MMFC: 87.72% mAP.

**Why it matters for us:**  
This paper shows that modern deep-learning systems can extract structured garment-care information at much larger scale than earlier methods. However, recognition remains imperfect, which motivates our treatment of garment-care information as uncertain rather than automatically correct.

**Important distinction:**  
FashionCare primarily focuses on **care operations**, such as washing, bleaching, drying, ironing, and professional maintenance. We do **not** treat FashionCare as a direct material-classification benchmark.

If material composition is needed by our planning stage, it must come from care-label text/OCR, known metadata, or a separate material-recognition component.

**Relevant limitation for our project:**  
The paper focuses on improving recognition performance. It does not investigate how recognition uncertainty should affect downstream laundry-grouping decisions.

**Reference:**  
DOI: https://doi.org/10.1007/s00530-024-01461-2

---

## 3. AI-Assisted Laundry Sorting

### Rabethge and Kummert — *Developing a Human-centred AI-based System to Assist Sorting Laundry* (2024)

**What it does:**  
Rabethge and Kummert develop a human-centered AI system for industrial laundry classification. Camera images and convolutional neural networks are used to predict attributes including:

- color;
- garment type;
- soiling;
- washing color;
- damage and other laundry-related categories.

Their experiments include both AreaScan and LineScan datasets. On the LineScan dataset, reported results include:

- garment-type accuracy: 96.52%;
- washing-color accuracy: 97.13%.

The system also reports prediction **certainty** and uses Layer-Wise Relevance Propagation (LRP) to inspect model behavior.

High-certainty predictions tend to focus on the garment itself, while lower-certainty examples can correspond to incorrect attention to background regions or other recognition failures.

**Why it matters for us:**  
This paper is the closest bridge between deep-learning perception and practical laundry sorting.

It demonstrates that:

1. AI-based laundry classification already exists;
2. these classifiers can provide confidence/certainty information; and
3. low-certainty predictions can correspond to genuine model failures.

**Relevant limitation for our project:**  
The paper does not investigate how prediction uncertainty should propagate into a multi-garment laundry-planning problem.

In addition, its reported certainty values are neural-network confidence scores. The work does not establish that these values are calibrated probabilities of correctness.

**Reference:**  
DOI: https://doi.org/10.1007/978-3-658-43705-3_3

---

## 4. Deterministic Laundry Optimization

### Geng et al. — *Strategies for Optimizing Resource Efficiency in the Laundry Process* (Water Supply, 2025)

**What it does:**  
Geng et al. study resource-efficient laundry planning using factors including garment material, stain information, detergent properties, and washing frequency.

For the mixed-washing component, the paper considers 132 garments belonging to eight material categories.

Material combinations that cannot be washed together are represented as incompatibilities in a graph. The authors formulate the problem as graph coloring and use depth-first search with backtracking to obtain efficient mixed-washing groups.

In the solution shown in **Figure 6**, the eight material categories are divided into four washing groups:

1. cotton, cotton linen, and cotton-polyester;
2. silk and rayon;
3. polyester and wool-polyester;
4. pure wool separately.

### Primary Anchor Paper

This is the **primary anchor paper for our project**.

### Chosen Result to Reproduce

We plan to reproduce the deterministic material-compatibility graph-coloring result shown in **Figure 6**.

The original formulation assumes that compatibility-relevant garment information is known.

Our project modifies this assumption by considering the case where garment-care information comes from an imperfect AI perception model.

**Why it matters for us:**  
This paper provides the downstream planning framework that we will extend.

**Relevant limitation for our project:**  
The optimization stage treats garment properties and compatibility constraints as known inputs. It does not explicitly study how uncertainty in those inputs should alter the resulting laundry plan.

**Reference:**  
DOI: https://doi.org/10.2166/ws.2025.075

---

## 5. Confidence Calibration

### Guo et al. — *On Calibration of Modern Neural Networks* (ICML 2017)

**What it does:**  
Guo et al. distinguish classification accuracy from **confidence calibration**.

A classifier is calibrated when its predicted confidence corresponds to its empirical probability of being correct. For example, among predictions made with approximately 80% confidence, roughly 80% should actually be correct.

The authors show that modern neural networks can achieve strong classification accuracy while still being substantially miscalibrated.

They evaluate calibration using tools such as:

- reliability diagrams;
- Expected Calibration Error (ECE).

They also show that **temperature scaling**, a simple post-hoc calibration method using a validation set, can substantially improve calibration without changing the model's predicted class.

**Why it matters for us:**  
Our planner intends to use recognition confidence when estimating the risk of grouping garments together.

Therefore, a raw softmax score such as 0.80 should not automatically be interpreted as an 80% probability that the prediction is correct.

Before using confidence values as decision probabilities, we need to evaluate and potentially calibrate them.

**How we use the idea:**  
We plan to compare planning based on:

1. raw model confidence; and
2. calibrated model confidence.

This allows us to test whether calibration improves downstream laundry safety.

**Reference:**  
https://proceedings.mlr.press/v70/guo17a.html

---

## 6. Selective Prediction and Abstention

### Geifman and El-Yaniv — *Selective Classification for Deep Neural Networks* (NeurIPS 2017)

**What it does:**  
Geifman and El-Yaniv study **selective classification**, also known as classification with a reject option.

Instead of forcing a prediction for every input, a classifier may abstain when confidence is too low.

The paper formalizes this as a **risk-coverage trade-off**:

- rejecting more uncertain cases can reduce prediction risk;
- however, doing so also reduces coverage, meaning fewer cases are handled automatically.

**Why it matters for us:**  
Selective prediction provides a natural baseline for dealing with uncertain garment recognition.

A simple laundry system could apply a rule such as:

> If confidence is below a threshold, isolate the garment or request human confirmation.

This is safer than blindly trusting every top-1 prediction.

**How we use the idea:**  
We plan to use selective rejection as a baseline.

Our proposed method goes one step further: rather than making only a binary decision between "accept" and "reject," we aim to use calibrated uncertainty directly when estimating whether two garments can safely share a washing load.

**Reference:**  
https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

---

## 7. Current Research Gap

Existing work establishes several important pieces of the problem.

### Perception

AI systems can recognize:

- care-label text and symbols;
- garment-care operations;
- garment and laundry categories.

However, recognition is imperfect, and confidence can vary substantially across inputs and attributes.

### Uncertainty Handling

Prior machine-learning research shows that:

- raw neural-network confidence may be miscalibrated;
- uncertain predictions can be handled using rejection or abstention.

### Planning / Optimization

Laundry optimization methods can use garment properties and compatibility constraints to construct more efficient mixed-washing plans.

However, these stages are generally studied separately.

Existing laundry optimizers typically construct compatibility constraints from garment information that is treated as known.

In a real AI-assisted system, however, that information may instead come from an imperfect perception model.

### Our Research Question

We therefore investigate:

> **Can uncertainty-aware planning reduce unsafe laundry decisions caused by imperfect garment-care recognition without substantially increasing the number of washing loads?**

More generally:

> **How should calibrated perception uncertainty propagate into a downstream laundry load-planning problem?**

We do **not** claim that uncertainty has never been considered in laundry planning.

Instead, our contribution is to explicitly study the **perception-to-decision interface** and experimentally evaluate how different strategies for handling recognition uncertainty affect downstream laundry safety and efficiency.

---

## 8. Proposed Contribution

Our project connects AI-based garment-care perception with laundry-load planning.

The proposed pipeline is:

**Garment/Care-Label Recognition → Confidence Calibration → Uncertainty-Aware Compatibility Risk → Laundry Load Planning**

We plan to compare the following strategies.

### 1. Deterministic Planner

Uses only the model's top-1 prediction and treats that prediction as correct.

### 2. Selective / Reject Planner

Uses a confidence threshold.

Low-confidence garments are isolated or deferred for human confirmation.

### 3. Raw Uncertainty-Aware Planner

Uses the model's raw, uncalibrated confidence scores when estimating whether garments can safely be grouped together.

### 4. Calibrated Uncertainty-Aware Planner — Proposed Method

Uses calibrated predictive uncertainty when estimating compatibility risk and constructing laundry loads.

### 5. Oracle Planner

Uses ground-truth garment-care information.

This represents the best planning result achievable under perfect perception and serves as an upper-bound reference.

---

## 9. Planned Evaluation

We will evaluate both the perception and planning stages.

### Perception Metrics

- classification accuracy / task-appropriate recognition metric;
- Expected Calibration Error (ECE);
- reliability diagrams.

### Downstream Planning Metrics

- safety-violation rate;
- unsafe pair/group rate;
- number of laundry loads;
- load utilization;
- decision regret relative to the oracle planner.

The central trade-off is:

> **How much additional safety can uncertainty-aware planning achieve, and how much efficiency must be sacrificed to obtain it?**

---

## 10. Related-Work Story for the Final Presentation

A concise presentation narrative is:

1. **Care-label recognition already exists**, from early OCR/symbol-based systems to modern multimodal deep-learning models.

2. **AI-assisted laundry sorting also exists**, and current systems can output both predictions and confidence information.

3. **Recognition is not perfect**, and neural-network confidence itself may be miscalibrated.

4. **Laundry optimization has also been studied**, including deterministic graph-coloring approaches for mixed washing.

5. However, these stages are generally studied separately:
   - perception produces uncertain predictions;
   - optimization often assumes correct garment information.

6. A simple response to uncertainty is to reject low-confidence cases.

7. **Our project instead studies whether calibrated recognition uncertainty can be propagated directly into the downstream laundry-planning decision.**

Our contribution is therefore not a new garment classifier or a new laundry application by itself.

It is an investigation of:

> **Perception uncertainty → downstream decision risk → safety-efficiency trade-off.**
