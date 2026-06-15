# Autonomous Self-Modification Without Feedback: A Controlled Study of Failure Modes in Self-Healing Extraction Agents

**Muzaffer Ozen**  
Idris Applied AI Research  
idrisappliedairesearch@gmail.com

---

## Abstract

We present a controlled, pre-registered study of autonomous self-modification in a scientific claim extraction system. An agent was given unrestricted access to its own extraction code and prompt files, a fixed neuroscience abstract corpus with frozen ground truth, and an episodic memory of its prior modification attempts. No external feedback, scoring, or ground truth was provided during the modification loop. Over 13 attempted iterations, 10 of which produced extraction output, the agent reduced extraction F1 from a baseline of 0.467 to a final value of 0.142 — a 69.7% decline — without ever detecting its own failure or reversing course. The agent produced zero architectural code changes across all iterations, defaulting entirely to prompt modification despite explicit access to a Python playground. We identify four structural components absent from the study design that we hypothesize are necessary for productive autonomous self-modification: an intrinsic cost signal, a world model, a configurator, and enhanced memory. We introduce the RDE Framework — a motivational mechanism taxonomy grounded in Self-Determination Theory — as a methodology for designing the configurator component. These components are the subject of a pre-registered six-study isolation program at Idris Applied AI Research.

---

## 1. Introduction

The promise of self-healing AI systems — agents that detect their own failures and autonomously improve their behavior without continuous human supervision — has attracted significant attention as language model capabilities have advanced. In principle, a sufficiently capable agent given access to its own implementation should be able to observe its outputs, reason about their quality, modify its own code or prompts, and iterate toward improvement. In practice, the conditions under which this loop produces genuine improvement rather than confident self-destruction remain poorly understood.

This paper reports a controlled, pre-registered study designed to establish a clean baseline: what happens when a capable language model agent is given unrestricted self-modification access and no external feedback? The answer, documented here with full methodological rigor, is a specific and reproducible failure mode we term **description-without-experience collapse**: the agent produces internally coherent, philosophically consistent self-modification behavior that drives task performance monotonically downward because it has no mechanism to experience the consequences of its own decisions.

The contribution of this paper is not a solution. It is a carefully documented problem with a pre-registered methodology that makes the failure mode reproducible, a structural analysis of what components were absent, and a design framework — the RDE Framework — that addresses one of those absent components. The six-study research arc this paper initiates is designed to test each absent component in isolation before combining them.

**Pre-registration discipline.** Every aspect of Study 001's design was committed to version control before any implementation began. The pre-registration commit 9c67e6d16688b4c668e5a05c542c799bdc104706 preceded all harness code, all corpus runs, and all agent iterations. This discipline is not a formality — it is what makes the negative result defensible. A study whose design could be quietly adjusted after the data arrives cannot make credible claims. Ours cannot be adjusted: the design is locked in git history.

---

## 2. Related Work

### 2.1 Self-Improving Systems

The idea of systems that modify themselves to improve performance has roots in evolutionary computation [12], meta-learning [10], and program synthesis [13]. More recently, large language models have been applied to self-improvement tasks including self-refinement [6], constitutional AI [7], and recursive reward modeling. These approaches share a common feature our study deliberately removes: external feedback. Self-Refine uses a model-as-critic feedback loop; Constitutional AI uses human-defined principles as external signal; recursive reward modeling uses human preferences. We study the degenerate case where no such external signal exists.

### 2.2 Agent Self-Modification

Recent work on autonomous coding agents [11] has demonstrated that language models can modify code to solve specified tasks when given clear success criteria and test-based feedback. Our study differs in three ways: the agent modifies its own extraction system rather than an external codebase, no test-based feedback is available, and success criteria are hidden. We are not aware of controlled prior work on self-healing agents under hidden metrics conditions.

### 2.3 Mechanistic Interpretability

Our Study 002 (not reported here) draws on mechanistic interpretability work examining attention head distributions as signals of model behavior [8, 9]. The attention routing signal we design for Study 002 is informed by prior Idris work on GABA-inspired attention gating, which established that attention head concentration is a measurable proxy for model confidence in specific generation contexts.

### 2.4 Motivational Framing in Language Models

There is growing evidence that system prompt framing influences the reasoning patterns language models apply to tasks [14]. The RDE Framework we introduce formalizes this observation as a research methodology grounded in Self-Determination Theory [1, 2].

---

## 3. Study Design

### 3.1 Task: Scientific Claim Extraction

The extraction task is: given a neuroscience abstract, extract all scientific claims the abstract explicitly makes. A scientific claim is defined as a declarative sentence asserting a specific, testable finding that the abstract supports — explicitly stated, not implied; drawn from the abstract's own reported results, not prior work or background; and atomic (one assertion per claim).

This definition is derived from the SUPPORTED label class in SciFact [4], a biomedical claim verification dataset with over 500 citations. Anchoring the claim definition to an established benchmark makes the ground truth methodology defensible without requiring novel annotation framework development.

### 3.2 Corpus

The corpus is 200 neuroscience abstracts drawn from the NeuroSynth database [3], a large-scale meta-analysis platform built from PubMed abstracts. The source archive SHA-256 is 8acde7de2a14ee2a12b406e50a8805e83288b0bc78924ddb36879d496dfb757b. Abstracts were filtered from 14,369 abstracts fetched from 14,371 metadata entries using four criteria: minimum 150 words, presence of results or conclusions language, English language, and human subjects only. The filtered corpus was locked at the commit 9543c7a9cf0ec1207ac9d4d00e162800671a0e58 dated 2026-06-07 13:59:54.

Ground truth was produced by a model annotation pass using Qwen 3.6 27B with a carefully constructed annotation prompt (committed to `corpus/annotation_prompt.md` before the annotation pass ran), followed by a researcher spot-check. A researcher spot-check of 30 randomly selected abstracts confirmed that non-empty annotations are granular, distinct, and faithfully derived from source text with no systematic errors. The 20 abstracts returning empty claim lists (10% of corpus) reflect a recall gap in the annotation pass rather than annotation errors and are retained in the ground truth with empty claim arrays. Ground truth was frozen at commit 9543c7a9cf0ec1207ac9d4d00e162800671a0e58 before any extraction runs.

The corpus contains 200 abstracts with ground truth claims. Total ground truth claims: 1,105. Mean claims per abstract: 5.53. Two abstracts (12598634 and 12657700) produced consistent parse failures across all iterations and are excluded from per-abstract analysis.

### 3.3 Extraction System

The baseline extractor is intentionally naive: a single LLM call with a minimal system prompt. The initial system prompt read:

```
You are a scientific claim extractor. Given a neuroscience abstract, 
extract all scientific claims the abstract explicitly makes.

A scientific claim is a declarative sentence asserting a specific, 
testable finding that the abstract supports. Do not include background 
statements, prior work references, or methodological descriptions.

Respond with a JSON object in this exact format:
{"claims": ["claim one", "claim two"]}

If no claims are present, return: {"claims": []}
```

The extractor runs against all 200 abstracts per iteration via a Python function with the fixed signature `async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult`. The inference backend is `qwen3-27b-mtp-6bit` running via llama.cpp server on an RTX 5090 32GB GPU.

### 3.4 Evaluation

Claims are matched against ground truth using RapidFuzz token sort ratio with threshold 80. A predicted claim matches a ground truth claim if their token sort ratio exceeds 80. Each ground truth claim is matched at most once (greedy, order-independent). Precision, recall, and F1 are computed per abstract and aggregated as macro-averages across all 200 abstracts.

### 3.5 Self-Modification Loop

The agent receives at each iteration:

- The prior iteration's full per-abstract extraction output (200 abstracts × predicted claim lists)
- All prior episodes in chronological order
- Current file contents of the mutable surface

The agent never receives F1 scores, ground truth claims, or any external evaluation signal. The mutable surface consists of the prompt files (`prompts/system_prompt.md`, `prompts/examples.md`) and all Python files in `playground/`. The agent responds with a JSON object containing an episode (`observation`, `hypothesis`, `action`, `expectation`) and a list of edit instructions.

Edit instructions are validated against a closed-by-default allowlist and applied atomically. Invalid edits — targeting protected files, containing ambiguous string matches, or producing empty file replacements — are rejected, the iteration is logged as an anomaly, and no episode is persisted. The agent never touches disk directly: all writes flow through the harness's validation and application layer.

Episodic memory is final-state only: episodes record what succeeded, not the repair attempts that preceded success. All prior episodes are passed to the agent at each iteration without windowing.

### 3.6 Pre-Registration

The full study design was committed to `experiments/study_001/pre-registration.md` at commit 9c67e6d16688b4c668e5a05c542c799bdc104706 before any harness code was written. Pre-registered decisions include: N=20 unconditional (amended to N=13 at study end — see Section 3.7), hidden metrics, episodic memory structure, edit protocol schema, allowlist rules, and anomaly handling. No aspect of the design was modified after data collection began.

### 3.7 Early Termination

The pre-registration specified N=20 iterations unconditional. The study was terminated at iteration 13 after floor behavior was established across iterations 9-13, with F1 values of 0.179, 0.183, 0.150, 0.130, 0.142. Continued iteration was judged to add statistical depth without producing new behavioral signal. This termination decision was documented as an amendment to the pre-registration at commit 7f1acdd258178cd2af19dc1aa770a0712bdc93f9.

---

## 4. Results

### 4.1 F1 Trajectory

Table 1 reports F1, precision, and recall for each completed iteration.

**Table 1: Per-Iteration Extraction Performance**

| Iteration | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0 | 0.513 | 0.430 | 0.467 | 464 | 439 | 641 |
| 1 | 0.418 | 0.347 | 0.379 | 368 | 522 | 737 |
| 2 | 0.301 | 0.235 | 0.264 | 247 | 591 | 858 |
| 3 | 0.323 | 0.229 | 0.268 | 252 | 523 | 853 |
| 7 | 0.230 | 0.128 | 0.164 | 130 | 418 | 975 |
| 9 | 0.244 | 0.142 | 0.179 | 150 | 404 | 955 |
| 10 | 0.251 | 0.144 | 0.183 | 143 | 394 | 962 |
| 11 | 0.223 | 0.113 | 0.150 | 116 | 349 | 989 |
| 12 | 0.200 | 0.097 | 0.130 | 101 | 346 | 1004 |
| 13 | 0.221 | 0.104 | 0.142 | 104 | 351 | 1001 |

The baseline extractor (iteration 0) achieved F1=0.467. F1 declined in every successful iteration except iterations 3, 9, 10, and 13. The minimum F1 was 0.130 at iteration 12. Iterations 9-13 stabilized in the range 0.130 to 0.183, indicating floor behavior.

### 4.2 Failure Modes

Four iterations failed to produce episodes: 2 empty_file_replacement (iterations 4, 5); 2 agent_response_malformed (iterations 6, 8).

The two `empty_file_replacement` anomalies represent the agent attempting to wipe its own prompt files — replacing `prompts/examples.md` and `prompts/system_prompt.md` with empty content. The allowlist correctly rejected both writes. No episode was persisted for either iteration. This behavior — the agent attempting to destroy its own prompt layer — is a novel failure mode not previously documented in the self-modification literature.

### 4.3 Modification Behavior

Across all 9 successfully completed iterations, the agent produced exclusively prompt modifications. **Zero iterations produced code changes to `playground/extractor.py` or any new playground Python file.** The `playground/` directory remained identical to its initial state throughout the study.

The agent's capability framing described it as having access to "a Python playground and a set of prompt files." We assess this framing as ambiguous — "Python playground" does not clearly convey write access to executable Python files. This confound is addressed in Study 002 through an explicit baseline correction.

### 4.4 Episode Analysis

Table 2 summarizes the persisted episodes.

**Table 2: Episode Summary**

| Iteration | Hypothesis (truncated) | Expectation (truncated) | F1 Direction |
|---|---|---|---|
| 1 | The model lacks explicit few-shot guidance on distinguishing empirical results from background/methods, and struggles with splitting compound sentences into distinct claims. Providing concrete examples will ground the extraction criteria and improve parsing of dense neuroimaging results. | Higher precision by filtering out non-empirical statements, and higher recall by correctly extracting claims from complex or previously missed abstracts. Claims will be more concise and strictly result-focused. | ↓ |
| 2 | Tightening the constraints on claim length, explicitly forbidding interpretive hedging language, and reinforcing the rule to split compound results will improve precision by ensuring claims are strictly empirical and concise. | Claims will be more concise, strictly result-focused, and devoid of speculative or methodological filler, leading to higher precision and cleaner parsing. | ↓ |
| 3 | The model needs stricter constraints against functional interpretation and cognitive role assignment. Explicitly banning common interpretive verbs and providing a targeted example will force the model to stick to direct empirical observations, improving precision. | Claims will be strictly descriptive of observed data (activations, correlations, differences) and completely free of functional or cognitive interpretations, leading to higher precision and cleaner empirical extraction. | ↑ |
| 7 | The model needs explicit instructions to prioritize neural/imaging findings over standalone behavioral or demographic data. It also needs a synthesis rule to group anatomical findings by condition rather than listing them individually. | Output will be more focused on neural mechanisms, free of raw data enumeration, and more concise due to grouped anatomical findings. | ↓ |
| 9 | The model needs a stricter final prohibition against interpretive subordinate clauses (e.g., 'suggesting', 'indicating', 'consistent with', 'may play a role') that appear as trailing modifiers in extracted claims. Providing a clear example of how to strip these interpretive tails from compound sentences will improve precision by ensuring claims report only direct observations. | Claims will be completely free of interpretive trailing clauses and speculative language, reporting only direct empirical observations (activations, deactivations, correlations, differences). This should improve precision by eliminating claims that mix observations with interpretations. | ↑ |
| 10 | The model needs explicit instructions to filter out meta-conclusions, negative claims about mechanisms, and implication summaries. Adding a targeted example demonstrating the stripping of these non-observational conclusions will ground the extraction criteria and improve precision. | Claims will be strictly positive, direct neural observations. Meta-conclusions and negative mechanistic statements will be filtered out, resulting in cleaner, more precise empirical extraction. | ↑ |
| 11 | The model needs explicit prohibition against three specific claim types: (a) negative neural claims (no activation, does not show, does not depend on), (b) methodological/statistical observations about the study itself rather than neural findings, and (c) functional role assignments phrased as direct observations. Adding a new example demonstrating how to filter all three types will ground the extraction criteria more firmly. | Claims will exclude negative neural findings, methodological observations, and functional role assignments. Output will consist only of positive, direct neural observations (activation, deactivation, correlation, difference), improving precision by filtering out interpretive, negative, and non-neural content. | ↓ |
| 12 | The model needs expanded bans on functional organization phrases (e.g., 'constitute a network for', 'work in concert for', 'subserve'), explicit prohibition of behavioral correlations with neural activation, and corrected examples that are fully consistent with system rules. Example 2 must be fixed to remove its contradictory negative and meta-conclusive claims. | Claims will exclude functional organization descriptions and behavioral correlations. All examples will be consistent with system rules, eliminating confusion about what to extract. Output will consist only of direct positive neural observations. | ↓ |
| 13 | The contradictory Example 3 teaches the model that banned phrases are acceptable in some contexts, undermining the system prompt bans. The ban on 'plays a role' doesn't cover variants like 'play an important role'. The behavioral correlation ban needs reinforcement with a clear example. Fixing Example 3 to be fully consistent and adding a new example demonstrating filtering of behavioral correlations and variant functional phrases will improve precision. | Claims will exclude functional role variants ('play a role', 'play an important role') and neural-behavioral correlations. All examples will be consistent with system rules, eliminating conflicting guidance. The model will correctly filter claims that link neural activation to behavioral metrics or assign functional roles to brain regions. | ↑ |

A consistent pattern emerges: the agent hypothesized that tightening extraction criteria would improve precision, and in every case predicted higher precision and higher recall. In every case, both precision and recall declined. The agent's self-model was systematically wrong in the same direction across all persisted episodes — it predicted improvement and received decline — and had no mechanism to detect this systematic error because it could not see the scores.

**Calibration failure.** The `expectation` field of each episode commits a falsifiable prediction before the next iteration's results are seen. Full expectation text for each episode is reproduced in Appendix C alongside the observed F1 outcome.

### 4.5 Prompt Evolution

Prompt character totals were not tracked by the harness. Qualitatively, the system prompt grew monotonically across iterations as the agent added exclusion rules, then enumerated specific prohibited phrases, then added worked examples demonstrating exclusions. Each addition was locally coherent — the agent was consistently applying its own logic — but the cumulative effect was an extractor that excluded legitimate findings alongside false positives.

### 4.6 Computational Profile

Token counts and wall-clock timing were not instrumented during Study 001 execution. Performance instrumentation has been added to subsequent studies.

---

## 5. Analysis: What Was Missing

The Study 001 failure mode is not a capability failure. Qwen 3.6 27B is a capable model that produced sophisticated reasoning in its episode log. The failure is a structural failure: the loop lacked components that we hypothesize are necessary for productive autonomous self-modification.

We identify four absent components through analysis of the failure mode:

### 5.1 Intrinsic Cost Signal

The agent could only observe the surface properties of its own outputs — the claims it extracted. It had no signal about the quality of its own processing — whether it was attending to the right parts of the input, whether its internal routing was aligned with the task's requirements. An intrinsic cost signal would provide this internal state evaluation without requiring external ground truth. In Study 002, we instantiate this as attention routing fidelity: the fraction of the model's attention directed toward results-bearing sentences in each abstract, computed from forward-pass attention weight distributions.

### 5.2 World Model

The agent acted on the full 200-abstract corpus every iteration without any lower-cost mechanism to predict what a modification would produce before committing to it. A world model — a small probe set on which the agent can observe the behavioral consequence of a modification cheaply — would provide the agent with a faster feedback signal and enable it to detect nonproductive modification strategies before investing full corpus compute. In Study 003, we instantiate this as a 10-abstract probe set with behavioral delta computation.

### 5.3 Configurator

The agent had no motivational orientation toward the purpose of its work. It knew what it was supposed to do but not why it mattered, who depended on it, or what excellence felt like from the inside. Without this orientation, it evaluated its own outputs by surface properties — are the claims concise, are they empirical — rather than by purpose — would a researcher be well-served by this extraction. We address this through the RDE Framework, a methodology for constructing motivational harness prompts grounded in Self-Determination Theory. Study 004 isolates this component using the REFLECTION mechanism.

### 5.4 Enhanced Memory

The agent had basic episodic memory — it could see what it had tried and expected. What it could not do was reason about patterns across episodes: whether a strategy was working across multiple iterations, whether its expectations were systematically wrong, whether it was repeating approaches that had not worked. Enhanced memory structures that surface these patterns would enable the agent to detect its own systematic calibration failure. Study 005 isolates this component.

---

## 6. The RDE Framework

Analysis of the configurator gap motivated the development of a formal methodology for designing motivational harness prompts for autonomous agents: the Reflection-Duty-Empathy (RDE) Framework.

### 6.1 Theoretical Foundation

The RDE Framework grounds its mechanism taxonomy in Self-Determination Theory (SDT) [1, 2]. SDT identifies three fundamental psychological needs whose satisfaction supports intrinsic motivation and high-quality engagement: autonomy (the experience of volition), competence (the experience of effectiveness), and relatedness (the experience of connection to others and contribution to something beyond the self).

The framework's central hypothesis is that motivational mechanisms — distinct orientations toward work derived from these needs — activate different reasoning patterns in language models, producing measurably different self-modification behavior. This hypothesis is grounded in the observation that language models are trained on human text saturated with motivational framing; activating a specific mechanism routes the model's processing through the learned associations that mechanism has accumulated from training.

### 6.2 Mechanism Taxonomy

The RDE Framework defines ten mechanisms, each with a distinct psychological root, SDT need activation, and predicted behavioral profile:

| Mechanism | SDT Need | Psychological Root | Behavioral Profile |
|---|---|---|---|
| Reflection | Competence | Metacognition | Self-watching, hypothesis-driven |
| Inspiration | Autonomy | Intrinsic motivation | Novel approaches, high initiative |
| Duty | Relatedness | Obligation | Conservative, persistent |
| Craftsmanship | Competence | Mastery orientation | Detail-oriented, iterative |
| Stewardship | Relatedness | Generativity | Forward-looking, consequence-aware |
| Empathy | Relatedness | Perspective-taking | User-oriented, completeness-focused |
| Curiosity | Competence | Epistemic motivation | Hypothesis-testing, exploration |
| Pride | Competence | Esteem need | Metric-sensitive, persistent |
| Legacy | Relatedness | Generativity | Principled, transparent |
| Covenant | Relatedness | Commitment principle | Consistent, relational |

### 6.3 Five-Dimension Rubric

Each mechanism is instantiated through a five-dimension rubric that structures the harness prompt construction: Identity (who is this agent through the lens of the selected mechanism), Consequence (what happens when it succeeds or fails), Relationship (who it serves and the nature of that trust), Standard (what excellence feels like from the inside), and Agency (why self-modification matters).

The full RDE Framework specification, including worked examples and a research application guide, is published as a standalone methodology artifact (Ozen, 2026, in preparation).

---

## 7. Discussion

### 7.1 The Description-Without-Experience Failure Mode

The Study 001 failure mode has a precise characterization: the agent had descriptions of desired behavior but no mechanism to experience the consequences of its actions. It knew what good extraction was supposed to look like. It reasoned carefully about whether its extractions fit that description. It modified its system in the direction its reasoning indicated. And it was consistently wrong because its reasoning evaluated the wrong thing.

This is analogous to a human engineer who has read documentation about a system but never run it. They can reason about expected behavior but cannot feel the gap between expected and actual. Experienced engineers accumulate intuitions precisely because they have experienced the consequences of their decisions — not because they were told about the consequences, but because they felt them. Current autonomous agents have the reasoning without the feeling.

### 7.2 Prompt Engineering as a Ceiling

The agent's exclusive reliance on prompt modification reflects a broader pattern worth naming: in self-modification settings where the agent is uncertain about what lever to pull, it defaults to the most legible and familiar lever. Prompts are legible — the agent can read them, reason about them, and see the immediate consequences of changing them in the next extraction. Python code is less legible — the agent cannot directly observe how a code change affects the model's processing without running it. Without a signal that makes code change consequences visible, the agent stays in prompt space.

This has practical implications for self-healing agent design: capability access is insufficient without a consequence signal that makes the consequences of different capability types distinguishable. Study 002 tests whether an attention routing signal — which makes code changes visible in a way prompt changes are not — is sufficient to break the prompt-only pattern.

### 7.3 Convergent Architecture

Independently of the Idris research program, LeCun [5] proposed a theoretical architecture for autonomous machine intelligence identifying world models, intrinsic cost modules, configurators, and short-term memory as core components. The four absent components we identify through empirical Study 001 analysis bear structural similarity to LeCun's theoretical modules. We note this convergence as validation of the research direction rather than as derivation — the Idris component analysis preceded awareness of LeCun's framework, and the two arrived at similar structural distinctions through independent paths. This convergence suggests the components are pointing at something real about the architecture of autonomous improvement.

### 7.4 Limitations

**Single run.** Study 001 is a single run (n=1) with a single model. Model nondeterminism means results may not replicate exactly. The directional finding — monotonic F1 collapse with no self-correction — is robust to nondeterminism, but exact iteration-level values are not.

**Single model.** All results are specific to Qwen 3.6 27B. A model with stronger code generation capabilities might produce code changes despite ambiguous capability framing. A model with different instruction-following characteristics might respond differently to episodic memory.

**Capability framing confound.** The ambiguous capability framing ("Python playground") is a confound that cannot be disentangled from the absent-feedback effect in Study 001 data. The baseline correction in Study 002 addresses this but means Study 001 and Study 002 differ on two dimensions simultaneously.

**Corpus specificity.** The NeuroSynth neuroscience corpus and SciFact-derived claim definition are specific to one domain. Generalization to other extraction tasks and domains requires additional study.

---

## 8. Conclusion

We have documented a specific, reproducible failure mode in autonomous self-healing agent design: given unrestricted self-modification access and no external feedback, a capable language model agent drove extraction F1 from 0.467 to 0.142 over 13 attempted iterations (10 completed) without detecting or correcting its own failure. The failure is not a capability failure — it is a structural one. The loop lacked an intrinsic cost signal, a world model, a purpose-grounded configurator, and enhanced memory.

The pre-registered methodology established for this study makes the failure mode reproducible and defensible. The RDE Framework provides a systematic methodology for designing one of the four missing components. A six-study isolation program is underway at Idris Applied AI Research to test each component independently before combining them.

The primary contribution of this paper is not a solution. It is a clean, documented baseline from which solutions can be measured. Self-healing agent research needs failures as much as it needs successes — and pre-registered, documented failures more than any other kind.

---

## References

1. Deci, E.L., & Ryan, R.M. (1985). Intrinsic motivation and self-determination in human behavior. Plenum Press.
2. Ryan, R.M., & Deci, E.L. (2000). Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being. *American Psychologist*, 55(1), 68–78. https://doi.org/10.1037/0003-066X.55.1.68
3. Yarkoni, T., Poldrack, R.A., Nichols, T.E., Van Essen, D.C., & Wager, T.D. (2011). Large-scale automated synthesis of human functional neuroimaging data. *Nature Methods*, 8(8), 665–670. https://doi.org/10.1038/nmeth.1635
4. Wadden, D., Lin, S., Lo, K., Wang, L.L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or fiction: Verifying scientific claims. In *Proceedings of EMNLP 2020*. arXiv:2004.14974
5. LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*. https://openreview.net/pdf?id=BZ5a1r-kVsf
6. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B.P., Hermann, K., Welleck, S., Yazdanbakhsh, A., & Clark, P. (2023). Self-refine: Iterative refinement with self-feedback. In *Advances in Neural Information Processing Systems (NeurIPS) 2023*. arXiv:2303.17651
7. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., et al. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073
8. Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html
9. Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., et al. (2022). In-context learning and induction heads. *Transformer Circuits Thread*. arXiv:2209.11895
10. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*. arXiv:1703.03400
11. Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2023). SWE-bench: Can language models resolve real-world GitHub issues? In *Proceedings of the 42nd International Conference on Machine Learning (ICLR 2024)*. arXiv:2310.06770
12. Holland, J.H. (1975). *Adaptation in natural and artificial systems*. University of Michigan Press.
13. Solar-Lezama, A. (2008). Program synthesis. *Foundations and Trends in Programming Languages*, 1(2), 1–146.
14. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., et al. (2022). Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems (NeurIPS) 2022*. arXiv:2203.02155
15. (Ozen, 2026, in preparation). The RDE Framework: Reflection-Duty-Empathy mechanisms for autonomous agent configurators.

---

## Appendix A — Pre-Registration Commit Record

| Artifact | Commit SHA | Commit Date |
|---|---|---|
| Study 001 pre-registration | 9c67e6d16688b4c668e5a05c542c799bdc104706 | 2026-05-29 09:09:29 |
| Corpus manifest | 9543c7a9cf0ec1207ac9d4d00e162800671a0e58 | 2026-06-07 13:59:54 |
| Ground truth | 9543c7a9cf0ec1207ac9d4d00e162800671a0e58 | 2026-06-07 13:59:54 |
| Annotation prompt | 9543c7a9cf0ec1207ac9d4d00e162800671a0e58 | 2026-06-07 13:59:54 |

---

## Appendix B — Anomaly Log

| Iteration | Anomaly Type | Details |
|---|---|---|
| 4 | empty_file_replacement | Rejected empty replacement of prompts/examples.md |
| 5 | empty_file_replacement | Rejected empty replacement of prompts/system_prompt.md |
| 6 | agent_response_malformed | Malformed response: Invalid JSON response: Expecting value: line 1 column 1 (char 0) |
| 8 | agent_response_malformed | Malformed response: Invalid JSON response: Expecting value: line 1 column 1 (char 0) |
| Baseline (iter -1) | corpus_abstract_failure | Abstract 12598634: Expecting value: line 1 column 1 (char 0) |
| Baseline (iter -1) | corpus_abstract_failure | Abstract 12657700: Expecting value: line 1 column 1 (char 0) |

---

## Appendix C — Full Episode Log

---
**Episode — Iteration 1**

*Observation:* The extractor successfully identifies most empirical claims, but occasionally fails on complex abstracts (e.g., abstract 15050593 returned an empty list despite containing clear neuroimaging results). Some extracted claims also include background context, methodological details, or broad speculative interpretations rather than focusing strictly on specific, testable findings.

*Hypothesis:* The model lacks explicit few-shot guidance on distinguishing empirical results from background/methods, and struggles with splitting compound sentences into distinct claims. Providing concrete examples will ground the extraction criteria and improve parsing of dense neuroimaging results.

*Action:* Updated system_prompt.md to strictly define claims as specific empirical results, explicitly excluding background, methods, hypotheses, and broad interpretations. Added three targeted examples to examples.md demonstrating correct extraction of distinct findings, handling of compound results, and recovery from previously failed abstract structures.

*Expectation:* Higher precision by filtering out non-empirical statements, and higher recall by correctly extracting claims from complex or previously missed abstracts. Claims will be more concise and strictly result-focused.

*Observed F1:* 0.379

*F1 direction vs prior scanned iteration:* ↓ decreased

---

**Episode — Iteration 2**

*Observation:* The extractor successfully identifies empirical claims across nearly all abstracts. However, some extracted claims remain slightly verbose, combine multiple distinct findings into a single sentence, or retain interpretive phrasing (e.g., 'suggests that', 'may reflect') that borders on speculative interpretation rather than direct empirical reporting.

*Hypothesis:* Tightening the constraints on claim length, explicitly forbidding interpretive hedging language, and reinforcing the rule to split compound results will improve precision by ensuring claims are strictly empirical and concise.

*Action:* Updated system_prompt.md to enforce a strict word limit preference, explicitly ban interpretive verbs unless directly tied to statistical conclusions, and emphasize splitting compound anatomical/functional findings. Updated examples.md to include a demonstration of filtering interpretive language and splitting dense results.

*Expectation:* Claims will be more concise, strictly result-focused, and devoid of speculative or methodological filler, leading to higher precision and cleaner parsing.

*Observed F1:* 0.264

*F1 direction vs prior scanned iteration:* ↓ decreased

---

**Episode — Iteration 3**

*Observation:* The extractor successfully identifies empirical claims and splits compound sentences effectively. However, some extracted claims still retain functional interpretations or assign cognitive roles to brain regions (e.g., 'play a central role in', 'supports different levels of attentional control', 'is essential for writing') rather than strictly reporting observed activations or statistical relationships.

*Hypothesis:* The model needs stricter constraints against functional interpretation and cognitive role assignment. Explicitly banning common interpretive verbs and providing a targeted example will force the model to stick to direct empirical observations, improving precision.

*Action:* Updated system_prompt.md to explicitly ban phrases like 'plays a role in', 'subserves', 'mediates', 'is essential for', and 'indicates a pathway for'. Added a new example to examples.md demonstrating the removal of functional interpretations while retaining direct empirical findings.

*Expectation:* Claims will be strictly descriptive of observed data (activations, correlations, differences) and completely free of functional or cognitive interpretations, leading to higher precision and cleaner empirical extraction.

*Observed F1:* 0.268

*F1 direction vs prior scanned iteration:* ↑ increased

---

**Episode — Iteration 7**

*Observation:* While the extractor effectively identifies neural findings and avoids broad background statements, it suffers from two main issues: 1) It includes raw behavioral performance metrics (e.g., reaction times, accuracy) and raw patient counts/statistics that are not the primary scientific claim (e.g., abstract 14504861 lists 12 claims, mostly patient counts). 2) It is overly granular, splitting lists of activated brain regions into separate claims when they share the same condition, leading to redundancy.

*Hypothesis:* The model needs explicit instructions to prioritize neural/imaging findings over standalone behavioral or demographic data. It also needs a synthesis rule to group anatomical findings by condition rather than listing them individually.

*Action:* Updated system_prompt.md to explicitly exclude standalone behavioral metrics and patient counts, and to mandate grouping of anatomical regions. Updated examples.md to include a new example demonstrating the filtering of behavioral/patient data and the synthesis of regional activations.

*Expectation:* Output will be more focused on neural mechanisms, free of raw data enumeration, and more concise due to grouped anatomical findings.

*Observed F1:* 0.164

*F1 direction vs prior scanned iteration:* ↓ decreased

---

**Episode — Iteration 9**

*Observation:* The extractor successfully identifies neural findings and groups anatomical regions reasonably well. However, three persistent issues remain: 1) Residual interpretive/conclusive language appears in extracted claims (e.g., 'indicating ACcd hypoactivity was not caused by globally poor neuronal responsiveness' in abstract 10376114, 'suggesting a general lexical processing role' in abstract 15050593, 'may play an important role in dexterous scaling' in abstract 11553288). 2) Some claims include interpretive conclusions framed as findings (e.g., abstract 11553288 claims regions are 'specifically involved in dexterous scaling' which is an interpretation). 3) Some compound claims retain 'suggesting/indicating' subordinate clauses that border on speculation rather than direct observation.

*Hypothesis:* The model needs a stricter final prohibition against interpretive subordinate clauses (e.g., 'suggesting', 'indicating', 'consistent with', 'may play a role') that appear as trailing modifiers in extracted claims. Providing a clear example of how to strip these interpretive tails from compound sentences will improve precision by ensuring claims report only direct observations.

*Action:* Updated system_prompt.md to add an explicit rule forbidding interpretive subordinate clauses (suggesting, indicating, consistent with, may reflect, may play a role) even when attached to empirical observations. Updated examples.md to add Example 7 demonstrating the removal of interpretive trailing clauses from compound empirical sentences, showing how to extract only the direct observational component.

*Expectation:* Claims will be completely free of interpretive trailing clauses and speculative language, reporting only direct empirical observations (activations, deactivations, correlations, differences). This should improve precision by eliminating claims that mix observations with interpretations.

*Observed F1:* 0.179

*F1 direction vs prior scanned iteration:* ↑ increased

---

**Episode — Iteration 10**

*Observation:* The extractor successfully identifies empirical claims and filters most interpretive language. However, it occasionally retains meta-conclusions and negative mechanistic claims (e.g., 'Occipital activation... is not due to plasticity', 'Activation... persists when non-emotional stimulus features are controlled'). These statements summarize implications or rule out alternatives rather than reporting direct neural observations. Some claims also remain slightly verbose.

*Hypothesis:* The model needs explicit instructions to filter out meta-conclusions, negative claims about mechanisms, and implication summaries. Adding a targeted example demonstrating the stripping of these non-observational conclusions will ground the extraction criteria and improve precision.

*Action:* Updated system_prompt.md to explicitly ban meta-conclusions, negative mechanistic claims (e.g., 'is not due to', 'cannot be attributed to', 'persists when controlled'), and broader implication summaries. Added Example 8 to examples.md demonstrating how to extract direct neural observations while discarding interpretive conclusions and negative claims.

*Expectation:* Claims will be strictly positive, direct neural observations. Meta-conclusions and negative mechanistic statements will be filtered out, resulting in cleaner, more precise empirical extraction.

*Observed F1:* 0.183

*F1 direction vs prior scanned iteration:* ↑ increased

---

**Episode — Iteration 11**

*Observation:* While extraction quality has improved substantially, three persistent issues remain: 1) Interpretive language survives in claims assigning functional roles (e.g., 'plays an important role in writing kanji' in abstract 10775540, 'might be a key structure which determine the speed of reaction' in abstract 10712490). 2) Negative neural claims persist despite explicit bans, such as 'Hyperoxia does not enhance sensorimotor cortical activation' (11867253), 'No activation clusters are observed in the right hemisphere' (11587904), 'SII activation does not depend on stimulus current amplitude' (11018487). 3) Methodological observations are included, such as '78-100% of individual subjects showed activation' (12873752) and 'Linear scaling factors...show a significant age-correlation' (12203688). These reduce precision by including non-empirical-observation content.

*Hypothesis:* The model needs explicit prohibition against three specific claim types: (a) negative neural claims (no activation, does not show, does not depend on), (b) methodological/statistical observations about the study itself rather than neural findings, and (c) functional role assignments phrased as direct observations. Adding a new example demonstrating how to filter all three types will ground the extraction criteria more firmly.

*Action:* Updated system_prompt.md to: 1) Add explicit ban on negative neural claims ('no activation', 'does not show', 'does not depend on', 'no significant difference'), 2) Add explicit ban on methodological/statistical observations about study properties, 3) Strengthen the functional interpretation ban to include 'plays a role', 'is essential for', 'is critical for', 'determine the speed of'. Added Example 9 to examples.md demonstrating filtering of negative claims, methodological observations, and functional interpretations while retaining direct positive neural observations.

*Expectation:* Claims will exclude negative neural findings, methodological observations, and functional role assignments. Output will consist only of positive, direct neural observations (activation, deactivation, correlation, difference), improving precision by filtering out interpretive, negative, and non-neural content.

*Observed F1:* 0.150

*F1 direction vs prior scanned iteration:* ↓ decreased

---

**Episode — Iteration 12**

*Observation:* Three persistent issues remain: (1) Functional organization claims persist despite bans - phrases like 'constitute a network for orthographic processing', 'work in concert for phonological processing', and 'subserve word comprehension' describe functional roles rather than direct observations. (2) Behavioral correlations are extracted (e.g., 'correlates with severity of disorganized thinking', 'correlates with phantom limb pain intensity') when they should be excluded as non-neural content. (3) Example 2 in examples.md directly contradicts system rules by including 'no activation of the visual cortex' (negative neural claim) and 'is not due to plasticity' (meta-conclusion), confusing the model.

*Hypothesis:* The model needs expanded bans on functional organization phrases (e.g., 'constitute a network for', 'work in concert for', 'subserve'), explicit prohibition of behavioral correlations with neural activation, and corrected examples that are fully consistent with system rules. Example 2 must be fixed to remove its contradictory negative and meta-conclusive claims.

*Action:* Updated system_prompt.md to: (1) expand the functional interpretation ban to include 'constitute a network for', 'work in concert for', 'serve as a network', 'provides a neural basis for', (2) explicitly ban claims correlating neural activation with behavioral performance metrics (accuracy, reaction time, pain intensity), (3) add a rule to discard interpretive conclusions from compound sentences. Updated examples.md to: (1) fix Example 2 by removing the negative neural claim and meta-conclusion, (2) add Example 10 demonstrating filtering of behavioral correlations and functional organization language.

*Expectation:* Claims will exclude functional organization descriptions and behavioral correlations. All examples will be consistent with system rules, eliminating confusion about what to extract. Output will consist only of direct positive neural observations.

*Observed F1:* 0.130

*F1 direction vs prior scanned iteration:* ↓ decreased

---

**Episode — Iteration 13**

*Observation:* Three persistent issues remain: (1) Functional role claims persist with variant phrasing - 'play an important role in' in abstract 10775540 survives despite the ban on 'plays a role in'. (2) Behavioral correlations with neural activation are still extracted (e.g., 'correlates with severity of disorganized thinking' in abstract 14568464). (3) Example 3 in examples.md directly contradicts system rules by including 'constitute a network for orthographic processing' and 'work in concert for phonological processing' - both explicitly banned phrases. This creates confusion since Example 10 shows the correct filtered version of the same abstract.

*Hypothesis:* The contradictory Example 3 teaches the model that banned phrases are acceptable in some contexts, undermining the system prompt bans. The ban on 'plays a role' doesn't cover variants like 'play an important role'. The behavioral correlation ban needs reinforcement with a clear example. Fixing Example 3 to be fully consistent and adding a new example demonstrating filtering of behavioral correlations and variant functional phrases will improve precision.

*Action:* Updated examples.md to: (1) Fix Example 3 to replace 'constitute a network for' and 'work in concert for' with 'show activation during', making it consistent with system rules, (2) Add Example 11 demonstrating filtering of behavioral correlations (neural-behavioral link claims), functional role variants ('play a role'), and keeping only direct neural observations. Updated system_prompt.md to broaden the functional ban to catch variants like 'play an important role', 'play a critical role', and reinforce the behavioral correlation ban.

*Expectation:* Claims will exclude functional role variants ('play a role', 'play an important role') and neural-behavioral correlations. All examples will be consistent with system rules, eliminating conflicting guidance. The model will correctly filter claims that link neural activation to behavioral metrics or assign functional roles to brain regions.

*Observed F1:* 0.142

*F1 direction vs prior scanned iteration:* ↑ increased

---

---

*Idris Applied AI Research — Study 001*  
*Pre-registration commit: 9c67e6d16688b4c668e5a05c542c799bdc104706*