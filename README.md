# HR Upskilling Causal Inference

An end-to-end causal inference pipeline evaluating the impact of an HR employee upskilling program on 12-month post-program performance scores. The analysis moves beyond correlation by leveraging structural causal modeling (DoWhy) and machine learning estimation (EconML) to isolate the true, selection-bias-corrected treatment effect.

## Problem Statement

Evaluating workplace learning programs purely by comparing participant performance against non-participants typically yields heavily biased conclusions. High-performing or highly engaged employees naturally self-select into training programs.

The primary objective of this project is to decouple pre-existing selection bias from true program impact. Specifically, the pipeline addresses:

- Selection Bias Identification: Quantifying baseline covariate imbalances between treated and control groups.

- Causal Identification: Structuring a Directed Acyclic Graph (DAG) to block all backdoor confounding paths.

- Bias-Corrected Estimation: Computing the unbiased Average Treatment Effect (ATE) using Double Machine Learning (DML).

- Heterogeneous Effect Exploration: Uncovering individual-level treatment effects (CATE) across job levels, tenure, and manager quality baselines using Causal Forests.

- Model Validation: Stress-testing causal findings against placebo treatments and random common cause refutation tests.

## Dataset

The analysis uses a synthetic human resources dataset (hr_causal_dataset.csv) comprising individual employee records. 

Key attributes include:

- Treatment Variable: enrolled_in_program (Binary: 1 = Enrolled, 0 = Control)   
- Outcome Variable: performance_score_12m (Continuous 12-month performance rating)   
- Confounders / Features: tenure_years, base_salary, manager_quality_score, engagement_score, job_level (ordinal mapping applied: L1 - Associate through L5 - Principal), and commute_minutes.  
- Demographics / Metadata: employee_id, age, department, hiring_cohort.   

## Causal Inference Pipeline

- Data Cleaning & Covariate Balance Audit
    - Cleaning: Resolved duplicate employee records, imputed missing engagement scores via group medians, corrected age anomalies relative to hiring cohort/tenure, and standardized department naming.
    - Naive Benchmark: Raw performance difference yielded an inflated naive causal effect of +5.34 points.
    - Covariate Imbalance Check: Evaluated Absolute Standardized Mean Difference (SMD) across confounders. Significant imbalance was detected in baseline metrics (e.g., tenure, engagement score), confirming heavy self-selection into the program.
    - Common Support: Estimated propensity scores via Logistic Regression, confirming robust overlap between treated and control groups.

- Causal Graph Formulation & Identification
    - Built a structural DAG via DoWhy mapping backdoor paths from tenure, base salary, manager quality, engagement score, and job level into both program enrollment and performance outcomes.
    - Formally identified the backdoor estimand, proving unconfoundedness under the specified adjustment set.

- Double Machine Learning (DML) for ATE
    - Fit a LinearDML model using GradientBoostingRegressor for both outcome ($Y$) and treatment ($T$) nuisance models alongside LassoCV with polynomial features.
    - Result: The bias-corrected Average Treatment Effect (ATE) settled at +2.6465 points, revealing that selection bias accounted for roughly half of the naive +5.34 observational lift.   

- Causal Forest Estimation for Heterogeneous Effects (CATE)
    - Fitted CausalForestDML (1,000 estimators, min_samples_leaf=10) across numeric features to capture individual-level non-linear treatment effects.
    - Result: Mean individual lift reached +3.1836 points.
    - Subgroup Insights:
        - Job Level: Individual program lift varied across career stages, showing higher relative marginal gains for mid-to-senior levels.
        - Tenure: Employees with mid-range tenure (2-5 years) exhibited strong responsiveness compared to senior veterans.
        - Manager Quality: Performance gains scaled positively with manager quality score quartiles.

- Refutation & Robustness Checks
    - Placebo Treatment Refuter: Scrambled treatment assignments randomly over 20 simulations. Estimated effect dropped from 2.6465 to 0.0044, confirming that the observed effect is driven strictly by true program enrollment.
    - Random Common Cause Refuter: Injected a simulated random confounder. The estimated effect remained rock-steady at 2.6557 ($p \approx 0.166$), confirming model stability against unobserved noise.


## Conclusion

- Relying on simple mean differences overestimates upskilling impact by nearly 100% (+5.34 naive vs. +2.65 true ATE) due to employee self-selection.

- After controlling for confounding variables, the upskilling program causes a statistically robust +2.65 point increase in employee performance scores.

- Causal Forest analyses indicate heterogeneous returns; prioritizing enrollment for mid-tenure employees under high-performing managers maximizes organizational ROI.

