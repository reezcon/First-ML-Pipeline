# Capstone Report — <your lane>

- **Author:** Reezcon Vivo
- **Lane:** Machine Learning
- **Repo:** https://github.com/reezcon/First-ML-Pipeline.git
- **Date:** September 2026

## 1. Problem framing

Managing a website with 30,000 pieces of content, or more, is overwhelming. Every day, some pieces get traffic while others get none. Some readers engage deeply (scroll, click, share), while others do not get interactions. With a limited editorial team who can improve content each week, constantly updating and editing becomes a struggle. The question is simple but time sensitive: **Which 50 pieces should we focus on?**

Without data, you might guess. You might refresh the oldest content, or the pages you remember writing, or the ones with the fanciest headlines. However, guessing wastes time, which is a critical and finite resource. You might spend a week refreshing a page that no one visits. Or you might miss a high-traffic page that's slowly losing readers. This work solves that problem. The system **ranks all 30,000 pieces by how much they'll benefit from editorial action**. The ranking uses signals we can measure, such as how many people see the page, how many click it, how long it's been since we updated it, and how long the article is. By learning which combinations of signals predict engagement, it can tell the editorial team which 50 pieces have the most potential.

**Unit of analysis:**
- One row represents one content piece measured over a trailing 90-day window. 
- Each content piece has search, content, freshness, and engagement information. 

**Output:**
- The output is a ranked queue of content pieces, where each piece receives a score and rank. The recommended review size is the top 20–50 pieces.

**Action:**
- A content team manager uses the ranked queue to decide which pieces should be reviewed first. Possible actions include improving a title or meta description, refreshing outdated content, adding internal links, checking search visibility, or testing a different editorial approach.
- The ranking supports human decisions; it does not automatically change or delete content.

**Cost of a wrong call:**
- A **false positive** causes the team to spend time and editorial effort on a piece that has little improvement potential.
- A **false negative** means the team may miss a high-potential piece, allowing an opportunity for more traffic or engagement to go unused. 

**Why ML helps:**
Engagement is associated with several signals at the same time, including content type, position, freshness, word count, CPC, search volume, and competition. These signals can interact in ways that are difficult to represent with many fixed if-statements. A machine learning model can examine these combinations and produce a consistent ranking across thousands of content pieces. In this project, the **Random Forest model** is also useful for exploring which signals are associated with engagement. 

## 2. Data safety

The analysis draws from the **FlyRank internship warehouse**, which is a real-world dataset of 30,000 content pieces collected over a single 90-day period and hosted on Hugging Face. The dataset is public but gated, requiring acceptance of FlyRank's terms. 
Access the dataset here: https://huggingface.co/datasets/FlyRank/internship-warehouse. All the content and client identifiers are pseudonymous (hashed), meaning no domain names, author names, or real company information appears in the data.

We used exactly **seven features** to predict engagement: content_type (the format of the piece, such as keyword article, feedly article, or comparison article), position_tier (where the piece ranks in Google search results, such as top 3, page 1, or page 3-5), freshness_tier (how recently the piece was updated, measured in days since last update and binned into categories: 0-30 days, 31-90 days, 91-180 days, or 181+ days), word_count (the number of words in the article), competition_level (whether the target keyword has low, medium, or high competition), cpc (the cost per click in dollars for the target keyword), and search_volume (the estimated monthly search volume for the keyword). These seven features are sufficient to capture the signal structure without requiring proprietary or sensitive data.

Several columns were **deliberately excluded** because although they seemed helpful, they contained what we call "leakage"—information derived from the label we are trying to predict, which would artificially inflate model performance. Most importantly, I excluded trend_direction and trend_pct. The trend_direction column indicates whether engagement is trending up, down, or stable; trend_pct measures the percentage change in engagement over time. Both of these are computed *from* engagement_rate itself, so using them to predict engagement_rate would be circular reasoning. I also excluded intermediate metrics such as impressions_90d, clicks_90d, and sessions_90d. While these are real measurements, engagement_rate is calculated directly from these numbers (engaged_sessions / total_sessions × 100), so including both would be double-counting the same signal.

i took several **precautions against other forms of leakage**. I never used impressions_last_30d, clicks_last_30d, or sessions_last_30d because these overlap with the 90-day label window and create leakage. I also excluded ai_traffic_pct and scroll_rate because these are derived from engagement and would indirectly leak label information. I confirmed that content_id and client_id are used only for grouping and train-test splitting, never as input features to the model. No personal or client-identifying information appears anywhere in the work/ directory; all outputs contain only pseudonymous IDs and aggregate metrics.

The dataset has one important structural limitation: it is a single 90-day snapshot. All 30,000 content pieces are measured over the same time window, so we cannot observe seasonal effects, algorithm updates, or long-term trends. Additionally, the dataset only includes content that has accumulated traffic (either from search, internal links, or prior promotion). New, unpublished, or unlinked content is invisible. This creates survivor bias: we are only learning from content that has already "survived" enough to get measured. Finally, while we can observe that certain signals correlate with engagement, we cannot prove causation. For example, longer articles correlate with higher engagement, but we do not know whether writing longer articles *causes* higher engagement, or whether better-written articles simply tend to be longer, or whether longer articles naturally target broader topics with more reader interest. This distinction matters for how actionable our recommendations are.

## 3. Baseline

Before building a machine learning model, we created a simple, interpretable baseline rule: flag content with ≥1,000 impressions + CTR ≤0.5% (title/description not compelling), or ≥1,000 impressions + position >10 (poor ranking despite demand), or ≥1,000 impressions + missing position (indexing or tracking issue), or 500-999 impressions + CTR ≤0.5%.

We encoded this as: baseline_score = 3×(high_impr) + 2×(med_impr) + 2×(low_ctr) + 2×(poor_position) + 1×(missing_position), with thresholds: high_impr ≥1000, med_impr 500–999, low_ctr ≤0.5%, poor_position >10, missing_position = null or 0. These are standard SEO thresholds, not tuned on test data.

**Test set results** (6,163 pieces, grouped by client):
- Top-20 mean engagement: 2.3415
- Top-50 mean engagement: 2.9996
- Top-100 mean engagement: 3.6169
- Base rate: 2.91

The baseline successfully identifies high-engagement content. It is a fair comparison because it uses the same data, metric, and test set as the model, and any person can understand and explain it.

## 4. Model / analysis

I chose **Random Forest Regressor** as it handles mixed data types (categorical and numeric) natively, is robust to missing values (learns to branch on missingness), produces interpretable feature importance scores, and trains quickly.

**Preprocessing**: Categorical features filled missing with "MISSING". Numeric features filled missing with median. No scaling, transformations, or feature engineering.

**Exact feature list** (seven features after preprocessing):
1. content_type (keyword article, feedly article, comparison article, MISSING)
2. position_tier (top_3, page_1, striking, page_3_5, deep, MISSING)
3. freshness_tier (0-30, 31-90, 91-180, 181+, MISSING)
4. word_count (median-filled numeric)
5. competition_level (LOW, MEDIUM, HIGH, MISSING)
6. cpc (median-filled numeric)
7. search_volume (median-filled numeric)

**Excluded on purpose**: trend_direction, trend_pct (leakage), impressions_90d, clicks_90d, sessions_90d (intermediate metrics), impressions_last_30d, clicks_last_30d (temporal leakage), ai_traffic_pct, scroll_rate (derived from engagement).

**Target**: engagement_rate, a continuous metric from 0-100 = (engaged_sessions_90d / total_sessions_90d) × 100, observed from data.

## 5. Evaluation

Your split (grouped by client? time-aware?) and why. Metrics, model vs baseline **on the same
split**. What the errors look like — a short error analysis beats a big metric table.

## 6. Interpretation

What the model/clusters actually found. Feature importances or cluster profiles in plain
words. Surprises and negative results — a well-understood "no effect" is a valid result.

## 7. Recommendation

The ranked actions or decisions your output supports, and how a FlyRank editor would use them
tomorrow. State your confidence and the limits explicitly.

## 8. Reproducibility

The exact commands to re-run everything from a fresh clone, your random seeds, and your
environment (`pip freeze` highlights or `requirements.txt` deltas).

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> **Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
> precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
> baseline are the honest discrimination numbers.
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
