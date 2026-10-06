# Summary Report — Mall Shopper Profiling

## 1. Business problem and dataset
A large Indian mall chain wants to understand how its visitors spend so it can redesign store layouts, plan promotional events and offer personalised loyalty rewards. I used the Kaggle *Mall Customer Segmentation* dataset: 200 customers with Age, Gender, Annual Income (k$) and Spending Score (1–100). The data is clean (no missing values; 112 female, 88 male), so the real work was in tuning the algorithms and interpreting the segments.

## 2. Features used (2D vs 5D)
I ran two experiments with K-Means. **Experiment A (2D)** used AnnualIncome and SpendingScore. **Experiment B** added Age and Gender_enc (the exam calls it "5D", although it has four features). All features were standardised first, since distance-based algorithms would otherwise be dominated by income. The 2D set gave far better-separated clusters (silhouette **0.555** vs **0.314**) because Age and Gender overlap heavily between groups, and the correlation matrix showed income has almost no relationship with age or spending. The 2D set was used for all three algorithms and the final model.

## 3. Best algorithm by internal metrics
| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | Noise |
|---|---|---|---|---|---|
| K-Means (k=5) | 5 | 0.5547 | 0.5722 | 248.65 | 0% |
| Agglomerative (Ward, 5) | 5 | 0.5538 | 0.5779 | 244.41 | 0% |
| DBSCAN (eps=0.4, min_samples=10) | 4 | 0.5972 | 0.4734 | 263.61 | 25.5% |

DBSCAN scores highest, but only on the 74.5% of customers it does not label as noise. Among algorithms that assign every shopper, **K-Means is best** and was chosen as the final model. It matched business intuition: the elbow and silhouette plots both pointed to k = 5, the same five groups visible in the Income vs Spending scatter plot. Ward clustering agreed with K-Means on 98% of customers, and K-Means was perfectly stable across five random seeds.

## 4. Shopper segments
- **Big Spenders (39):** high income (~86k) and high spending (~82); premium-brand buyers.
- **Young Aspirers (22):** youngest group (~25), low income but high spending; trend-driven.
- **Mainstream Shoppers (81):** average income and spending; the largest group.
- **Budget Shoppers (23):** low income and low spending; value and deal seekers.
- **Careful Spenders (35):** high income (~88k) but very low spending; wealthy but not engaged.

DBSCAN found four of these groups and labelled most Budget Shoppers (21 of 23) as noise, so noise there means sparse segments, not "average" customers.

## 5. Next steps
1. Gather behavioural data: purchase categories, basket value, visit frequency from footfall sensors and loyalty-app activity.
2. Re-run segmentation with these features and test semi-supervised methods once some segments are validated by the business.
3. Wrap `classify_shopper()` in a real-time persona-tagging API inside the mall app, so offers can be personalised at the time of visit.
4. Run A/B tests of the segment strategies (e.g. ₹50-off coupons for Budget Shoppers) to measure real revenue impact.
