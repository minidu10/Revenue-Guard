# Problem Definition

## Business problem
A telecom company loses revenue when customers cancel. Finding likely
leavers early allows the retention team to contact them with an offer.

## ML problem
Binary classification. For each customer, predict the probability that
Churn = Yes.

## Target
Column: Churn (Yes / No)

## Features
19 customer attributes: demographics, services subscribed, contract type,
payment method, monthly charges, total charges.
Excluded: customerID (identifier only, no predictive value).

## Data
IBM Telco Customer Churn. 7043 rows, 21 columns.
Class balance: about 73% No, 27% Yes.

## Known data issues
- TotalCharges is stored as text and contains 11 blank values.
- The dataset is imbalanced.

## Baseline to beat
Always predict "No" -> 73% accuracy, 0% of churners caught.
This shows accuracy is the wrong metric here.

## Metrics
- Primary: ROC-AUC (how well the model ranks risky customers).
- Secondary: recall (what share of real leavers we catch) and precision
  (how many of our alerts are correct).
- Accuracy is reported but not used for decisions.