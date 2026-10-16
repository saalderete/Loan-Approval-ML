# Model limitations

## What the model estimates

The calibrated probability that a 2025 Texas HMDA application with the given 13 characteristics was
**denied** by the lender, among applications that were approved or denied. It is a description of
historical decisions. It is not a credit score, a default probability or a forecast for any person.

## Known limitations

1. **Missing underwriting variables.** No credit score, credit history, assets, reserves, employment,
   or appraisal detail. These drive real decisions and are correlated with the inputs the model does
   see.
2. **Historical bias.** If past decisions were unfair, the model can reproduce that. The fairness
   audit measures disparities but cannot explain their cause.
3. **Proxy effects.** Geography and lender are excluded, but income, loan size and property value can
   still correlate with protected characteristics.
4. **Scope.** Texas, 2025, complete applications only, site-built 1–4 unit homes, closed-end
   consumer loans. Applications with missing fields (denied far more often) are out of scope.
5. **Rounded and banded inputs.** Amounts are rounded to $10k by HMDA; DTI is banded outside 36–49%.
6. **No applicant identity.** Repeat applications by the same person to several lenders cannot be
   linked, so near-duplicates may exist across splits.
7. **Threshold metrics are illustrative.** The research threshold maximises F1 on the selection split.
   It is not a policy.
8. **Explanations are model behaviour.** Shapley contributions and sensitivity curves describe how
   this model responds. They are not causes, not adverse-action reasons, and not advice.
9. **Drift monitoring has no outcomes.** Demo traffic has no labels, so monitoring can detect input
   shift but cannot measure live accuracy or concept drift.

## Prohibited uses

Approving, denying, pricing or pre-screening real applications; marketing or steering; any use with
real personal or credit-report data; presenting estimates as anyone's chance of approval; using
attributions as adverse-action reasons.

A real lending model would need, at minimum: fair-lending testing (disparate treatment and impact),
model-risk management (e.g. SR 11-7), ECOA / Regulation B adverse-action processes, privacy and
security review, ongoing outcome monitoring, and legal sign-off.
