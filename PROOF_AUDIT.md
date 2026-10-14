# Verification scope, version 0.3.0

See VALIDATION_REPORT.json, exact_checks_report.json,
analytic_certificate_checks.json, analytic_report.json,
relative_precision_report.json and relative_precision_plot_report.json.
All listed checks passed on 14 October 2026. The analytic suite ran all
28 groups without skipping slow checks and disables SDP solving in the
core checker. The complete 66-test suite also passed. Some numerical
controls can produce solver-accuracy warnings; numerical results are not
used as substitutes for exact certificate identities and positivity.

The universal mathematical proofs are in the article and supplement.
The relative-accuracy corollary uses the same coupled witnesses and upper
construction. Public data paths were renamed by content; certificate
payloads were preserved. Download matching code and data version 0.3.0.
