# Security policy

## Supported versions

Security fixes are made to the latest released version.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting (the **Security** tab of the
repository, then *Report a vulnerability*) and include what you found, how to reproduce it and what you think the impact is.
You can expect an answer within a week.

## What the project already does

* No telemetry and no network calls unless you load a URL or a cloud source, or use the optional local language model.
* SQL typed by the user is validated as read-only before it runs (comments and strings aware, one statement, no file-reading functions).
* Formulas are evaluated by a small safe interpreter, never with `eval`.
* Excel exports store text that starts with `=` as text, not as a formula.
* The model API listens only on this computer unless a token is required.

## Things to be aware of

* **Model packages (`.klmodel`, `.pkl`) are pickle files and can run code when opened.** Only open files you created or trust. The dashboard asks you to confirm.
* **A recipe that fingerprints columns stores its salt in clear text.** Keep such recipes private if the fingerprints must stay unlinkable.
* Synthetic data lowers the risk of exposing people but does not remove it; check the privacy figures in its report.
