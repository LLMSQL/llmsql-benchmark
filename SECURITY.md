# Security Policy

## Supported Versions

The project maintainers provide security updates for the following versions:

| Version             | Supported |
| ------------------- | --------- |
| `main` branch       | ✔️        |
| Latest PyPI release | ✔️        |
| Older releases      | ❌         |

If your issue affects a version that is not supported, we still encourage reporting it so we can assess the impact.

---

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public issue**.

Instead, contact us privately:

* **Email:** [dim.pigulsky@gmail.com](mailto:dim.pigulsky@gmail.com)

We will acknowledge receipt within **48–72 hours**, and provide a more detailed response — including a proposed remediation timeline — within **7 days**.

When reporting, please include:

* A description of the vulnerability
* Steps to reproduce (proof-of-concept if possible)
* The potential impact
* Any suggested fixes

We greatly appreciate responsible disclosure.

---

## Security Expectations for Contributors

When contributing code, please:

* Avoid introducing dependencies with known security issues
* Do not commit secrets, passwords, tokens, or private keys
* Follow secure coding practices (input validation, safe deserialization, etc.)
* Use HTTPS endpoints for all network communication
* Prefer well-maintained libraries and avoid deprecated APIs

Submissions failing basic security checks may be rejected.

---

## Dependency & Build Security

This project uses the following to maintain supply-chain security:

* **Pinned dependencies** (`pyproject.toml`)
* Regular checks with tools such as

  * GitHub Dependabot
