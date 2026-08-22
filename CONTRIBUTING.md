# Contributing to AcesVision

Thank you for contributing. Please discuss substantial changes in an issue
before implementation, keep camera and biometric data out of commits, and add
tests for changed behaviour.

Before opening a pull request, run:

```bash
.venv/bin/python -m unittest
QT_QPA_PLATFORM=offscreen .venv/bin/python -m acesvision.gui --smoke-test
```

Never commit `.env`, `cameras.json`, `automations.json`, recordings, enrolled
faces, databases, tokens, or downloaded model binaries. Use the committed
`*.example.json` files for configuration examples.
