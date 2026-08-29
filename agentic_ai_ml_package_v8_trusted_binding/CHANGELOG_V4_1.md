# CHANGELOG V4.1

## Bug fix — Google Colab detection

The previous notebook used:

```python
IN_COLAB = "google.colab" in sys.modules
```

This can incorrectly return `False` before `google.colab` has been imported,
even when the notebook is actually running in Google Colab.

V4.1 changes detection to:

```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
```

The `ml/agentic_ai` overlay cell also performs its own Colab detection, so it
still works even if the initial environment-check cell was skipped.

Other changes:
- Updated stale V3 wording to V4.1.
- Clarified that uploading the package in Colab does not upload it to GitHub.
