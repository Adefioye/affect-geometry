# Run logs

This directory intentionally remains tracked for auditability. Each execution writes to:

```text
<model-slug>__<run-id>/run.log
```

Stages using the same model and run ID append to the same chronological UTC log.
