# Disposable CI gate verification

This branch exists only to verify that the required `cpu` check blocks merging
into `aethergate` when it fails and permits merging when it passes.
The assertion failure is intentional and will be restored before the positive check.
Do not merge this test pull request.
