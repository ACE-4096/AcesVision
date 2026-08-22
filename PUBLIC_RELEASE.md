# Public release checklist

This checkout was developed privately. Do not change the visibility of its
existing remote or push its full history to a public host: earlier commits
contain machine-specific development paths and household-oriented examples.

Publish a fresh repository from the reviewed current tree instead:

1. Run the checks in `CONTRIBUTING.md` and confirm `git status --short` only
   lists intentional release work.
2. Create a new empty public repository with no generated files, model weights,
   enrolled faces, camera configurations, databases, or tokens.
3. Export the reviewed tree as a new initial commit. This keeps private commit
   history private while preserving the current source, licence, documentation,
   and release metadata.
4. Configure private vulnerability reporting on the hosting service, then
   update `SECURITY.md` with that exact contact route before announcing it.
5. Confirm the public repository licence, model-download instructions, and
   third-party dependency terms are appropriate for the release you intend.

The `scripts/install-desktop.sh` installer is portable: it expands the
checkout location and selected Python executable into the user-level systemd
unit. It does not install a system service or modify hardware configuration.
