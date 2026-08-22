# Security policy

Please report suspected vulnerabilities privately to the maintainers rather
than opening a public issue. Do not include tokens, camera URLs, credentials,
face images, recordings, or other biometric data in the report.

AcesVision binds its preview and event API to loopback by default and protects
camera frames and events with a local bearer token. Public deployments or
remote exposure are out of scope for the default configuration.
