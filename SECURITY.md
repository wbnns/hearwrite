# Security

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/wbnns/hearwrite/security/advisories/new)
rather than opening a public issue. Expect an acknowledgement within a few days.

HearWrite is a young project maintained by one person, so there is no formal SLA.
What there is: a real reply, credit if you want it, and a fix before disclosure.

## What is worth reporting

The parts of HearWrite that touch anything dangerous are small and known:

- **Model downloads.** `hearwrite.models` fetches weights over HTTPS and checks
  them against a pinned SHA-256. A way to get an unverified file past that check,
  or to make a download write outside the cache directory, is a real finding.
- **Archive extraction.** Tarballs are extracted with `filter="data"`, which
  refuses absolute paths, parent traversal and device nodes. A way around that
  is a real finding.
- **The WebSocket service.** It accepts binary audio from anyone who can reach
  it. Authentication is the deployer's job and is deliberately not built in, but
  a way to crash the process, exhaust memory past the admission limit, or read
  another session's audio is a real finding.
- **Anything that executes downloaded content.** ONNX models are data, not code,
  but they are fed to a runtime that parses them. A crafted model that escapes
  that boundary is worth reporting to us and to the runtime's maintainers.

## What is not a vulnerability

- **Model weights are third party.** HearWrite pins their checksums so you get
  the file we tested, but their contents, biases and behaviour are their
  authors'. See `NOTICE` for who those authors are.
- **The service has no authentication.** That is a documented design decision:
  the audio path is meant to sit behind a short lived token issued by your own
  application, not to authenticate people itself.
- **Transcripts are not encrypted at rest or in transit by HearWrite.** Run it
  behind TLS. It does not manage keys and does not want to.
