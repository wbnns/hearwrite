# Contributing to HearWrite

Issues and pull requests are welcome. HearWrite is early, so the most useful
contributions right now are real-world failure cases -- audio where the speaker
labels go wrong, or where an endpoint fires mid-sentence.

## Running it locally

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'   # no models, no downloads, no GPU
bin/check                            # lint, types, Tier 1 -- the definition of done
```

`bin/check` is what CI runs. If it is green locally, the pipeline will be green.
It must stay fast and must never need a GPU, a network or a model download; a
change that breaks that is the wrong change.

## Before you open a PR

- **Add a Tier 1 test.** Almost every real bug in this project is a policy bug,
  and policy bugs are reproducible with the scripted fakes in `src/hearwrite/*/fake.py`.
  A test that needs a real model belongs in `tests/tier2/`.
- **Do not break the append-only rule.** Committed output is final. If a change
  makes it possible for a `commit` to be contradicted, that is a bug regardless
  of what it improves.
- **Keep the base install dependency-free.** `pip install hearwrite` must pull no
  model runtime. Backends go behind an extra.
- **New models need a NOTICE entry** with the licence of the weights, not just
  the code. Gated weights are not accepted.

## License

By contributing you agree that your contributions are licensed under the MIT
License, as in [LICENSE](./LICENSE).
