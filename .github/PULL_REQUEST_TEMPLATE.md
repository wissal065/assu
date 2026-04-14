## Summary

- What changed?
- Why is this needed?

## Checklist

- [ ] My branch is based on `dev`
- [ ] I tested the changes locally
- [ ] I ran the CI checks relevant to my changes
- [ ] I updated documentation if needed
- [ ] This PR targets `dev` unless it is a release or hotfix

## Validation

- Commands run:
  - `python -m uv run ruff check auto_insurance tests`
  - `python -m uv run --with httpx pytest tests -v`

## Deployment impact

- [ ] No deployment impact
- [ ] Requires deploy on Render
- [ ] Requires new secret or environment variable
