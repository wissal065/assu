# Branch strategy

## Main branches

- `main`: production-ready branch, used for stable releases and Render deployment
- `dev`: integration branch for day-to-day team development

## Feature workflow

1. Create a feature branch from `dev`
2. Develop and commit on the feature branch
3. Open a Pull Request to `dev`
4. Wait for CI to pass and review to be completed
5. Merge into `dev`
6. Merge `dev` into `main` when the team is ready to release

## Recommended naming

- `feature/<name>`
- `fix/<name>`
- `docs/<name>`
- `devops/<name>`

## Recommended GitHub branch protection

Apply these rules in GitHub repository settings:

### For `dev`

- Require a pull request before merging
- Require status checks to pass before merging
- Require the `CI pipeline` workflow to pass
- Block direct pushes except for admins if needed

### For `main`

- Require a pull request before merging
- Require status checks to pass before merging
- Require the `CI pipeline` workflow to pass
- Restrict direct pushes
- Allow only release merges from `dev` or emergency hotfixes

## Deployment rule

- Only `main` should trigger automatic deployment to Render
