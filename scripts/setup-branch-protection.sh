#!/usr/bin/env bash
# Protect the main branch: require a passing CI check and one approving review
# before merging, and block force-pushes and branch deletion.
#
#   scripts/setup-branch-protection.sh
#
# Branch protection needs a public repository or GitHub Pro. Run this once after
# the repo is public. enforce_admins is false so the owner can still merge their
# own pull requests (GitHub does not let anyone approve their own PR, so a strict
# rule would otherwise lock a solo maintainer out). Re-running it is safe.
set -euo pipefail

REPO="${REPO:-NikiO-INO/diarizen-apple}"
BRANCH="${BRANCH:-main}"

gh api -X PUT "repos/$REPO/branches/$BRANCH/protection" --input - <<'JSON'
{
  "required_status_checks": { "strict": true, "contexts": ["build-test"] },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON

echo "Branch protection applied to $REPO:$BRANCH"
