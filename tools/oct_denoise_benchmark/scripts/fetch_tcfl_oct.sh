#!/usr/bin/env bash
set -euo pipefail
root="${1:-third_party_external/TCFL-OCT}"
commit="7320fc84fec37280d9643b1010d8afc4381f5a48"
if [[ ! -d "$root/.git" ]]; then
  git clone https://github.com/gengmufeng/TCFL-OCT "$root"
fi
git -C "$root" fetch origin
git -C "$root" checkout --detach "$commit"
test "$(git -C "$root" rev-parse HEAD)" = "$commit"
