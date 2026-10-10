#!/usr/bin/env bash
set -euo pipefail
asset=go-containerregistry_Linux_x86_64.tar.gz
curl --fail --location --retry 4 --silent --show-error \
  "https://github.com/google/go-containerregistry/releases/download/v0.22.1/$asset" -o "/tmp/$asset"
printf '%s  %s\n' 0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0 "/tmp/$asset" | sha256sum --check
tar -xzf "/tmp/$asset" -C /tmp crane
sudo install -m 0755 /tmp/crane /usr/local/bin/crane
rm "/tmp/$asset" /tmp/crane
crane version
