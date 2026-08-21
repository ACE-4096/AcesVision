#!/usr/bin/env bash
# Install AcesVision as a normal Plasma application. It starts a stable user
# service rather than a second GUI process that would contend for the camera.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}"
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}"

install -Dm644 "$repo_dir/packaging/acesvision-gui.service" \
  "$config_dir/systemd/user/acesvision-gui.service"
install -Dm644 "$repo_dir/packaging/acesvision.desktop" \
  "$data_dir/applications/acesvision.desktop"

systemctl --user daemon-reload
# Replace only the temporary service created during development. It has no
# durable desktop identity and would otherwise retain the camera port.
if systemctl --user is-active --quiet acesvision-gui-current.service; then
  systemctl --user stop acesvision-gui-current.service
fi
systemctl --user start acesvision-gui.service

printf '%s\n' "Installed AcesVision. Find it in the application launcher or pin it to the panel."
