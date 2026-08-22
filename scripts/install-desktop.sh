#!/usr/bin/env bash
# Install AcesVision as a normal Plasma application. It starts a stable user
# service rather than a second GUI process that would contend for the camera.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}"
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}"
python_bin="${ACESVISION_PYTHON:-$repo_dir/.venv/bin/python}"

if [ ! -x "$python_bin" ]; then
  printf '%s\n' "AcesVision Python was not found at $python_bin." >&2
  printf '%s\n' "Create .venv first, or rerun with ACESVISION_PYTHON=/path/to/python." >&2
  exit 1
fi

install -d "$config_dir/systemd/user"
escape_sed_replacement() {
  printf '%s' "$1" | sed 's/[\\&|]/\\\\&/g'
}
repo_dir_escaped="$(escape_sed_replacement "$repo_dir")"
python_bin_escaped="$(escape_sed_replacement "$python_bin")"
sed -e "s|@PROJECT_DIR@|$repo_dir_escaped|g" \
    -e "s|@PYTHON_BIN@|$python_bin_escaped|g" \
    "$repo_dir/packaging/acesvision-gui.service.in" \
    > "$config_dir/systemd/user/acesvision-gui.service"
chmod 0644 "$config_dir/systemd/user/acesvision-gui.service"
install -Dm644 "$repo_dir/packaging/acesvision.desktop" \
  "$data_dir/applications/acesvision.desktop"

systemctl --user daemon-reload
# Replace only the temporary service created during development. It has no
# durable desktop identity and would otherwise retain the camera port.
if systemctl --user is-active --quiet acesvision-gui-current.service; then
  systemctl --user stop acesvision-gui-current.service
fi
# Older development launches also used this final unit name through
# ``systemd-run``. A failed transient remains preferred over the installed
# fragment until it is explicitly cleared, and it lacked WorkingDirectory.
# Remove only that known transient before loading the persistent unit.
if [ "$(systemctl --user show acesvision-gui.service -p FragmentPath --value 2>/dev/null || true)" \
     = "/run/user/$(id -u)/systemd/transient/acesvision-gui.service" ]; then
  systemctl --user stop acesvision-gui.service || true
  systemctl --user reset-failed acesvision-gui.service || true
  systemctl --user daemon-reload
fi
systemctl --user start acesvision-gui.service

printf '%s\n' "Installed AcesVision. Find it in the application launcher or pin it to the panel."
