#!/usr/bin/env bash
# Compile + flash the CameraWebServer firmware to an AI-Thinker ESP32-CAM.
# Auto-reset isn't wired on this board, so put it in bootloader mode first:
#   jumper IO0 -> GND, tap RST, then run this. Remove the jumper + tap RST after.
#
#   ./flash.sh [/dev/ttyUSB0]
set -euo pipefail
export PATH="$HOME/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
SKETCH="$HERE/CameraWebServer"
PORT="${1:-/dev/ttyUSB0}"
FQBN="esp32:esp32:esp32cam"
CORE="$HOME/.arduino15/packages/esp32/hardware/esp32/3.3.7"
ESPTOOL="$(find "$HOME/.arduino15/packages/esp32/tools/esptool_py" -maxdepth 2 -name esptool -type f | head -1)"
BOOT_APP0="$CORE/tools/partitions/boot_app0.bin"

echo "[1/2] compiling..."
arduino-cli compile --fqbn "$FQBN" --output-dir "$HERE/build" "$SKETCH" >/dev/null
echo "      ok"

echo "[2/2] flashing $PORT (board must be in bootloader mode)..."
"$ESPTOOL" --chip esp32 --port "$PORT" --baud 460800 \
  --before no_reset --after hard_reset write_flash -z \
  --flash_mode dio --flash_freq 40m --flash_size 4MB \
  0x1000  "$HERE/build/CameraWebServer.ino.bootloader.bin" \
  0x8000  "$HERE/build/CameraWebServer.ino.partitions.bin" \
  0xe000  "$BOOT_APP0" \
  0x10000 "$HERE/build/CameraWebServer.ino.bin"
echo "done. Remove the IO0->GND jumper and tap RST to run."
