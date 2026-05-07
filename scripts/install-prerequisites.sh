#!/usr/bin/env bash
#
# One-time prerequisites for Enable Agents (Docker Engine + Compose plugin).
#
# Linux: installs via https://get.docker.com (official), enables systemd service,
#        adds SUDO_USER to the docker group (new login required for group to apply).
# macOS: prefers Homebrew --cask docker (Docker Desktop); otherwise prints manual steps.
#
# Usage:
#   ./scripts/install-prerequisites.sh
#   curl ... | sudo bash   # not recommended — prefer running the file after review
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

say() { printf '%s\n' "$*"; }

need_sudo_linux() {
  if [ "${EUID:-0}" -ne 0 ]; then
    say "Re-running with sudo for Docker installation..."
    exec sudo -E env SUDO_USER="${SUDO_USER:-$USER}" bash "$0" "$@"
  fi
}

linux_install_docker() {
  need_sudo_linux "$@"

  # Docker without Compose v2 plugin
  if command -v docker >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
    say "Installing docker-compose-plugin..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-compose-plugin || true
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y docker-compose-plugin || true
    elif command -v yum >/dev/null 2>&1; then
      yum install -y docker-compose-plugin || true
    fi
  fi

  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      say "Docker and Docker Compose are already installed and the daemon is running."
    else
      say "Docker is installed but the daemon is not running. Starting..."
      systemctl enable docker 2>/dev/null || true
      systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
    fi
    if command -v apt-get >/dev/null 2>&1; then
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq lsof 2>/dev/null || true
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y lsof 2>/dev/null || true
    fi
    return 0
  fi

  say "Installing Docker Engine using the official Docker install script..."
  if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq
      apt-get install -y -qq ca-certificates curl
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y curl ca-certificates
    elif command -v yum >/dev/null 2>&1; then
      yum install -y curl ca-certificates
    else
      say "Install curl, then re-run this script."
      exit 1
    fi
  fi

  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sh /tmp/get-docker.sh
  rm -f /tmp/get-docker.sh

  if ! docker compose version >/dev/null 2>&1; then
    say "Installing docker-compose-plugin..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-compose-plugin || true
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y docker-compose-plugin || true
    fi
  fi

  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq lsof 2>/dev/null || true
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y lsof 2>/dev/null || true
  fi

  if command -v systemctl >/dev/null 2>&1; then
    systemctl enable docker
    systemctl start docker
  elif command -v service >/dev/null 2>&1; then
    service docker start || true
  fi

  local u="${SUDO_USER:-}"
  if [ -n "$u" ] && id "$u" >/dev/null 2>&1; then
    usermod -aG docker "$u" || true
    say ""
    say "Added user '$u' to the docker group."
    say "Log out and back in (or run: newgrp docker) so you can use docker without sudo."
  fi
}

macos_install_hint() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      say "Docker is installed and running."
      return 0
    fi
    say "Docker is installed but not running. Open Docker Desktop, wait until it is ready, then retry."
    return 1
  fi

  if command -v brew >/dev/null 2>&1; then
    say "Installing Docker Desktop via Homebrew..."
    brew install --cask docker
    say "Open Docker Desktop from Applications and wait until it finishes starting."
  else
    say "Install Docker Desktop for Mac: https://docs.docker.com/desktop/install/mac-install/"
    say "Or install Homebrew (https://brew.sh) and re-run this script."
    return 1
  fi
}

case "$OS" in
  Linux)
    linux_install_docker "$@"
    ;;
  Darwin)
    macos_install_hint "$@" || exit 1
    ;;
  *)
    say "Unsupported OS: $OS"
    say "Install Docker Engine and the Docker Compose v2 plugin manually:"
    say "  https://docs.docker.com/engine/install/"
    exit 1
    ;;
esac

say ""
say "Verify:"
say "  docker --version"
say "  docker compose version"
say "  docker info"
