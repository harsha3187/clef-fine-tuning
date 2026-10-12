#!/bin/sh
# Start the Adaptive Intelligence Engine. Prints one line about how it will sign in to Azure, then hands over to the server.
set -eu

mode="${AIE_AZURE_AUTH:-cli}"
case "$mode" in
  managed-identity)
    if [ -n "${IDENTITY_ENDPOINT:-}" ]; then echo "azure sign-in: managed-identity protocol at $IDENTITY_ENDPOINT (the host token broker, or the platform)"
    else echo "azure sign-in: managed identity"; fi ;;
  *)
    if [ -n "${AZURE_CLIENT_SECRET:-}" ] && [ -n "${AZURE_CLIENT_ID:-}" ] && [ -n "${AZURE_TENANT_ID:-}" ]; then
      echo "azure sign-in: service principal from environment (AIE_AZURE_AUTH=$mode)"
    elif [ -n "${IDENTITY_ENDPOINT:-}" ] && [ -n "${IDENTITY_HEADER:-}" ]; then
      echo "azure sign-in: token from $IDENTITY_ENDPOINT (AIE_AZURE_AUTH=$mode)"
    elif command -v az >/dev/null 2>&1 && az account show >/dev/null 2>&1; then
      echo "azure sign-in: Azure CLI session $(az account show --query user.name -o tsv 2>/dev/null)"
    else
      echo "azure sign-in: NOT SIGNED IN. The page will load, but model calls will fail until you sign in:"
      echo "  docker compose exec app az login --use-device-code      (the session is kept in a volume)"
      echo "  or run the token broker on the host: python -m routing_lab.token_broker --write-env .env   (then restart the container)"
      echo "  or set AZURE_TENANT_ID, AZURE_CLIENT_ID and AZURE_CLIENT_SECRET (and AIE_AZURE_AUTH=default) in .env"
    fi
    ;;
esac

exec python -m agent_demo.server "$@"
