#!/bin/sh
set -eu

TLS_ENABLE="${NGINX_TLS_ENABLE:-0}"
SERVER_NAME="${NGINX_SERVER_NAME:-_}"
TLS_CERT_FILE="${NGINX_TLS_CERT_FILE:-/etc/nginx/certs/fullchain.pem}"
TLS_KEY_FILE="${NGINX_TLS_KEY_FILE:-/etc/nginx/certs/privkey.pem}"

CONFIG_SRC="/etc/nginx/nginx-http.conf"
if [ "$TLS_ENABLE" = "1" ]; then
  if [ -f "$TLS_CERT_FILE" ] && [ -f "$TLS_KEY_FILE" ]; then
    CONFIG_SRC="/etc/nginx/nginx-https.conf"
  else
    echo "warning: TLS enabled but certificate files were not found; starting HTTP mode" >&2
  fi
fi

sed \
  -e "s|__SERVER_NAME__|$SERVER_NAME|g" \
  -e "s|__TLS_CERT_FILE__|$TLS_CERT_FILE|g" \
  -e "s|__TLS_KEY_FILE__|$TLS_KEY_FILE|g" \
  "$CONFIG_SRC" > /tmp/nginx.conf

exec nginx -c /tmp/nginx.conf -g "daemon off;"
