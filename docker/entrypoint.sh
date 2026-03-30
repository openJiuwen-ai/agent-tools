#!/bin/sh
set -e

# 镜像内 nginx.conf 占位为 jiuwen-backend:8100，启动时替换为实际上游。
# 未设置 BACKEND_UPSTREAM 时默认 localhost:8100（可解析，避免无后端时 Nginx 因 DNS 失败起不来）。
# 有后端时请设置，例如：marketplace:8100
# Docker 内若要访问宿主机后端：host.docker.internal:8100（Desktop）或见文档
BACKEND_UPSTREAM="${BACKEND_UPSTREAM:-localhost:8100}"
sed -i "s|jiuwen-backend:8100|${BACKEND_UPSTREAM}|g" /etc/nginx/nginx.conf

exec nginx -g "daemon off;"
