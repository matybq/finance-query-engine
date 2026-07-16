#!/usr/bin/env bash
# One-time VPS setup for web UI deployment: a dedicated `deploy` user that
# owns only the static root, the nginx server block from deploy/nginx.conf,
# and a first manual deploy. After this, .github/workflows/deploy.yml keeps
# the UI updated on every push to main.
#
# Run from the repo root on a machine with root SSH access to the VPS.
set -euo pipefail

HOST=187.127.9.91
KEY=$HOME/.ssh/fqe-deploy

[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C fqe-deploy -f "$KEY"

[ -d frontend/dist ] || (cd frontend && npm ci && npm run build)

scp deploy/nginx.conf "root@$HOST:/tmp/fqe-nginx.conf"

ssh "root@$HOST" "PUBKEY='$(cat "$KEY.pub")' bash -s" <<'EOF'
set -euo pipefail
id -u deploy >/dev/null 2>&1 || useradd -m -s /bin/sh deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
grep -qF "$PUBKEY" /home/deploy/.ssh/authorized_keys 2>/dev/null \
  || echo "$PUBKEY" >> /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys
install -d -o deploy -g deploy /var/www/finance-query-engine

# Clean up any backup a previous run left inside sites-enabled — nginx
# includes every file there, so it would duplicate the default_server.
for f in /etc/nginx/sites-enabled/*.bak; do
  [ -e "$f" ] || continue
  dest="/etc/nginx/$(basename "$f" .bak).orig"
  [ -f "$dest" ] || cp "$f" "$dest"
  rm -f "$f"
done

target=$(grep -l default_server /etc/nginx/sites-enabled/* | head -1)
if grep -q locus "$target"; then
  echo "ABORT: the default_server block shares a file with unrelated site config; split it first" >&2
  exit 1
fi
# The backup must live outside sites-enabled: nginx includes every file
# there, so a copy would register a duplicate default_server.
backup="/etc/nginx/$(basename "$target").orig"
[ -f "$backup" ] || cp "$target" "$backup"
mv /tmp/fqe-nginx.conf "$target"
nginx -t
systemctl reload nginx
echo "server setup ok: $target (original saved at $backup)"
EOF

rsync -az --delete -e "ssh -i $KEY" frontend/dist/ "deploy@$HOST:/var/www/finance-query-engine/"
echo "first deploy ok: http://$HOST/"
echo "next: gh secret set DEPLOY_SSH_KEY < $KEY"
