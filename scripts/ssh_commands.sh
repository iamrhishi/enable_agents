ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no rhishi@34.70.101.143 <<REMOTE
ls -la ~/enable_agents/agent-app/build 2>/dev/null || (cd ~/enable_agents/agent-app && npm run build 2>&1 | tail -20)
sudo mkdir -p /usr/share/nginx/html
sudo cp -r ~/enable_agents/agent-app/build/* /usr/share/nginx/html/ 2>/dev/null || echo "Copy failed"
sudo ls -la /usr/share/nginx/html/ | head -15
sudo systemctl restart nginx
curl -s http://localhost/ | head -20
curl -s http://localhost/api/ | head -20
REMOTE
