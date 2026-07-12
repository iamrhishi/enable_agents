# Deployment Guide

## Infrastructure

- **GCP Project:** `enable-agents`
- **VM Instance:** `instance-20260419-210128` (us-central1-f)
- **VM Type:** e2-medium (2 vCPU, 4GB RAM, 4GB swap)
- **Domain:** enableyou.co
- **Live URL:** https://agents.enableyou.co/

## SSL/HTTPS

- HTTP automatically redirects to HTTPS (301)
- SSL certificates managed by Let's Encrypt (certbot)
- Nginx handles SSL termination inside Docker

### GCP Firewall Rules Required

```bash
# Port 80 (HTTP - for redirect)
gcloud compute firewall-rules create allow-http --allow=tcp:80

# Port 443 (HTTPS)
gcloud compute firewall-rules create allow-https --allow=tcp:443
```

### Verify HTTPS

```bash
# Should return HTTP/2 200
curl -I https://agents.enableyou.co/

# Should return 301 redirect to HTTPS
curl -I http://agents.enableyou.co/
```

## Deployment Methods

### Method 1: Build on VM (Simple)

SSH into VM and rebuild containers:

```bash
# SSH into VM
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f

# Navigate to project
cd /home/rhishi/enable_agents

# Pull latest code
sudo git fetch origin && sudo git reset --hard origin/harsh-code

# Rebuild and restart (with cache - fast)
sudo docker compose build backend-remote frontend-remote
sudo docker compose up -d backend-remote frontend-remote celery-remote beat-remote

# Or rebuild without cache (slower, use when Dockerfile/requirements change)
sudo docker compose build --no-cache backend-remote frontend-remote
sudo docker compose up -d
```

### Method 2: Build Locally, Push to GCR (Faster deploys)

Build images on local machine, push to Google Container Registry:

```bash
# Configure Docker for GCR (one-time)
gcloud auth configure-docker gcr.io

# Build images locally
docker build -t gcr.io/enable-agents/backend:latest -f backend/Dockerfile backend/
docker build -t gcr.io/enable-agents/frontend:latest \
  --build-arg REACT_APP_API_URL=https://enableyou.co \
  -f frontend/Dockerfile frontend/

# Push to GCR
docker push gcr.io/enable-agents/backend:latest
docker push gcr.io/enable-agents/frontend:latest

# On VM: Pull and restart
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="
  sudo docker pull gcr.io/enable-agents/backend:latest && \
  sudo docker pull gcr.io/enable-agents/frontend:latest && \
  cd /home/rhishi/enable_agents && \
  sudo docker compose up -d
"
```

## Quick Deploy Commands

### From local machine:

```bash
# Pull latest, build with cache, restart
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="
  cd /home/rhishi/enable_agents && \
  sudo git fetch origin && sudo git reset --hard origin/harsh-code && \
  sudo docker compose build backend-remote frontend-remote && \
  sudo docker compose up -d
"
```

### Check status:

```bash
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="sudo docker ps"
```

### View logs:

```bash
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="sudo docker compose logs -f backend-remote"
```

## Services

| Service | Container | Port |
|---------|-----------|------|
| Backend API | enable_agents_backend_remote | 8000 |
| Frontend | enable_agents_frontend_remote | 80 (internal) |
| Nginx | enable_agents_nginx | 80, 443 |
| Celery Worker | enable_agents_celery_remote | - |
| Celery Beat | enable_agents_beat_remote | - |
| Redis | enable_agents_redis | 6379 |
| MySQL | enable_agents_mysql | 3306 |

## Troubleshooting

### VM freezes during build
- Added 4GB swap to prevent OOM
- Use `--no-cache` only when requirements.txt or Dockerfile changes
- Check memory: `free -h`

### Check swap status
```bash
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="free -h && sudo swapon --show"
```

### Restart all services
```bash
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="
  cd /home/rhishi/enable_agents && sudo docker compose down && sudo docker compose up -d
"
```

### HTTPS not working
1. Check GCP firewall rules allow port 443:
```bash
gcloud compute firewall-rules list --filter="allowed:tcp:443"
```
2. Check nginx container is running:
```bash
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="sudo docker ps | grep nginx"
```
3. Check SSL certificate exists:
```bash
gcloud compute ssh instance-20260419-210128 --zone=us-central1-f --command="sudo docker exec enable_agents_nginx ls -la /etc/letsencrypt/live/"
```

## Branches

- `main` - Production stable
- `staging` - Pre-production testing
- `harsh-code` - Active development
