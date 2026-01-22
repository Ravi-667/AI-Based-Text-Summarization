# Render.com Deployment Guide

## Quick Deploy to Render.com

### Prerequisites
✅ GitHub account with repository access
✅ Render.com account (free tier available)

### Deployment Steps

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +" → "Web Service"**
3. **Connect GitHub Repository**:
   - Authorize Render to access GitHub
   - Select: `AI-Based-Text-Summarization`
4. **Configure Service**:
   - **Name**: `quicksummary-api`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free
5. **Environment Variables**:
   - Already configured in render.yaml
   - DEVICE=cpu
   - LOG_LEVEL=INFO
6. **Click "Create Web Service"**
7. **Wait for deployment** (~5-10 minutes)

### After Deployment

Your API will be available at:
```
https://quicksummary-api.onrender.com
```

**API Documentation:**
```
https://quicksummary-api.onrender.com/docs
```

**Health Check:**
```
https://quicksummary-api.onrender.com/api/v1/health
```

### Test Your Deployment

```bash
curl -X POST "https://quicksummary-api.onrender.com/api/v1/summarize/extractive" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "ratio": 0.3
  }'
```

### Custom Domain (Optional)

1. Purchase domain (e.g., QuickSummary.io)
2. In Render dashboard → Settings → Custom Domains
3. Add your domain
4. Configure DNS records as shown by Render

### Important Notes

⚠️ **Free Tier Limitations:**
- Service spins down after 15 min of inactivity
- First request after sleep takes ~30 seconds
- 750 hours/month free
- Model loading on first request (~30s)

💡 **Upgrade to Paid Plan for:**
- Always-on service
- Faster cold starts
- More resources
- Better performance

### Troubleshooting

**Build Failed?**
- Check Dockerfile syntax
- Verify requirements.txt dependencies
- Check build logs in Render dashboard

**Service Not Starting?**
- Check health check endpoint
- Verify PORT environment variable
- Check application logs

**Slow Response?**
- Free tier has resource limits
- Consider upgrading plan
- Models load on first request

### Support

- Render Docs: https://render.com/docs
- Community: https://community.render.com
