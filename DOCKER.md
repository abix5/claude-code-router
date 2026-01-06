# Docker Deployment

## Quick Start

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up -d --build

# Stop
docker-compose down
```

## Configuration

Place your `config.json` in `./data/` directory - it will be mounted to `/root/.claude-code-router` inside container.

## Environment Variables

- `LOG_STDOUT=true` - Output logs to stdout (enabled by default in docker-compose)
- `LOG_LEVEL` - Log level: fatal/error/warn/info/debug/trace (default: debug)
- `NODE_ENV` - Environment mode (production/development)
