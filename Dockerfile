# Base stage with pnpm installed
FROM node:20-alpine AS base
RUN npm install -g pnpm

# Build stage
FROM base AS builder
WORKDIR /app

# Copy package files for dependency installation
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/shared/package.json ./packages/shared/
COPY packages/core/package.json ./packages/core/
COPY packages/server/package.json ./packages/server/
COPY packages/cli/package.json ./packages/cli/
COPY packages/ui/package.json ./packages/ui/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source code and build
COPY . .
RUN pnpm build

# Runtime stage
FROM base AS runtime
WORKDIR /app

# Copy only built artifacts
COPY --from=builder /app/dist ./dist

# Create config directory
RUN mkdir -p /root/.claude-code-router

# Expose port
EXPOSE 3456

# Start CCR
CMD ["node", "dist/cli.js", "start"]
