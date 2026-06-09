# Production Dockerfile for TCC (multi-stage)

# --- Stage 1: build ---
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies (cached unless package files change)
COPY package*.json ./
RUN npm ci

# Copy source code and build static assets
COPY . .
RUN npm run build

# --- Stage 2: serve ---
FROM node:18-alpine

WORKDIR /app

# Lightweight static file server
RUN npm install -g serve

# Copy only the build output from the previous stage
COPY --from=build /app/build ./build

# Expose port
EXPOSE 3000

# Serve the static build as a SPA (fallback to index.html)
CMD ["serve", "-s", "build", "-l", "3000"]
