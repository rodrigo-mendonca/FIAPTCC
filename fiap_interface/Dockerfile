# Multi-stage build for React app
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

    # Install dependencies
    RUN npm ci
    
    # Copy source code
COPY . .

# Set environment variables for production build
ENV NODE_ENV=production

# Build the app
RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Create non-root user
RUN addgroup -g 1001 -S nodejs \
    && adduser -S react -u 1001 -G nodejs

# Change ownership of nginx directories
RUN chown -R react:nodejs /var/cache/nginx \
    && chown -R react:nodejs /var/log/nginx \
    && chown -R react:nodejs /etc/nginx/conf.d

# Create nginx.pid file with correct permissions
RUN touch /var/run/nginx.pid \
    && chown -R react:nodejs /var/run/nginx.pid

# Switch to non-root user
USER react

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
