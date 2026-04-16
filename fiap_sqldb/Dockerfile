FROM postgres:15

# Set environment variables
ENV POSTGRES_DB=fiap_db
ENV POSTGRES_USER=fiap_user
ENV POSTGRES_PASSWORD=admin

# Install MCP extension dependencies and Python
RUN apt-get update && apt-get install -y \
    postgresql-contrib \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/

# Expose default PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pg_isready -U admin -d loja