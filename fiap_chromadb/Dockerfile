# ChromaDB Dockerfile  
FROM chromadb/chroma:latest

# Create data directory
RUN mkdir -p /chroma/data

# Set working directory
WORKDIR /chroma

# Expose ChromaDB port
EXPOSE 8200

# Start ChromaDB server with the correct syntax
ENTRYPOINT ["chroma"]
CMD ["run", "--host", "0.0.0.0", "--port", "8200", "--path", "/chroma/data"]
