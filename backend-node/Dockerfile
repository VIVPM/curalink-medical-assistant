# Express API (Node), run directly by node. Build context is the repo root.
FROM node:22-alpine

ENV NODE_ENV=production
WORKDIR /app

# package files first so the install layer survives code edits.
COPY backend-node/package.json backend-node/package-lock.json ./
RUN npm ci --omit=dev

COPY backend-node/ ./

# Non-root; the app writes nothing to disk.
RUN addgroup -S app && adduser -S app -G app && chown -R app /app
USER app

EXPOSE 4000

# Real health endpoint, not a TCP probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:4000/health || exit 1

CMD ["node", "index.js"]
