# Builds the production Express API container.

FROM node:22-alpine

ENV NODE_ENV=production
WORKDIR /app

COPY backend-node/package.json backend-node/package-lock.json ./
RUN npm ci --omit=dev

COPY backend-node/ ./

RUN addgroup -S app && adduser -S app -G app && chown -R app /app
USER app

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:4000/health || exit 1

CMD ["node", "index.js"]
