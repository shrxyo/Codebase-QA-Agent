# Stage 1: build the static bundle with Node.
FROM node:20-alpine AS build
WORKDIR /app

# Vite inlines VITE_* vars into the JS at build time, so the backend URL is a
# build arg, not a runtime env var. It is read by the *browser*, so it must be
# a URL the user's browser can reach (e.g. http://localhost:8000), not a
# compose-internal hostname like http://backend:8000.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# package-lock.json is gitignored, so fall back to npm install on a clean clone.
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

# Stage 2: serve the bundle with nginx. Node and node_modules stay behind.
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
