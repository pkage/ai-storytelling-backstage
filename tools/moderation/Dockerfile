FROM node:16-alpine

WORKDIR /app

COPY . .
RUN npm install
RUN cd frontend && npm install
RUN cd frontend && npx vite build

CMD [ "node", "server.mjs" ]
