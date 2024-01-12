FROM node:20.10.0-alpine as build-stage

WORKDIR /app
COPY yarn.lock ./
COPY package.json ./

RUN yarn install --immutable

COPY ./ .

RUN yarn run build

FROM nginx as production-stage

COPY nginx.conf /etc/nginx/nginx.conf

COPY --from=build-stage /app/dist /app


ENV NGINX_ENVSUBST_OUTPUT_DIR /app/assets

WORKDIR /app/assets
RUN mkdir /etc/nginx/templates; \
    for file in *.js; \
    do \
        mv -- "$file" "/etc/nginx/templates/$file.template"; \
    done
