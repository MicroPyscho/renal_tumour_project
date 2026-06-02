FROM nginx:alpine
COPY templates/index.html /usr/share/nginx/html/index.html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 7860
CMD ["nginx", "-g", "daemon off;"]