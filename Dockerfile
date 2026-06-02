FROM nginx:alpine
COPY templates/index.html /usr/share/nginx/html/index.html
EXPOSE 7860
CMD ["nginx", "-g", "daemon off;"]
