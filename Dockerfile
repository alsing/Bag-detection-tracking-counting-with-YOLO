FROM ultralytics/ultralytics
WORKDIR /app
COPY ./ ./
RUN pip install -r requirements.txt
CMD [ "python", "src/bagcounting.py" ]