web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 --max-requests 100 --max-requests-jitter 10 --worker-class sync --preload

