# Schema Profiler - ヘッドレス版
FROM python:3.11-slim

# 作業ディレクトリ設定
WORKDIR /app

# システム依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# 環境変数設定
ENV PYTHONPATH=/app
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=production

# ポート公開
EXPOSE 5000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 実行ユーザー作成（セキュリティ）
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# アプリケーション起動
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]