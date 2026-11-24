FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md CITATION.cff LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests
COPY reproducibility ./reproducibility

RUN pip install --no-cache-dir -e .

CMD ["pytest", "-q", "tests/"]
