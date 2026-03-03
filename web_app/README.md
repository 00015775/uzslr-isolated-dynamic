# Web Interface

> Docker Image: https://hub.docker.com/repository/docker/00015775/uzslr-web

## Local Hosting

In order to ensure that the main `uzslr-signs` stays intact, no dependency conflits occur, create a new conda env by cloning existing packages from `uzslr-signs`.

> `requirements-local.txt` for local hosting.

```
# first
conda create -n web-uzslr-signs --clone uzslr-signs

# activate conda env
conda activate web-uzslr-signs

# then
cd web_app

# web packages
pip install -r requirements-local.txt

# go back to root path
cd ..

# run locally (without llm)
uvicorn web_app.backend.main:app --reload --port 8000

# run locally (llm enabled)
LLM_ENABLED=true uvicorn web_app.backend.main:app --reload --port 8000            
```

## Dockerization

> `requirements-docker.txt` for docker image.

Must force `linux/amd64` and **not** `linux/arm64` if building image in Apple Sillicon

```
# build — takes 5-10 min first time 
docker build --platform=linux/amd64 -f web_app/Dockerfile -t uzslr-web .

# run
docker run -p 7860:7860 uzslr-web

# open
open http://localhost:7860
```
