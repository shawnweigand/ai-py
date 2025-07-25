# Define variables
DOCKER_COMPOSE = docker-compose
APP_CONTAINER = app  # Change this to the name of your app container
SHELL_COMMAND = /bin/bash  # Or /bin/sh, depending on your container

.PHONY: up install py #exec run

# Bring up the Docker Compose services
up:
	$(DOCKER_COMPOSE) up -d

# Execute into the app container
# exec:
# 	$(DOCKER_COMPOSE) exec $(APP_CONTAINER) $(SHELL_COMMAND)

# Combined target to run both up and exec
# run: up exec

install:
	pip install -r requirements.txt

py:
	source .venv/bin/activate