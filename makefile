# Makefile

# Target to set up the environment
setting_up:
	@echo "Creating a database..."
	docker volume create strorage  # Create a Docker volume named 'strorage'
	docker volume ls  # List all Docker volumes
	docker volume inspect strorage  # Inspect the created 'strorage' volume

# Target to start the services
up:
	@echo "Starting up services..."
	cd deployment && docker compose up -d server  # Navigate to 'deployment' directory and start Docker services in detached mode
	cd deployment && docker compose up -d streamlit
	@echo "UI running on http://localhost:8501/"
	@echo "Server running on http://0.0.0.0:8000/docs/"

# Target to stop the services
down:
	@echo "Stopping services..."
	cd deployment && docker compose down  # Navigate to 'deployment' directory and stop Docker services

