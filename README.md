
# CS:GO Tournament Simulation and Prediction API

## Description

This project provides an API to predict CS:GO match outcomes and simulate tournaments. It uses machine learning models to evaluate team performance and offers endpoints for match simulation, team statistics, and tournament simulations.

## Features

- **Match Outcome Prediction**: Predicts the winner between two teams using logistic regression.
- **Tournament Simulation**: Simulates tournaments with 2, 4, 8, 16, or 32 teams.
- **Random Tournament Simulation**: Generates random tournaments with predefined teams.
- **Team Information**: Retrieves detailed information about available teams.
- **Model Metrics**: Provides metrics like accuracy and classification reports for the trained model.

## Requirements

- **Backend**:
  - Python 3.9+
  - Flask
  - Flasgger (for API documentation)
  - Scikit-learn
  - Pandas
  - Pipenv (for dependency management)

- **Frontend**:
  - Node.js
  - Vite
  - jQuery
  - Axios

## Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/api_cs_go.git
   cd api_cs_go
   ```

2. Install Python dependencies:
   ```bash
   pipenv install
   ```

3. Activate the Pipenv environment:
   ```bash
   pipenv shell
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

2. Run the frontend in development mode:
   ```bash
   npm run dev
   ```

3. Build the frontend for production:
   ```bash
   npm run build
   ```

## API Endpoints

### `/api/match_between` [POST]
Predicts the winner between two teams.

- **Request Body**:
  ```json
  {
    "id_team_1": 1,
    "id_team_2": 2
  }
  ```

### `/api/get_metrics` [GET]
Returns accuracy and classification report for the model.

### `/api/simulate_tournament` [POST]
Simulates a tournament with given teams.

- **Request Body**:
  ```json
  {
    "teams": [1, 2, 3, 4, 5, 6, 7, 8]
  }
  ```

### `/api/simulate_random_tournament` [GET]
Simulates a random tournament with 8 teams.

### `/api/get_teams` [GET]
Retrieves the list of available teams.

### `/` [GET]
Renders the index page.

## Usage

- Use the provided API documentation (powered by Flasgger) at `/apidocs`.
- Interact with the endpoints using tools like Postman or cURL.

## Development

- To run both backend and frontend in development mode:
  ```bash
  npm run start
  ```

## License

This project is licensed under the ISC License. See the `LICENSE` file for details.
