# Virtual Try-On API

This project provides a virtual try-on API that allows you to superimpose clothing items (tops or bottoms) onto a person's image using a ComfyUI backend.

## Features

-   **Virtual Try-On:** Swap clothing on a person's image with a new garment.
-   **Tops and Bottoms:** Supports both upper-body and lower-body garments.
-   **FastAPI Backend:** A robust and easy-to-use API built with FastAPI.
-   **Image Validation:** Checks for valid image formats and file sizes.
-   **ComfyUI Integration:** Communicates with a ComfyUI server to process the images.
-   **Health Check & Debugging:** Endpoints to monitor the API and ComfyUI connection status.
-   **Batch Processing:** Process multiple garments for a single person in a batch.

## Requirements

-   Python 3.8+
-   A running ComfyUI server with the necessary custom nodes and models.
-   FastAPI and other Python packages listed in `pyproject.toml`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd comfyuiapi
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If a `requirements.txt` file is not available, you can generate one from `pyproject.toml` or install the dependencies manually.)*

3.  **Set up your ComfyUI server:**
    -   Ensure your ComfyUI server is running and accessible.
    -   Update the `server_address` and `server_url` variables in `comfy.py` to point to your ComfyUI instance.

## Usage

1.  **Start the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```

2.  **Send a request to the `/try-on` endpoint:**
    -   Use a tool like `curl` or an API client like Postman.
    -   Provide the person's image, the garment image, and the `garment_type` (`top` or `bottom`).

    **Example using `curl`:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/try-on" \
         -F "person_image=@/path/to/person.jpg" \
         -F "outfit_image=@/path/to/garment.png" \
         -F "garment_type=top" \
         -o result.png
    ```

## API Endpoints

-   `POST /try-on`: The main endpoint for performing the virtual try-on.
    -   **Parameters:**
        -   `person_image`: The image of the person.
        -   `outfit_image`: The image of the garment.
        -   `garment_type`: The type of garment (`top` or `bottom`).
-   `GET /`: A welcome message and basic API information.
-   `GET /health`: Checks the health of the API and the connection to the ComfyUI server.
-   `GET /debug`: Provides debugging information about the ComfyUI connection.
-   `GET /hello`: A simple endpoint to check if the API is running.

## Configuration

-   **ComfyUI Server Address:** The `server_address` and `server_url` variables in `comfy.py` must be configured to point to your ComfyUI instance.
-   **Garment Directories:** The `comfy.py` file looks for pre-stored garments in the `garments/tops` and `garments/bottoms` directories for batch processing.
