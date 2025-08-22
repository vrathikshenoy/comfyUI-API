from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image
import zipfile
import os
from datetime import datetime

# Import your updated ComfyUI service functions
from comfy import (
    batch_try_on_sequential,
    batch_try_on_sync,
    get_available_garments,
    ensure_garments_directories,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(image_file) -> str:
    """Convert uploaded image to base64"""
    image = Image.open(image_file)

    # Convert to RGB if necessary
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Resize if too large (optional)
    max_size = (1024, 1024)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str


@app.post("/batch-try-on")
async def batch_try_on_endpoint(person_image: UploadFile = File(...)):
    """
    Batch try-on endpoint that processes person image with all stored garments
    """
    try:
        # Validate uploaded file
        if not person_image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Please upload a valid image file"
            )

        # Convert person image to base64
        person_base64 = image_to_base64(person_image.file)

        # Process batch try-on
        # Use sequential processing for stability (change to batch_try_on_sync for async)
        result = batch_try_on_sequential(person_base64)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch try-on failed: {str(e)}")


@app.get("/available-garments")
async def get_available_garments_endpoint():
    """
    Get list of available garments in the project directory
    """
    try:
        garments = get_available_garments()
        return JSONResponse(
            content={
                "success": True,
                "garments": garments,
                "total_tops": len(garments["tops"]),
                "total_bottoms": len(garments["bottoms"]),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get garments: {str(e)}")


@app.post("/batch-try-on-download")
async def batch_try_on_with_download(person_image: UploadFile = File(...)):
    """
    Batch try-on with downloadable ZIP containing all results
    """
    try:
        # Validate uploaded file
        if not person_image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Please upload a valid image file"
            )

        # Convert person image to base64
        person_base64 = image_to_base64(person_image.file)

        # Process batch try-on
        result = batch_try_on_sequential(person_base64)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=result.get("message", "Batch processing failed")
            )

        # Create ZIP file with all successful results
        zip_buffer = BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add successful results to ZIP
            for i, item in enumerate(result["results"]):
                if item["success"]:
                    # Decode base64 image
                    image_data = base64.b64decode(item["result_image"])

                    # Create filename
                    filename = (
                        f"{i + 1:02d}_{item['garment_type']}_{item['garment_filename']}"
                    )

                    # Add to ZIP
                    zip_file.writestr(filename, image_data)

            # Add summary JSON
            summary = {
                "total_processed": result["total_processed"],
                "successful": result["successful"],
                "failed": result["failed"],
                "timestamp": timestamp,
                "results_summary": [
                    {
                        "garment_name": item["garment_name"],
                        "garment_type": item["garment_type"],
                        "success": item["success"],
                    }
                    for item in result["results"]
                ],
            }
            zip_file.writestr("summary.json", json.dumps(summary, indent=2))

        zip_buffer.seek(0)

        # Return ZIP file
        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=try_on_results_{timestamp}.zip"
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch try-on with download failed: {str(e)}"
        )


# Initialize garment directories on startup
@app.on_event("startup")
async def startup_event():
    ensure_garments_directories()
    print("Garment directories initialized")

    # Print available garments
    garments = get_available_garments()
    print(
        f"Available garments: {len(garments['tops'])} tops, {len(garments['bottoms'])} bottoms"
    )
