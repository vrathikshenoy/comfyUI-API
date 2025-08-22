from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import comfyuiservice
import logging
from typing import Optional
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Virtual Try-On API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Test ComfyUI connection on startup with detailed diagnostics"""
    logger.info("Testing connection to ComfyUI server...")

    try:
        if comfyuiservice.test_connection():
            logger.info("✅ ComfyUI server is reachable")

            # Test additional endpoints
            endpoint_results = comfyuiservice.test_comfyui_endpoints()
            for endpoint, result in endpoint_results.items():
                if result["success"]:
                    logger.info(f"✅ {endpoint}: HTTP {result['status']}")
                else:
                    logger.warning(f"⚠️ {endpoint}: {result.get('error', 'Failed')}")
        else:
            logger.warning("⚠️ ComfyUI server is not reachable - operations may fail")
            logger.info("Run the debug script to diagnose connection issues")

    except Exception as e:
        logger.error(f"Error during startup connection test: {e}")


@app.post("/try-on")
async def try_on_outfit(
    person_image: UploadFile = File(..., description="Person image"),
    outfit_image: UploadFile = File(..., description="Outfit image"),
    garment_type: Optional[str] = Form(
        "bottom", description="Type of garment: 'top' or 'bottom'"
    ),
):
    try:
        logger.info(
            f"Received try-on request: person={person_image.filename}, outfit={outfit_image.filename}, garment_type={garment_type}"
        )

        # Validate garment type
        if garment_type not in ["top", "bottom"]:
            raise HTTPException(
                status_code=400, detail="garment_type must be either 'top' or 'bottom'"
            )

        # Validate file types
        valid_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]

        if person_image.content_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Person image must be one of: {', '.join(valid_types)}. Received: {person_image.content_type}",
            )

        if outfit_image.content_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Outfit image must be one of: {', '.join(valid_types)}. Received: {outfit_image.content_type}",
            )

        # Check file sizes (limit to 10MB each)
        max_size = 10 * 1024 * 1024  # 10MB
        if person_image.size and person_image.size > max_size:
            raise HTTPException(
                status_code=400, detail="Person image file too large (max 10MB)"
            )
        if outfit_image.size and outfit_image.size > max_size:
            raise HTTPException(
                status_code=400, detail="Outfit image file too large (max 10MB)"
            )

        # Read image files
        logger.info("Reading uploaded images...")
        person_bytes = await person_image.read()
        outfit_bytes = await outfit_image.read()

        logger.info(
            f"Image sizes: person={len(person_bytes)} bytes, outfit={len(outfit_bytes)} bytes"
        )

        # Convert to base64
        logger.info("Converting images to base64...")
        person_base64 = comfyuiservice.image_to_base64(person_bytes)
        outfit_base64 = comfyuiservice.image_to_base64(outfit_bytes)

        logger.info(
            f"Base64 lengths: person={len(person_base64)}, outfit={len(outfit_base64)}"
        )

        # Test connection before processing
        if not comfyuiservice.test_connection():
            logger.error("ComfyUI server is not reachable")
            raise HTTPException(
                status_code=503,
                detail="ComfyUI server is not available. Please try again later.",
            )

        # Process through ComfyUI with garment type
        logger.info(f"Sending request to ComfyUI for {garment_type} garment...")
        result_image = comfyuiservice.fetch_image_from_comfy(
            person_base64, outfit_base64, garment_type
        )

        if result_image is None:
            logger.error("ComfyUI returned None - check ComfyUI logs")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate image. The ComfyUI service may be overloaded or misconfigured. Please check that all required models are loaded and try again.",
            )

        logger.info(
            f"Successfully generated {garment_type} image: {len(result_image)} bytes"
        )

        # Return the image
        image_stream = io.BytesIO(result_image)
        return StreamingResponse(
            image_stream,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=try-on-{garment_type}-result.png"
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        # Log the full traceback for debugging
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}. Check server logs for details.",
        )


@app.get("/")
def read_root():
    return {
        "message": "Virtual Try-On API",
        "version": "1.0.0",
        "endpoints": {
            "try-on": "POST /try-on - Upload person and outfit images for virtual try-on. Supports garment_type parameter ('top' or 'bottom')",
            "health": "GET /health - Check API health",
            "debug": "GET /debug - Debug connection to ComfyUI",
            "hello": "GET /hello - Simple hello message",
        },
        "parameters": {
            "garment_type": {
                "description": "Type of garment to try on",
                "options": ["top", "bottom"],
                "default": "bottom",
                "mask_components": {"top": "5,14,15", "bottom": "17,16,9,6"},
            }
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint with detailed status"""
    try:
        comfy_status = comfyuiservice.test_connection()

        if comfy_status:
            # Test additional endpoints for more detailed health check
            endpoint_results = comfyuiservice.test_comfyui_endpoints()
            failed_endpoints = [
                ep for ep, result in endpoint_results.items() if not result["success"]
            ]

            if failed_endpoints:
                return {
                    "status": "degraded",
                    "comfyui_connection": "partial",
                    "message": f"API is running but some ComfyUI endpoints are not accessible: {failed_endpoints}",
                    "endpoint_details": endpoint_results,
                }
            else:
                return {
                    "status": "healthy",
                    "comfyui_connection": "ok",
                    "message": "API is running and ComfyUI is fully accessible",
                    "endpoint_details": endpoint_results,
                }
        else:
            return {
                "status": "degraded",
                "comfyui_connection": "failed",
                "message": "API is running but ComfyUI is not reachable",
            }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "comfyui_connection": "error",
            "message": f"Health check failed: {str(e)}",
        }


@app.get("/debug")
def debug_comfyui():
    """Debug endpoint to test ComfyUI connectivity"""
    try:
        # Test basic connection
        basic_test = comfyuiservice.test_connection()

        # Test specific endpoints
        endpoint_results = comfyuiservice.test_comfyui_endpoints()

        return {
            "basic_connection": basic_test,
            "endpoint_tests": endpoint_results,
            "server_url": comfyuiservice.server_url,
            "server_address": comfyuiservice.server_address,
            "recommendations": [
                "If basic_connection is False, check if ComfyUI server is running",
                "If endpoint tests fail, verify ComfyUI configuration",
                "Run the debug_test_script.py for more detailed diagnostics",
                "Check server logs for any error messages",
            ],
        }

    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return {
            "error": str(e),
            "message": "Debug test failed - check server logs for details",
        }


@app.get("/hello")
def read_hello():
    return {
        "message": "Hello World! Virtual Try-On API is running with top/bottom garment support!",
        "version": "1.0.0",
        "status": "operational",
    }


# Add exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Check server logs for details.",
            "type": type(exc).__name__,
        },
    )
