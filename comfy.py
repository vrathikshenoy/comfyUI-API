import os
import json
from typing import List, Dict, Any
import asyncio
import concurrent.futures
from pathlib import Path
import websocket
import uuid
import json
import urllib.request
import urllib.parse
import urllib.error
import base64
import time
import ssl
import http.cookiejar
import re


save_image_websocket = "SaveImageWebsocket"
server_address = "5dj8ees0osohp7-8188.proxy.runpod.net"
server_url = "https://5dj8ees0osohp7-8188.proxy.runpod.net"
client_id = str(uuid.uuid4())

# Global cookie jar and XSRF token
cookie_jar = http.cookiejar.CookieJar()
xsrf_token = None


def get_xsrf_token():
    """Get XSRF token from the server"""
    global xsrf_token, cookie_jar

    try:
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create opener with cookie jar and SSL context
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)
        opener = urllib.request.build_opener(https_handler, cookie_handler)

        # First, try to access the main page to get cookies and XSRF token
        req = urllib.request.Request(f"{server_url}/")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")

        response = opener.open(req, timeout=30)
        html_content = response.read().decode("utf-8", errors="ignore")

        # Look for XSRF token in the HTML or try to extract from cookies
        xsrf_match = re.search(r'_xsrf["\']?\s*[:=]\s*["\']([^"\']+)', html_content)
        if xsrf_match:
            xsrf_token = xsrf_match.group(1)
            print(f"Found XSRF token in HTML: {xsrf_token[:10]}...")
            return opener

        # If not found in HTML, look in cookies
        for cookie in cookie_jar:
            if cookie.name == "_xsrf":
                xsrf_token = cookie.value
                print(f"Found XSRF token in cookies: {xsrf_token[:10]}...")
                return opener

        # Try to get token from the API endpoint
        try:
            api_req = urllib.request.Request(f"{server_url}/api")
            api_response = opener.open(api_req, timeout=15)
            # Check cookies again after API call
            for cookie in cookie_jar:
                if cookie.name == "_xsrf":
                    xsrf_token = cookie.value
                    print(f"Found XSRF token from API: {xsrf_token[:10]}...")
                    return opener
        except:
            pass

        print("No XSRF token found, will try without it")
        return opener

    except Exception as e:
        print(f"Error getting XSRF token: {e}")
        # Return basic opener even if XSRF token retrieval fails
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)
        return urllib.request.build_opener(https_handler, cookie_handler)


def create_request_with_headers():
    """Create a request with proper headers and SSL context"""
    # Get opener with XSRF token
    opener = get_xsrf_token()

    # Create SSL context that doesn't verify certificates (for development)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    return opener, ssl_context


def get_prompt_with_workflow(person_base64, outfit_base64, garment_type="top"):
    """Generate prompt with configurable garment type"""
    prompt_text = """
{
  "1": {
    "inputs": {
      "method": "human_parsing_lip",
      "confidence": 0.4,
      "crop_multi": 0,
      "mask_components": {
        "__value__": [
          15,
          14,
          5
        ]
      },
      "image": [
        "26",
        0
      ]
    },
    "class_type": "easy humanSegmentation",
    "_meta": {
      "title": "Human Segmentation"
    }
  },
  "2": {
    "inputs": {
      "guidance": 30,
      "conditioning": [
        "19",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "3": {
    "inputs": {
      "style_model_name": "flux1-redux-dev.safetensors"
    },
    "class_type": "StyleModelLoader",
    "_meta": {
      "title": "Load Style Model"
    }
  },
  "5": {
    "inputs": {
      "grow": 20,
      "blur": 7,
      "mask": [
        "1",
        1
      ]
    },
    "class_type": "INPAINT_ExpandMask",
    "_meta": {
      "title": "Expand Mask"
    }
  },
  "6": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "7": {
    "inputs": {
      "clip_name": "sigclip_vision_patch14_384.safetensors"
    },
    "class_type": "CLIPVisionLoader",
    "_meta": {
      "title": "Load CLIP Vision"
    }
  },
  "9": {
    "inputs": {
      "lora_name": "FLUX.1-Turbo-Alpha.safetensors",
      "strength_model": 1.1,
      "model": [
        "11",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "10": {
    "inputs": {
      "clip_name1": "t5xxl_fp16.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "11": {
    "inputs": {
      "unet_name": "flux1-fill-dev.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "12": {
    "inputs": {
      "text": "",
      "clip": [
        "10",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "13": {
    "inputs": {
    "lora_name": "migrationloracloth.safetensors",
      "strength_model": 1.0000000000000002,
      "model": [
        "9",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "14": {
    "inputs": {
      "conditioning": [
        "2",
        0
      ]
    },
    "class_type": "ConditioningZeroOut",
    "_meta": {
      "title": "ConditioningZeroOut"
    }
  },
  "16": {
    "inputs": {
      "samples": [
        "23",
        0
      ],
      "vae": [
        "6",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "18": {
    "inputs": {
      "crop": "center",
      "clip_vision": [
        "7",
        0
      ],
      "image": [
        "25",
        0
      ]
    },
    "class_type": "CLIPVisionEncode",
    "_meta": {
      "title": "CLIP Vision Encode"
    }
  },
  "19": {
    "inputs": {
      "image_strength": "highest",
      "conditioning": [
        "12",
        0
      ],
      "style_model": [
        "3",
        0
      ],
      "clip_vision_output": [
        "18",
        0
      ]
    },
    "class_type": "StyleModelApplySimple",
    "_meta": {
      "title": "StyleModelApplySimple"
    }
  },
  "20": {
    "inputs": {
      "patch_mode": "patch_right",
      "output_length": 1080,
      "patch_color": "#00FF00",
      "first_image": [
        "25",
        0
      ],
      "second_image": [
        "26",
        0
      ],
      "second_mask": [
        "5",
        0
      ]
    },
    "class_type": "AddMaskForICLora",
    "_meta": {
      "title": "Add Mask For IC Lora"
    }
  },
  "21": {
    "inputs": {
      "noise_mask": true,
      "positive": [
        "2",
        0
      ],
      "negative": [
        "14",
        0
      ],
      "vae": [
        "6",
        0
      ],
      "pixels": [
        "20",
        0
      ],
      "mask": [
        "20",
        1
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "22": {
    "inputs": {
      "width": [
        "20",
        4
      ],
      "height": [
        "20",
        5
      ],
      "position": "top-left",
      "x_offset": [
        "20",
        2
      ],
      "y_offset": [
        "20",
        3
      ],
      "image": [
        "16",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "23": {
    "inputs": {
      "seed": 583967762944198,
      "steps": 8,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "13",
        0
      ],
      "positive": [
        "21",
        0
      ],
      "negative": [
        "21",
        1
      ],
      "latent_image": [
        "21",
        2
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "25": {
    "inputs": {
      "base64_string": ""
    },
    "class_type": "Base64DecodeNode",
    "_meta": {
      "title": "cloth"
    }
  },
  "26": {
    "inputs": {
      "base64_string": ""
    },
    "class_type": "Base64DecodeNode",
    "_meta": {
      "title": "person"
    }
  },
  "28": {
    "inputs": {
      "images": [
        "22",
        0
      ]
    },
    "class_type": "SaveImageWebsocket",
    "_meta": {
      "title": "SaveImageWebsocket"
    }
  }
}
    """

    prompt_json = json.loads(prompt_text)

    # Set garment type specific mask components
    mask_components = {
        "top": [15, 14, 5],  # Upper body garments
        "bottom": [14, 9, 16, 17],  # Lower body garments
    }

    # Update mask components based on garment type
    prompt_json["1"]["inputs"]["mask_components"]["__value__"] = mask_components.get(
        garment_type, [15, 14, 5]
    )

    # Adjust grow parameter based on garment type
    if garment_type == "bottom":
        prompt_json["5"]["inputs"]["grow"] = 21  # Increase grow for bottom garments
    else:
        prompt_json["5"]["inputs"]["grow"] = 20  # Keep default for top garments

    # Set the base64 images correctly for Base64DecodeNode nodes
    clean_person_base64 = person_base64
    clean_outfit_base64 = outfit_base64

    # Remove data:image/... prefix if present
    if person_base64.startswith("data:image"):
        clean_person_base64 = person_base64.split(",", 1)[1]
    if outfit_base64.startswith("data:image"):
        clean_outfit_base64 = outfit_base64.split(",", 1)[1]

    # Set the base64 strings in the Base64DecodeNode nodes
    prompt_json["25"]["inputs"]["base64_string"] = clean_outfit_base64  # cloth
    prompt_json["26"]["inputs"]["base64_string"] = clean_person_base64  # person

    return prompt_json


def queue_prompt(prompt):
    """Queue prompt with XSRF token support"""
    global xsrf_token

    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    print(f"Sending prompt to ComfyUI: {len(data)} bytes")

    try:
        opener, _ = create_request_with_headers()

        req = urllib.request.Request(f"{server_url}/prompt", data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")
        req.add_header("Accept", "application/json")

        # Add XSRF token if available
        if xsrf_token:
            req.add_header("X-XSRFToken", xsrf_token)
            print(f"Added XSRF token to request: {xsrf_token[:10]}...")

        response = opener.open(req, timeout=60)  # Increased timeout
        result = json.loads(response.read())
        print(f"Prompt queued successfully. Response: {result}")
        return result

    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        error_body = e.read().decode("utf-8", errors="ignore")
        print(f"Response body: {error_body}")

        # If XSRF error, try to get a fresh token and retry
        if e.code == 403 and "_xsrf" in error_body:
            print("XSRF token issue detected, trying to refresh token...")
            xsrf_token = None  # Reset token
            opener = get_xsrf_token()  # Get fresh token

            if xsrf_token:
                print("Retrying with fresh XSRF token...")
                req = urllib.request.Request(f"{server_url}/prompt", data=data)
                req.add_header("Content-Type", "application/json")
                req.add_header("User-Agent", "Python-FastAPI-Client/1.0")
                req.add_header("Accept", "application/json")
                req.add_header("X-XSRFToken", xsrf_token)

                try:
                    response = opener.open(req, timeout=60)
                    result = json.loads(response.read())
                    print(f"Retry successful. Response: {result}")
                    return result
                except Exception as retry_error:
                    print(f"Retry failed: {retry_error}")

        raise
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error queuing prompt: {e}")
        raise


def get_image(filename, subfolder, folder_type):
    """Get image with better error handling and XSRF support"""
    try:
        opener, _ = create_request_with_headers()

        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)

        req = urllib.request.Request(f"{server_url}/view?{url_values}")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")

        # Add XSRF token for GET requests too if needed
        if xsrf_token:
            req.add_header("X-XSRFToken", xsrf_token)

        response = opener.open(req, timeout=60)
        return response.read()

    except Exception as e:
        print(f"Error getting image: {e}")
        raise


def get_history(prompt_id):
    """Get history with better error handling and XSRF support"""
    try:
        opener, _ = create_request_with_headers()

        req = urllib.request.Request(f"{server_url}/history/{prompt_id}")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")
        req.add_header("Accept", "application/json")

        # Add XSRF token if available
        if xsrf_token:
            req.add_header("X-XSRFToken", xsrf_token)

        response = opener.open(req, timeout=30)
        return json.loads(response.read())

    except Exception as e:
        print(f"Error getting history: {e}")
        raise


def get_images(ws, prompt):
    """WebSocket image retrieval with better error handling"""
    try:
        prompt_response = queue_prompt(prompt)
        prompt_id = prompt_response["prompt_id"]
        print(f"Monitoring execution for prompt ID: {prompt_id}")
    except Exception as e:
        print(f"Failed to queue prompt: {e}")
        return None

    output_image = None
    current_node = ""
    timeout_count = 0
    max_timeout = 300  # 5 minutes

    while timeout_count < max_timeout:
        try:
            # Set a timeout for receiving messages
            ws.settimeout(1.0)
            out = ws.recv()
            timeout_count = 0  # Reset timeout counter on successful receive

            if isinstance(out, str):
                message = json.loads(out)
                print(f"WebSocket message: {message.get('type', 'unknown')}")

                if message["type"] == "executing":
                    data = message["data"]
                    if data["prompt_id"] == prompt_id:
                        if data["node"] is None:
                            print("Execution completed")
                            break  # Execution is done
                        else:
                            node_number = data["node"]
                            current_node = prompt.get(node_number, {}).get(
                                "class_type", ""
                            )
                            print(f"Executing node {node_number}: {current_node}")

                elif message["type"] == "execution_error":
                    print(f"Execution error: {message}")
                    return None
                elif message["type"] == "execution_start":
                    print(
                        f"Execution started for prompt: {message['data']['prompt_id']}"
                    )
            else:
                # Binary data received
                print(
                    f"Received binary data: {len(out)} bytes from node: {current_node}"
                )
                if current_node == save_image_websocket:
                    # Skip the first 8 bytes (websocket frame header)
                    output_image = out[8:]
                    print(f"Captured output image: {len(output_image)} bytes")

        except websocket.WebSocketTimeoutException:
            timeout_count += 1
            if timeout_count % 30 == 0:  # Print every 30 seconds
                print(f"Waiting for WebSocket message... ({timeout_count}s)")
            continue
        except Exception as e:
            print(f"WebSocket receive error: {e}")
            break

    if timeout_count >= max_timeout:
        print("WebSocket timeout reached")

    return output_image


def fetch_image_from_comfy(person_base64, outfit_base64, garment_type="top"):
    """Main function with garment type support and better error handling"""
    print(
        f"Starting ComfyUI request with base64 lengths: person={len(person_base64)}, outfit={len(outfit_base64)}, garment_type={garment_type}"
    )

    try:
        # Initialize XSRF token first
        get_xsrf_token()

        # Create SSL context for WebSocket
        _, ssl_context = create_request_with_headers()

        # Use wss:// for secure websocket connection
        ws_url = f"wss://{server_address}/ws?clientId={client_id}"
        print(f"Attempting WebSocket connection to: {ws_url}")

        # Create WebSocket with SSL context
        ws = websocket.create_connection(
            ws_url,
            timeout=60,
            sslopt={"cert_reqs": ssl.CERT_NONE, "check_hostname": False},
        )
        print("WebSocket connection established")

        prompt = get_prompt_with_workflow(person_base64, outfit_base64, garment_type)
        print(f"Generated prompt with {len(prompt)} nodes for {garment_type} garment")

        images = get_images(ws, prompt)
        ws.close()
        print(f"WebSocket method result: {'Success' if images else 'Failed'}")
        return images

    except Exception as e:
        print(f"WebSocket connection error: {e}")
        print("Falling back to polling method...")
        # Fallback method without websocket
        return fetch_image_polling(person_base64, outfit_base64, garment_type)


def image_to_base64(image_bytes):
    """Convert image bytes to base64 string without data URL prefix"""
    return base64.b64encode(image_bytes).decode("utf-8")


def fetch_image_polling(person_base64, outfit_base64, garment_type="top"):
    """Fallback method using polling instead of websocket with garment type support"""
    print(f"Using polling method as fallback for {garment_type} garment")
    try:
        # Initialize XSRF token first
        get_xsrf_token()

        prompt = get_prompt_with_workflow(person_base64, outfit_base64, garment_type)
        prompt_response = queue_prompt(prompt)
        prompt_id = prompt_response["prompt_id"]

        print(f"Queued prompt with ID: {prompt_id}")

        # Poll for completion with timeout
        max_polls = 300  # 5 minutes timeout
        poll_count = 0

        while poll_count < max_polls:
            try:
                history = get_history(prompt_id)
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})

                    if status.get("completed", False):
                        print("Execution completed, looking for output images...")
                        outputs = history[prompt_id].get("outputs", {})
                        print(f"Available output nodes: {list(outputs.keys())}")

                        # Look for images in any output node
                        for node_id in outputs:
                            node_outputs = outputs[node_id]
                            print(
                                f"Node {node_id} outputs: {list(node_outputs.keys())}"
                            )

                            if "images" in node_outputs and node_outputs["images"]:
                                # Get the first image
                                image_info = node_outputs["images"][0]
                                filename = image_info["filename"]
                                subfolder = image_info.get("subfolder", "")
                                folder_type = image_info.get("type", "output")

                                print(
                                    f"Downloading image: {filename} from {folder_type}/{subfolder}"
                                )
                                # Download the image
                                image_data = get_image(filename, subfolder, folder_type)
                                print(f"Downloaded image: {len(image_data)} bytes")
                                return image_data

                        print("No images found in outputs")
                        return None

                    elif "status_str" in status and status["status_str"] == "error":
                        print(f"ComfyUI execution error: {status}")
                        return None
                    else:
                        status_str = status.get("status_str", "unknown")
                        print(
                            f"Execution in progress... Status: {status_str} (poll {poll_count}/{max_polls})"
                        )
                else:
                    print(f"Prompt {prompt_id} not found in history yet")

                time.sleep(2)  # Wait 2 seconds before checking again
                poll_count += 1

            except Exception as e:
                print(f"Error checking history: {e}")
                time.sleep(2)
                poll_count += 1

        print("Timeout waiting for completion")
        return None

    except Exception as e:
        print(f"Polling method error: {e}")
        return None


def test_connection():
    """Test connection to ComfyUI server with better error handling"""
    try:
        opener, _ = create_request_with_headers()

        req = urllib.request.Request(f"{server_url}/")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")

        response = opener.open(req, timeout=15)
        status_code = response.getcode()
        print(f"Connection test: HTTP {status_code}")
        return status_code == 200

    except urllib.error.HTTPError as e:
        print(f"HTTP Error during connection test: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"URL Error during connection test: {e.reason}")
        return False
    except Exception as e:
        print(f"Connection test failed with unexpected error: {e}")
        return False


# Rest of the code remains the same...
GARMENTS_DIR = "garments"  # Directory containing pre-stored garments
TOP_GARMENTS_DIR = os.path.join(GARMENTS_DIR, "tops")
BOTTOM_GARMENTS_DIR = os.path.join(GARMENTS_DIR, "bottoms")


def ensure_garments_directories():
    """Ensure garment directories exist"""
    os.makedirs(TOP_GARMENTS_DIR, exist_ok=True)
    os.makedirs(BOTTOM_GARMENTS_DIR, exist_ok=True)


def image_file_to_base64(file_path: str) -> str:
    """Convert image file to base64 string"""
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def get_available_garments() -> Dict[str, List[Dict[str, str]]]:
    """Get list of available garments from directories"""
    ensure_garments_directories()

    garments = {"tops": [], "bottoms": []}

    # Get top garments
    if os.path.exists(TOP_GARMENTS_DIR):
        for filename in os.listdir(TOP_GARMENTS_DIR):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                garments["tops"].append(
                    {
                        "filename": filename,
                        "name": os.path.splitext(filename)[0].replace("_", " ").title(),
                        "path": os.path.join(TOP_GARMENTS_DIR, filename),
                    }
                )

    # Get bottom garments
    if os.path.exists(BOTTOM_GARMENTS_DIR):
        for filename in os.listdir(BOTTOM_GARMENTS_DIR):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                garments["bottoms"].append(
                    {
                        "filename": filename,
                        "name": os.path.splitext(filename)[0].replace("_", " ").title(),
                        "path": os.path.join(BOTTOM_GARMENTS_DIR, filename),
                    }
                )

    return garments


async def process_single_garment(
    person_base64: str, garment_info: Dict[str, str], garment_type: str
) -> Dict[str, Any]:
    """Process a single garment try-on"""
    try:
        garment_base64 = image_file_to_base64(garment_info["path"])

        # Use your existing fetch_image_from_comfy function
        result_image = fetch_image_from_comfy(
            person_base64, garment_base64, garment_type
        )

        if result_image:
            # Convert result to base64 for easy transport
            result_base64 = base64.b64encode(result_image).decode("utf-8")
            return {
                "success": True,
                "garment_name": garment_info["name"],
                "garment_filename": garment_info["filename"],
                "garment_type": garment_type,
                "result_image": result_base64,
            }
        else:
            return {
                "success": False,
                "garment_name": garment_info["name"],
                "garment_filename": garment_info["filename"],
                "garment_type": garment_type,
                "error": "Failed to generate try-on result",
            }
    except Exception as e:
        return {
            "success": False,
            "garment_name": garment_info["name"],
            "garment_filename": garment_info["filename"],
            "garment_type": garment_type,
            "error": str(e),
        }


async def batch_try_on_with_stored_garments(person_base64: str) -> Dict[str, Any]:
    """Process all available garments with the person image"""
    garments = get_available_garments()
    all_results = []

    # Prepare all garment processing tasks
    tasks = []

    # Add top garments (limit to 3)
    for garment_info in garments["tops"][:3]:
        tasks.append(process_single_garment(person_base64, garment_info, "top"))

    # Add bottom garments (limit to 2)
    for garment_info in garments["bottoms"][:2]:
        tasks.append(process_single_garment(person_base64, garment_info, "bottom"))

    if not tasks:
        return {
            "success": False,
            "message": "No garments found in project directory",
            "results": [],
        }

    # Process all garments concurrently with a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent ComfyUI requests

    async def process_with_semaphore(task):
        async with semaphore:
            return await task

    # Execute all tasks
    try:
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks]
        )

        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        return {
            "success": True,
            "total_processed": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "results": results,
            "message": f"Processed {len(results)} garments: {len(successful_results)} successful, {len(failed_results)} failed",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Batch processing failed: {str(e)}",
            "results": [],
        }


def batch_try_on_sync(person_base64: str) -> Dict[str, Any]:
    """Synchronous wrapper for batch try-on"""
    return asyncio.run(batch_try_on_with_stored_garments(person_base64))


# Alternative sequential processing (if async causes issues)
def batch_try_on_sequential(person_base64: str) -> Dict[str, Any]:
    """Process garments sequentially (safer but slower)"""
    garments = get_available_garments()
    all_results = []

    # Process top garments (limit to 3)
    for garment_info in garments["tops"][:3]:
        try:
            garment_base64 = image_file_to_base64(garment_info["path"])
            result_image = fetch_image_from_comfy(person_base64, garment_base64, "top")

            if result_image:
                result_base64 = base64.b64encode(result_image).decode("utf-8")
                all_results.append(
                    {
                        "success": True,
                        "garment_name": garment_info["name"],
                        "garment_filename": garment_info["filename"],
                        "garment_type": "top",
                        "result_image": result_base64,
                    }
                )
            else:
                all_results.append(
                    {
                        "success": False,
                        "garment_name": garment_info["name"],
                        "garment_filename": garment_info["filename"],
                        "garment_type": "top",
                        "error": "Failed to generate try-on result",
                    }
                )
        except Exception as e:
            all_results.append(
                {
                    "success": False,
                    "garment_name": garment_info["name"],
                    "garment_filename": garment_info["filename"],
                    "garment_type": "top",
                    "error": str(e),
                }
            )

    # Process bottom garments (limit to 2)
    for garment_info in garments["bottoms"][:2]:
        try:
            garment_base64 = image_file_to_base64(garment_info["path"])
            result_image = fetch_image_from_comfy(
                person_base64, garment_base64, "bottom"
            )

            if result_image:
                result_base64 = base64.b64encode(result_image).decode("utf-8")
                all_results.append(
                    {
                        "success": True,
                        "garment_name": garment_info["name"],
                        "garment_filename": garment_info["filename"],
                        "garment_type": "bottom",
                        "result_image": result_base64,
                    }
                )
            else:
                all_results.append(
                    {
                        "success": False,
                        "garment_name": garment_info["name"],
                        "garment_filename": garment_info["filename"],
                        "garment_type": "bottom",
                        "error": "Failed to generate try-on result",
                    }
                )
        except Exception as e:
            all_results.append(
                {
                    "success": False,
                    "garment_name": garment_info["name"],
                    "garment_filename": garment_info["filename"],
                    "garment_type": "bottom",
                    "error": str(e),
                }
            )

    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]

    return {
        "success": True,
        "total_processed": len(all_results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "results": all_results,
        "message": f"Processed {len(all_results)} garments: {len(successful_results)} successful, {len(failed_results)} failed",
    }


# Test function to validate ComfyUI endpoints
def test_comfyui_endpoints():
    """Test various ComfyUI endpoints"""
    endpoints = [
        f"{server_url}/",
        f"{server_url}/queue",
        f"{server_url}/history",
        f"{server_url}/system_stats",
    ]

    results = {}
    opener, _ = create_request_with_headers()

    for endpoint in endpoints:
        try:
            req = urllib.request.Request(endpoint)
            req.add_header("User-Agent", "Python-FastAPI-Client/1.0")

            # Add XSRF token if available
            if xsrf_token:
                req.add_header("X-XSRFToken", xsrf_token)

            response = opener.open(req, timeout=10)
            results[endpoint] = {"status": response.getcode(), "success": True}

        except Exception as e:
            results[endpoint] = {"status": "error", "error": str(e), "success": False}

    return results
