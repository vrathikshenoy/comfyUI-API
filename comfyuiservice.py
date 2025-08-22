import websocket
import uuid
import json
import urllib.request
import urllib.parse
import urllib.error
import base64
import time
import ssl

save_image_websocket = "SaveImageWebsocket"
server_address = "3j0cjfcppp6ir2-8188.proxy.runpod.net"
server_url = "https://3j0cjfcppp6ir2-8188.proxy.runpod.net"
client_id = str(uuid.uuid4())


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
      "lora_name": "pytorch_lora_weights.safetensors",
      "strength_model": 1.0,
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
        "bottom": [9, 12, 16],  # Lower body garments
    }

    # Update mask components based on garment type
    prompt_json["1"]["inputs"]["mask_components"]["__value__"] = mask_components.get(
        garment_type, [15, 14, 5]
    )

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


def create_request_with_headers():
    """Create a request with proper headers and SSL context"""
    # Create SSL context that doesn't verify certificates (for development)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Create custom opener with SSL context
    https_handler = urllib.request.HTTPSHandler(context=ssl_context)
    opener = urllib.request.build_opener(https_handler)

    return opener, ssl_context


def queue_prompt(prompt):
    """Queue prompt with better error handling and headers"""
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    print(f"Sending prompt to ComfyUI: {len(data)} bytes")

    try:
        opener, _ = create_request_with_headers()

        req = urllib.request.Request(f"{server_url}/prompt", data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")
        req.add_header("Accept", "application/json")

        response = opener.open(req, timeout=60)  # Increased timeout
        result = json.loads(response.read())
        print(f"Prompt queued successfully. Response: {result}")
        return result

    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response body: {e.read().decode('utf-8', errors='ignore')}")
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
    """Get image with better error handling"""
    try:
        opener, _ = create_request_with_headers()

        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)

        req = urllib.request.Request(f"{server_url}/view?{url_values}")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")

        response = opener.open(req, timeout=60)
        return response.read()

    except Exception as e:
        print(f"Error getting image: {e}")
        raise


def get_history(prompt_id):
    """Get history with better error handling"""
    try:
        opener, _ = create_request_with_headers()

        req = urllib.request.Request(f"{server_url}/history/{prompt_id}")
        req.add_header("User-Agent", "Python-FastAPI-Client/1.0")
        req.add_header("Accept", "application/json")

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

            response = opener.open(req, timeout=10)
            results[endpoint] = {"status": response.getcode(), "success": True}

        except Exception as e:
            results[endpoint] = {"status": "error", "error": str(e), "success": False}

    return results

