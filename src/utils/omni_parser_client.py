# src/utils/omni_parser_client.py

import httpx
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class OmniParserClient:
    """
    An asynchronous client to interact with the OmniParser API for parsing UI screenshots.
    """
    def __init__(self, api_url: str = None):
        """
        Initializes the client.

        Args:
            api_url: The URL of the OmniParser API. Defaults to env variable or a fallback.
        """
        self.api_url = api_url or os.getenv("OMNIPARSER_API_URL", "http://127.0.0.1:8000/parse_image/")
        if not self.api_url.endswith('/'):
            self.api_url += '/'
        logger.info(f"OmniParserClient initialized for API URL: {self.api_url}")

    async def parse_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Sends an image to the OmniParser API and returns the structured UI elements.

        Args:
            image_path: The local path to the screenshot image file.

        Returns:
            A list of dictionaries, where each dictionary represents a parsed UI element.
            Returns an empty list if the API call fails.
        """
        if not os.path.exists(image_path):
            logger.error(f"[OmniParserClient] Image file not found at: {image_path}")
            return []

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(image_path, "rb") as f:
                    files = {'image': (os.path.basename(image_path), f, 'image/png')}
                    # For performance, generate_annotated_image is always False
                    data = {'generate_annotated_image': 'False'}
                    
                    logger.debug(f"Sending image to OmniParser API: {self.api_url}")
                    response = await client.post(self.api_url, files=files, data=data)
                    
                    response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
                    
                    result = response.json()

                    if result.get("status") == "success":
                        elements = result.get("parsed_elements", [])
                        logger.info(f"Successfully parsed {len(elements)} UI elements from desktop screenshot.")
                        return elements
                    else:
                        error_message = result.get("message", "Unknown error from OmniParser API")
                        logger.error(f"OmniParser API returned an error: {error_message}")
                        return []

        except httpx.RequestError as e:
            logger.error(f"An error occurred while requesting OmniParser API: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred in OmniParserClient: {e}", exc_info=True)
            return []