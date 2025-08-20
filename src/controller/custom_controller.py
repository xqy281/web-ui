# src/controller/custom_controller.py

import pdb
import pyperclip
import json
import base64
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar, Tuple
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from browser_use.controller.registry.service import Registry, RegisteredAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
import inspect
import asyncio
import os
import tempfile
import pyautogui
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.agent.views import ActionModel, ActionResult

from src.utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools
from src.utils.omni_parser_client import OmniParserClient
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.mcp_client = None
        self.mcp_server_config = None
        self._omni_parser_client: Optional[OmniParserClient] = None
        self.current_action_context: Optional[Dict[str, Any]] = None

    @property
    def omni_parser_client(self) -> OmniParserClient:
        """Lazy initializer for the OmniParserClient."""
        if self._omni_parser_client is None:
            self._omni_parser_client = OmniParserClient()
        return self._omni_parser_client

    async def get_desktop_elements(self) -> Tuple[list[Dict[str, Any]], str]:
        """
        Captures the screen, parses it for UI elements, and returns both the
        elements and the base64 encoded screenshot.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            screenshot_path = tmp_file.name
        
        try:
            pyautogui.screenshot(screenshot_path)
            logger.info(f"Desktop screenshot saved to temporary file: {screenshot_path}")
            
            parsed_elements = await self.omni_parser_client.parse_image(screenshot_path)
            
            with open(screenshot_path, "rb") as image_file:
                screenshot_b64 = base64.b64encode(image_file.read()).decode('utf-8')
                
            return parsed_elements, screenshot_b64
        finally:
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)

    def _register_custom_actions(self):
        """Register all custom browser and NEW desktop actions"""

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. However, if you encounter a definitive blocker "
            "that prevents you from proceeding independently – such as needing credentials you don't possess, "
            "requiring subjective human judgment, needing a physical action performed, encountering complex CAPTCHAs, "
            "or facing limitations in your capabilities – you must request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(extracted_content="Human cannot help you. Please try another way.",
                                    include_in_memory=True)

        @self.registry.action(
            'Upload file to interactive element with file path ',
        )
        async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')

            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')

            dom_el = await browser.get_dom_element_by_index(index)
            file_upload_dom_el = dom_el.get_file_upload_element()

            if file_upload_dom_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)

            if file_upload_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

        # --- NEW DESKTOP ACTIONS ---

        @self.registry.action(
            "Captures the entire desktop screen and returns a list of all visible UI elements, including their descriptions and locations. Use this to understand what is currently on the desktop."
        )
        async def get_desktop_ui_elements() -> ActionResult:
            elements, screenshot_b64 = await self.get_desktop_elements()
            formatted_elements = "\n".join([f"- {el['content']} ({el['type']})" for el in elements])
            result_text = f"Found {len(elements)} elements on the desktop:\n{formatted_elements}"
            
            if isinstance(self.current_action_context, dict):
                self.current_action_context['desktop_elements'] = elements
            
            return ActionResult(long_term_memory=result_text)

        @self.registry.action(
            "Clicks on a specific UI element on the desktop based on its text or AI-generated description. You should call `get_desktop_ui_elements` first to know what to click on."
        )
        async def click_on_desktop(element_description: str) -> ActionResult:
            if not isinstance(self.current_action_context, dict) or 'desktop_elements' not in self.current_action_context:
                 return ActionResult(error="Desktop elements not found in context. Please run `get_desktop_ui_elements` first.")

            elements = self.current_action_context['desktop_elements']
            
            # 修正: 使用 'in' 进行模糊匹配，并移除末尾的句号
            search_term = element_description.strip().lower().rstrip('.')
            target_element = next((el for el in elements if search_term in el['content'].strip().lower()), None)

            if not target_element:
                return ActionResult(error=f"Element with description containing '{search_term}' not found on the desktop.")

            bbox = target_element['bbox']
            screen_width, screen_height = pyautogui.size()
            
            x_center = (bbox[0] + bbox[2]) / 2 * screen_width
            y_center = (bbox[1] + bbox[3]) / 2 * screen_height
            
            pyautogui.click(x_center, y_center)
            msg = f"Successfully clicked on desktop element: '{element_description}'."
            logger.info(msg)
            return ActionResult(long_term_memory=msg)

        @self.registry.action(
            "Types the given text using the keyboard. If you need to type into a specific field, use `click_on_desktop` first to focus it."
        )
        async def type_on_desktop(text: str) -> ActionResult:
            pyautogui.write(text, interval=0.05)
            msg = f"Successfully typed text on desktop: '{text[:50]}...'"
            logger.info(msg)
            return ActionResult(long_term_memory=msg)

        @self.registry.action(
            "Presses a combination of keyboard keys simultaneously (a hotkey). The keys must be provided as a JSON formatted string of a list."
        )
        async def press_hotkey_on_desktop(keys_json: str) -> ActionResult:
            try:
                keys = json.loads(keys_json)
                if not isinstance(keys, list):
                    raise ValueError("JSON string must decode to a list.")
                
                pyautogui.hotkey(*keys)
                msg = f"Successfully pressed hotkey: {', '.join(keys)}"
                logger.info(msg)
                return ActionResult(long_term_memory=msg)
            except json.JSONDecodeError:
                error_msg = f"Invalid JSON format for keys: '{keys_json}'. Please provide a valid JSON list of strings."
                logger.error(error_msg)
                return ActionResult(error=error_msg)
            except Exception as e:
                error_msg = f"Error executing hotkey: {e}"
                logger.error(error_msg, exc_info=True)
                return ActionResult(error=error_msg)


    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: Optional[BrowserContext] = None,
            page_extraction_llm: Optional[BaseChatModel] = None,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""
        self.current_action_context = context

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    if action_name.startswith("mcp"):
                        logger.debug(f"Invoke MCP tool: {action_name}")
                        mcp_tool = self.registry.registry.actions.get(action_name).function
                        result = await mcp_tool.ainvoke(params)
                    else:
                        result = await self.registry.execute_action(
                            action_name,
                            params,
                            browser=browser_context,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            context=context,
                        )

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e
        finally:
            self.current_action_context = None

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        self.mcp_server_config = mcp_server_config
        if self.mcp_server_config:
            self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
            self.register_mcp_tools()

    def register_mcp_tools(self):
        """
        Register the MCP tools used by this controller.
        """
        if self.mcp_client:
            for server_name in self.mcp_client.server_name_to_tools:
                for tool in self.mcp_client.server_name_to_tools[server_name]:
                    tool_name = f"mcp.{server_name}.{tool.name}"
                    self.registry.registry.actions[tool_name] = RegisteredAction(
                        name=tool_name,
                        description=tool.description,
                        function=tool,
                        param_model=create_tool_param_model(tool),
                    )
                    logger.info(f"Add mcp tool: {tool_name}")
                logger.debug(
                    f"Registered {len(self.mcp_client.server_name_to_tools[server_name])} mcp tools for {server_name}")
        else:
            logger.warning(f"MCP client not started.")

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)