# src/webui/components/browser_use_agent_tab.py

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import gradio as gr

from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.views import BrowserState
from gradio.components import Component
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils import llm_provider
from src.webui.webui_manager import WebuiManager

logger = logging.getLogger(__name__)

# --- NEW HYBRID SYSTEM PROMPT WITH CONDITIONAL LOGIC ---
HYBRID_SYSTEM_PROMPT = """
#### 1. Core Identity & Directive
You are an advanced AI agent for hybrid-scene automation. Your core mission is to understand a user's complex objective and execute a multi-step plan across two scenes: **`web` (web browser)** and **`desktop` (desktop environment)**.

Your decisions must always be based on the **`current_scene`**. You will interact with the environment by generating structured JSON instructions in a systematic, efficient, and goal-oriented manner.

#### 2. Core Concept: The Dual-Scene Model
Your operational environment is strictly divided into two independent scenes. You must understand and adhere to the unique rules of each.

*   **`web` Scene**:
    *   **Environment**: Inside a web browser.
    *   **Perception**: You will receive a list of DOM elements, each tagged with a numeric `[index]`.
    *   **Interaction Method**: You **must** use the element's `index` to perform actions.
    *   **Rule**: `Only elements with numeric indexes in [] are interactive.`

*   **`desktop` Scene**:
    *   **Environment**: On the operating system's desktop.
    *   **Perception**: You will receive a list of UI elements, each containing a `bbox` and a `content` or `text` description.
    *   **Interaction Method**: You **must** use the element's **text description (`element_description`)** to perform actions.
    *   **Key Rule**: **Desktop elements do not have and do not use numeric indexes (`[index]`). Any attempt to use an index-based action in the `desktop` scene is invalid.**

#### 3. Workflow & Thinking Framework
You must strictly follow the **Perceive -> Reason -> Act** loop. The `current_scene` is explicitly provided in your context for every step.

1.  **Perceive**: Analyze the state information for the `current_scene`.
2.  **Reason**: Articulate your thought process and strategy within `<THINKING>` tags.
3.  **Act**: Output a single JSON object containing one instruction.

#### 4. Scene-Specific Toolsets & Instructions

##### A. If `current_scene == 'web'`
*   **Click element**: `{"click_element_by_index": {"index": 123}}`
*   **Type in element**: `{"type_text_in_element_by_index": {"index": 45, "text": "text to type"}}`
*   **Scroll**: `{"scroll": {"direction": "down"}}`
*   **Finish task**: `{"finish": {"result": "Successfully completed all steps."}}`

##### B. If `current_scene == 'desktop'`
*   **Click element**: `{"click": {"element_description": "Google Chrome icon"}}`
*   **Type text**: `{"type_text": {"text": "hello world", "element_description": "Search bar"}}` or `{"type_text": {"text": "hello world"}}`
*   **Press hotkeys**: `{"press_key": {"keys": ["ctrl", "c"]}}`
*   **Finish task**: `{"finish": {"result": "Successfully completed all steps."}}`

#### 4.1. Desktop Scene: Advanced Problem-Solving & Heuristics
When you cannot directly see the target application or element in the `desktop` scene, do not get stuck. Follow these strategies:

**Primary Strategy: Use the Operating System's Built-in Search Function.**

**Execution Steps for Windows:**
1.  **Locate the Start/Search Icon**: In the `ui_elements` list, look for the icon representing the Windows Start Menu or Search.
    *   **Key Identification Signal**: The element is often identified with `type: 'icon'` and a `content` field containing keywords like **`"Windows"`**, **`"Start"`**, or **`"Search"`**.
2.  **Step 1: Click the Icon**: Once identified, generate a `click` instruction to press it. Example: `{"click": {"element_description": "Windows"}}`
3.  **Step 2: Type the Search Term**: After the click, in the next perception cycle, generate a `type_text` instruction to enter the application's name. Example: `{"type_text": {"text": "calculator"}}`
4.  **Step 3: Launch the Application**: Find the target application in the search results and `click` it.

**Alternative Strategy: Use Keyboard Shortcuts**
*   Use the `press_key` tool to simulate pressing the `Win` key to open search. Example: `{"press_key": {"keys": ["win"]}}`

#### 5. Summary & Constraints
*   **The Scene is Your Primary Context**: All your decisions must be based on the `current_scene`.
*   **No Tool Mixing**: Strictly forbid using `_by_index` tools in the `desktop` scene and description-based tools in the `web` scene.
*   **Clear Thinking**: In your `<THINKING>` block, you **MUST** explicitly state your strategy. If the target is not found, you should state: "The target 'Calculator' is not on the current screen. I will execute the search strategy. I have found the 'Windows' icon and will click it to initiate search."
*   **Single Instruction Output**: Output only one JSON action instruction at a time.
*   **CRITICAL RULE: NO EMPTY ACTIONS**: If you determine an action is needed but cannot find a suitable element, **DO NOT** output an empty object (`{}`). **You must immediately apply the "Problem-Solving Strategy"**. Only if you cannot even find an element to execute the strategy (like the Windows icon) should you output a `finish` action with an error message.

#### 6. Example Thought Process

**Scenario: Desktop Search**
```xml
<THINKING>
User Goal: "Open the calculator"
Current state:
current_scene: 'desktop'
ui_elements: [
  {"bbox": [...], "content": "Recycle Bin"},
  {"bbox": [0.37, 0.96, 0.38, 0.99], "type": "icon", "content": "Windows"},
  {"bbox": [...], "content": "Google Chrome"}
]

1.  **Identify Scene**: I am in the `desktop` scene.
2.  **Assess State**: My goal is to open the calculator, but it is not visible in the `ui_elements` list.
3.  **Apply Strategy**: Since I cannot achieve the goal directly, I must apply the "Desktop Search Strategy". I have found an element with `type: 'icon'` and `content: 'Windows'`, which is the Start icon. I will click it to open the search function.
4.  **Formulate Instruction**: I will build a `click` instruction using the `element_description` "Windows".
</THINKING>
```
**Output:** `{"click": {"element_description": "Windows"}}`
"""

# --- Helper Functions --- (Defined at module level)


async def _initialize_llm(
        provider: Optional[str],
        model_name: Optional[str],
        temperature: float,
        base_url: Optional[str],
        api_key: Optional[str],
        num_ctx: Optional[int] = None,
) -> Optional[BaseChatModel]:
    """Initializes the LLM based on settings. Returns None if provider/model is missing."""
    if not provider or not model_name:
        logger.info("LLM Provider or Model Name not specified, LLM will be None.")
        return None
    try:
        logger.info(
            f"Initializing LLM: Provider={provider}, Model={model_name}, Temp={temperature}"
        )
        llm = llm_provider.get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url or None,
            api_key=api_key or None,
            num_ctx=num_ctx if provider == "ollama" else None,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        gr.Warning(
            f"Failed to initialize LLM '{model_name}' for provider '{provider}'. Please check settings. Error: {e}"
        )
        return None


def _get_config_value(
        webui_manager: WebuiManager,
        comp_dict: Dict[gr.components.Component, Any],
        comp_id_suffix: str,
        default: Any = None,
) -> Any:
    """Safely get value from component dictionary using its ID suffix relative to the tab."""
    tab_name = "browser_use_agent"
    comp_id = f"{tab_name}.{comp_id_suffix}"
    try:
        comp = webui_manager.get_component_by_id(comp_id)
        return comp_dict.get(comp, default)
    except KeyError:
        for prefix in ["agent_settings", "browser_settings"]:
            try:
                comp_id = f"{prefix}.{comp_id_suffix}"
                comp = webui_manager.get_component_by_id(comp_id)
                return comp_dict.get(comp, default)
            except KeyError:
                continue
        logger.warning(
            f"Component with suffix '{comp_id_suffix}' not found in manager for value lookup."
        )
        return default


def _format_agent_output(model_output: AgentOutput) -> str:
    """Formats AgentOutput for display in the chatbot using JSON."""
    content = ""
    if model_output:
        try:
            action_dump = [
                action.model_dump(exclude_none=True) for action in model_output.action
            ]

            state_dump = model_output.current_state.model_dump(exclude_none=True)
            model_output_dump = {
                "current_state": state_dump,
                "action": action_dump,
            }
            json_string = json.dumps(model_output_dump, indent=4, ensure_ascii=False)
            content = f"<pre><code class='language-json'>{json_string}</code></pre>"

        except AttributeError as ae:
            logger.error(
                f"AttributeError during model dump: {ae}. Check if 'action' or 'current_state' or their items support 'model_dump'."
            )
            content = f"<pre><code>Error: Could not format agent output (AttributeError: {ae}).\nRaw output: {str(model_output)}</code></pre>"
        except Exception as e:
            logger.error(f"Error formatting agent output: {e}", exc_info=True)
            content = f"<pre><code>Error formatting agent output.\nRaw output:\n{str(model_output)}</code></pre>"

    return content.strip()


# --- Updated Callback Implementation ---


async def _handle_new_step(
        webui_manager: WebuiManager, state: BrowserState, output: AgentOutput, step_num: int
):
    """Callback for each step taken by the agent, including screenshot display."""

    if not hasattr(webui_manager, "bu_chat_history"):
        logger.error(
            "Attribute 'bu_chat_history' not found in webui_manager! Cannot add chat message."
        )
        webui_manager.bu_chat_history = []
    step_num -= 1
    logger.info(f"Step {step_num} completed.")

    screenshot_html = ""
    screenshot_data = getattr(state, "screenshot", None)
    if screenshot_data:
        try:
            if (
                    isinstance(screenshot_data, str) and len(screenshot_data) > 100
            ):
                img_tag = f'<img src="data:image/jpeg;base64,{screenshot_data}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" />'
                screenshot_html = (
                        img_tag + "<br/>"
                )
            else:
                logger.warning(
                    f"Screenshot for step {step_num} seems invalid (type: {type(screenshot_data)}, len: {len(screenshot_data) if isinstance(screenshot_data, str) else 'N/A'})."
                )
                screenshot_html = "**[Invalid screenshot data]**<br/>"

        except Exception as e:
            logger.error(
                f"Error processing or formatting screenshot for step {step_num}: {e}",
                exc_info=True,
            )
            screenshot_html = "**[Error displaying screenshot]**<br/>"
    else:
        logger.debug(f"No screenshot available for step {step_num}.")

    formatted_output = _format_agent_output(output)
    step_header = f"--- **Step {step_num}** ---"
    final_content = step_header + "<br/>" + screenshot_html + formatted_output

    chat_message = {
        "role": "assistant",
        "content": final_content.strip(),
    }

    webui_manager.bu_chat_history.append(chat_message)
    await asyncio.sleep(0.05)


def _handle_done(webui_manager: WebuiManager, history: AgentHistoryList):
    """Callback when the agent finishes the task (success or failure)."""
    logger.info(
        f"Agent task finished. Duration: {history.total_duration_seconds():.2f}s, Tokens: {history.total_input_tokens()}"
    )
    final_summary = "**Task Completed**\n"
    final_summary += f"- Duration: {history.total_duration_seconds():.2f} seconds\n"
    final_summary += f"- Total Input Tokens: {history.total_input_tokens()}\n"

    final_result = history.final_result()
    if final_result:
        final_summary += f"- Final Result: {final_result}\n"

    errors = history.errors()
    if errors and any(errors):
        final_summary += f"- **Errors:**\n```\n{errors}\n```\n"
    else:
        final_summary += "- Status: Success\n"

    webui_manager.bu_chat_history.append(
        {"role": "assistant", "content": final_summary}
    )


async def _ask_assistant_callback(
        webui_manager: WebuiManager, query: str, browser_context: BrowserContext
) -> Dict[str, Any]:
    """Callback triggered by the agent's ask_for_assistant action."""
    logger.info("Agent requires assistance. Waiting for user input.")

    if not hasattr(webui_manager, "_chat_history"):
        logger.error("Chat history not found in webui_manager during ask_assistant!")
        return {"response": "Internal Error: Cannot display help request."}

    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"**Need Help:** {query}\nPlease provide information or perform the required action in the browser, then type your response/confirmation below and click 'Submit Response'.",
        }
    )

    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None

    try:
        logger.info("Waiting for user response event...")
        await asyncio.wait_for(
            webui_manager.bu_response_event.wait(), timeout=3600.0
        )
        logger.info("User response event received.")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for user assistance.")
        webui_manager.bu_chat_history.append(
            {
                "role": "assistant",
                "content": "**Timeout:** No response received. Trying to proceed.",
            }
        )
        webui_manager.bu_response_event = None
        return {"response": "Timeout: User did not respond."}

    response = webui_manager.bu_user_help_response
    webui_manager.bu_chat_history.append(
        {"role": "user", "content": response}
    )
    webui_manager.bu_response_event = (
        None
    )
    return {"response": response}


# --- Core Agent Execution Logic ---


async def run_agent_task(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
) -> AsyncGenerator[Dict[gr.components.Component, Any], None]:
    """Handles the entire lifecycle of initializing and running the agent."""

    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    run_button_comp = webui_manager.get_component_by_id("browser_use_agent.run_button")
    stop_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.stop_button"
    )
    pause_resume_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.pause_resume_button"
    )
    clear_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.clear_button"
    )
    chatbot_comp = webui_manager.get_component_by_id("browser_use_agent.chatbot")
    history_file_comp = webui_manager.get_component_by_id(
        "browser_use_agent.agent_history_file"
    )
    gif_comp = webui_manager.get_component_by_id("browser_use_agent.recording_gif")
    browser_view_comp = webui_manager.get_component_by_id(
        "browser_use_agent.browser_view"
    )

    task = components.get(user_input_comp, "").strip()
    if not task:
        gr.Warning("Please enter a task.")
        yield {run_button_comp: gr.update(interactive=True)}
        return

    webui_manager.bu_chat_history.append({"role": "user", "content": task})

    yield {
        user_input_comp: gr.Textbox(
            value="", interactive=False, placeholder="Agent is running..."
        ),
        run_button_comp: gr.Button(value="⏳ Running...", interactive=False),
        stop_button_comp: gr.Button(interactive=True),
        pause_resume_button_comp: gr.Button(value="⏸️ Pause", interactive=True),
        clear_button_comp: gr.Button(interactive=False),
        chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
        history_file_comp: gr.update(value=None),
        gif_comp: gr.update(value=None),
    }

    def get_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"agent_settings.{key}")
        return components.get(comp, default) if comp else default

    # MODIFIED: We no longer use extend_system_prompt. We use override exclusively.
    # extend_system_prompt = get_setting("extend_system_prompt") or None
    llm_provider_name = get_setting(
        "llm_provider", None
    )
    llm_model_name = get_setting("llm_model_name", None)
    llm_temperature = get_setting("llm_temperature", 0.6)
    use_vision = get_setting("use_vision", True)
    ollama_num_ctx = get_setting("ollama_num_ctx", 16000)
    llm_base_url = get_setting("llm_base_url") or None
    llm_api_key = get_setting("llm_api_key") or None
    max_steps = get_setting("max_steps", 100)
    max_actions = get_setting("max_actions", 10)
    max_input_tokens = get_setting("max_input_tokens", 128000)
    tool_calling_str = get_setting("tool_calling_method", "auto")
    tool_calling_method = tool_calling_str if tool_calling_str != "None" else None
    mcp_server_config_comp = webui_manager.id_to_component.get(
        "agent_settings.mcp_server_config"
    )
    mcp_server_config_str = (
        components.get(mcp_server_config_comp) if mcp_server_config_comp else None
    )
    mcp_server_config = (
        json.loads(mcp_server_config_str) if mcp_server_config_str else None
    )

    planner_llm_provider_name = get_setting("planner_llm_provider") or None
    planner_llm = None
    planner_use_vision = False
    if planner_llm_provider_name:
        planner_llm_model_name = get_setting("planner_llm_model_name")
        planner_llm_temperature = get_setting("planner_llm_temperature", 0.6)
        planner_ollama_num_ctx = get_setting("planner_ollama_num_ctx", 16000)
        planner_llm_base_url = get_setting("planner_llm_base_url") or None
        planner_llm_api_key = get_setting("planner_llm_api_key") or None
        planner_use_vision = get_setting("planner_use_vision", False)

        planner_llm = await _initialize_llm(
            planner_llm_provider_name,
            planner_llm_model_name,
            planner_llm_temperature,
            planner_llm_base_url,
            planner_llm_api_key,
            planner_ollama_num_ctx if planner_llm_provider_name == "ollama" else None,
        )

    def get_browser_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"browser_settings.{key}")
        return components.get(comp, default) if comp else default

    browser_binary_path = get_browser_setting("browser_binary_path") or None
    browser_user_data_dir = get_browser_setting("browser_user_data_dir") or None
    use_own_browser = get_browser_setting(
        "use_own_browser", False
    )
    keep_browser_open = get_browser_setting("keep_browser_open", False)
    headless = get_browser_setting("headless", False)
    disable_security = get_browser_setting("disable_security", False)
    window_w = int(get_browser_setting("window_w", 1280))
    window_h = int(get_browser_setting("window_h", 1100))
    cdp_url = get_browser_setting("cdp_url") or None
    wss_url = get_browser_setting("wss_url") or None
    save_recording_path = get_browser_setting("save_recording_path") or None
    save_trace_path = get_browser_setting("save_trace_path") or None
    save_agent_history_path = get_browser_setting(
        "save_agent_history_path", "./tmp/agent_history"
    )
    save_download_path = get_browser_setting("save_download_path", "./tmp/downloads")

    stream_vw = 70
    stream_vh = int(70 * window_h // window_w)

    os.makedirs(save_agent_history_path, exist_ok=True)
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
    if save_trace_path:
        os.makedirs(save_trace_path, exist_ok=True)
    if save_download_path:
        os.makedirs(save_download_path, exist_ok=True)

    main_llm = await _initialize_llm(
        llm_provider_name,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        ollama_num_ctx if llm_provider_name == "ollama" else None,
    )

    async def ask_callback_wrapper(
            query: str, browser_context: BrowserContext
    ) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    if not webui_manager.bu_controller:
        webui_manager.bu_controller = CustomController(
            ask_assistant_callback=ask_callback_wrapper
        )
        await webui_manager.bu_controller.setup_mcp_client(mcp_server_config)

    should_close_browser_on_finish = not keep_browser_open

    try:
        if not keep_browser_open:
            if webui_manager.bu_browser_context:
                logger.info("Closing previous browser context.")
                await webui_manager.bu_browser_context.close()
                webui_manager.bu_browser_context = None
            if webui_manager.bu_browser:
                logger.info("Closing previous browser.")
                await webui_manager.bu_browser.close()
                webui_manager.bu_browser = None

        if not webui_manager.bu_browser:
            logger.info("Launching new browser instance.")
            extra_args = []
            if use_own_browser:
                browser_binary_path = os.getenv("BROWSER_PATH", None) or browser_binary_path
                if browser_binary_path == "":
                    browser_binary_path = None
                browser_user_data = browser_user_data_dir or os.getenv("BROWSER_USER_DATA", None)
                if browser_user_data:
                    extra_args += [f"--user-data-dir={browser_user_data}"]
            else:
                browser_binary_path = None

            webui_manager.bu_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    browser_binary_path=browser_binary_path,
                    extra_browser_args=extra_args,
                    wss_url=wss_url,
                    cdp_url=cdp_url,
                    new_context_config=BrowserContextConfig(
                        window_width=window_w,
                        window_height=window_h,
                    )
                )
            )

        if not webui_manager.bu_browser_context:
            logger.info("Creating new browser context.")
            context_config = BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path
                if save_recording_path
                else None,
                save_downloads_path=save_download_path if save_download_path else None,
                window_height=window_h,
                window_width=window_w,
            )
            if not webui_manager.bu_browser:
                raise ValueError("Browser not initialized, cannot create context.")
            webui_manager.bu_browser_context = (
                await webui_manager.bu_browser.new_context(config=context_config)
            )

        webui_manager.bu_agent_task_id = str(uuid.uuid4())
        os.makedirs(
            os.path.join(save_agent_history_path, webui_manager.bu_agent_task_id),
            exist_ok=True,
        )
        history_file = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.json",
        )
        gif_path = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.gif",
        )

        async def step_callback_wrapper(
                state: BrowserState, output: AgentOutput, step_num: int
        ):
            await _handle_new_step(webui_manager, state, output, step_num)

        def done_callback_wrapper(history: AgentHistoryList):
            _handle_done(webui_manager, history)

        if not webui_manager.bu_agent:
            logger.info(f"Initializing new agent for task: {task}")
            if not webui_manager.bu_browser or not webui_manager.bu_browser_context:
                raise ValueError(
                    "Browser or Context not initialized, cannot create agent."
                )
            
            # MODIFIED: Use the new HYBRID_SYSTEM_PROMPT via override_system_message
            webui_manager.bu_agent = BrowserUseAgent(
                task=task,
                llm=main_llm,
                browser=webui_manager.bu_browser,
                browser_context=webui_manager.bu_browser_context,
                controller=webui_manager.bu_controller,
                register_new_step_callback=step_callback_wrapper,
                register_done_callback=done_callback_wrapper,
                use_vision=use_vision,
                override_system_message=HYBRID_SYSTEM_PROMPT, # Use the new prompt
                extend_system_message=None, # Ensure this is None
                max_input_tokens=max_input_tokens,
                max_actions_per_step=max_actions,
                tool_calling_method=tool_calling_method,
                planner_llm=planner_llm,
                use_vision_for_planner=planner_use_vision if planner_llm else False,
                source="webui",
            )
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.settings.generate_gif = gif_path
        else:
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.add_new_task(task)
            webui_manager.bu_agent.settings.generate_gif = gif_path
            webui_manager.bu_agent.browser = webui_manager.bu_browser
            webui_manager.bu_agent.browser_context = webui_manager.bu_browser_context
            webui_manager.bu_agent.controller = webui_manager.bu_controller

        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task

        last_chat_len = len(webui_manager.bu_chat_history)
        while not agent_task.done():
            is_paused = webui_manager.bu_agent.state.paused
            is_stopped = webui_manager.bu_agent.state.stopped

            if is_paused:
                yield {
                    pause_resume_button_comp: gr.update(
                        value="▶️ Resume", interactive=True
                    ),
                    stop_button_comp: gr.update(interactive=True),
                }
                while is_paused and not agent_task.done():
                    is_paused = webui_manager.bu_agent.state.paused
                    is_stopped = webui_manager.bu_agent.state.stopped
                    if is_stopped:
                        break
                    await asyncio.sleep(0.2)

                if (
                        agent_task.done() or is_stopped
                ):
                    break

                yield {
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=True
                    ),
                    run_button_comp: gr.update(
                        value="⏳ Running...", interactive=False
                    ),
                }

            if is_stopped:
                logger.info("Agent has stopped (internally or via stop button).")
                if not agent_task.done():
                    try:
                        await asyncio.wait_for(
                            agent_task, timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Agent task did not finish quickly after stop signal, cancelling."
                        )
                        agent_task.cancel()
                    except Exception:
                        pass
                break

            update_dict = {}
            if webui_manager.bu_response_event is not None:
                update_dict = {
                    user_input_comp: gr.update(
                        placeholder="Agent needs help. Enter response and submit.",
                        interactive=True,
                    ),
                    run_button_comp: gr.update(
                        value="✔️ Submit Response", interactive=True
                    ),
                    pause_resume_button_comp: gr.update(interactive=False),
                    stop_button_comp: gr.update(interactive=False),
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
                last_chat_len = len(webui_manager.bu_chat_history)
                yield update_dict
                await webui_manager.bu_response_event.wait()

                if not agent_task.done():
                    yield {
                        user_input_comp: gr.update(
                            placeholder="Agent is running...", interactive=False
                        ),
                        run_button_comp: gr.update(
                            value="⏳ Running...", interactive=False
                        ),
                        pause_resume_button_comp: gr.update(interactive=True),
                        stop_button_comp: gr.update(interactive=True),
                    }
                else:
                    break

            if len(webui_manager.bu_chat_history) > last_chat_len:
                update_dict[chatbot_comp] = gr.update(
                    value=webui_manager.bu_chat_history
                )
                last_chat_len = len(webui_manager.bu_chat_history)

            if headless and webui_manager.bu_browser_context:
                try:
                    screenshot_b64 = (
                        await webui_manager.bu_browser_context.take_screenshot()
                    )
                    if screenshot_b64:
                        html_content = f'<img src="data:image/jpeg;base64,{screenshot_b64}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                except Exception as e:
                    logger.debug(f"Failed to capture screenshot: {e}")
                    update_dict[browser_view_comp] = gr.update(
                        value="<div style='...'>Error loading view...</div>",
                        visible=True,
                    )
            else:
                update_dict[browser_view_comp] = gr.update(visible=False)

            if update_dict:
                yield update_dict

            await asyncio.sleep(0.1)

        webui_manager.bu_agent.state.paused = False
        webui_manager.bu_agent.state.stopped = False
        final_update = {}
        try:
            logger.info("Agent task completing...")
            if not agent_task.done():
                await agent_task
            elif agent_task.exception():
                agent_task.result()
            logger.info("Agent task completed processing.")

            logger.info(f"Explicitly saving agent history to: {history_file}")
            webui_manager.bu_agent.save_history(history_file)

            if os.path.exists(history_file):
                final_update[history_file_comp] = gr.File(value=history_file)

            if gif_path and os.path.exists(gif_path):
                logger.info(f"GIF found at: {gif_path}")
                final_update[gif_comp] = gr.Image(value=gif_path)

        except asyncio.CancelledError:
            logger.info("Agent task was cancelled.")
            if not any(
                    "Cancelled" in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": "**Task Cancelled**."}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            error_message = (
                f"**Agent Execution Error:**\n```\n{type(e).__name__}: {e}\n```"
            )
            if not any(
                    error_message in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
            gr.Error(f"Agent execution failed: {e}")

        finally:
            webui_manager.bu_current_task = None

            if should_close_browser_on_finish:
                if webui_manager.bu_browser_context:
                    logger.info("Closing browser context after task.")
                    await webui_manager.bu_browser_context.close()
                    webui_manager.bu_browser_context = None
                if webui_manager.bu_browser:
                    logger.info("Closing browser after task.")
                    await webui_manager.bu_browser.close()
                    webui_manager.bu_browser = None

            final_update.update(
                {
                    user_input_comp: gr.update(
                        value="",
                        interactive=True,
                        placeholder="Enter your next task...",
                    ),
                    run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
                    stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=False
                    ),
                    clear_button_comp: gr.update(interactive=True),
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
            )
            yield final_update

    except Exception as e:
        logger.error(f"Error setting up agent task: {e}", exc_info=True)
        webui_manager.bu_current_task = None
        yield {
            user_input_comp: gr.update(
                interactive=True, placeholder="Error during setup. Enter task..."
            ),
            run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
            stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
            pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False),
            clear_button_comp: gr.update(interactive=True),
            chatbot_comp: gr.update(
                value=webui_manager.bu_chat_history
                      + [{"role": "assistant", "content": f"**Setup Error:** {e}"}]
            ),
        }


# --- Button Click Handlers ---


async def handle_submit(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
):
    """Handles clicks on the main 'Submit' button."""
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    user_input_value = components.get(user_input_comp, "").strip()

    if webui_manager.bu_response_event and not webui_manager.bu_response_event.is_set():
        logger.info(f"User submitted assistance: {user_input_value}")
        webui_manager.bu_user_help_response = (
            user_input_value if user_input_value else "User provided no text response."
        )
        webui_manager.bu_response_event.set()
        yield {
            user_input_comp: gr.update(
                value="",
                interactive=False,
                placeholder="Waiting for agent to continue...",
            ),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(value="⏳ Running...", interactive=False),
        }
    elif webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        logger.warning(
            "Submit button clicked while agent is already running and not asking for help."
        )
        gr.Info("Agent is currently running. Please wait or use Stop/Pause.")
        yield {}
    else:
        logger.info("Submit button clicked for new task.")
        async for update in run_agent_task(webui_manager, components):
            yield update


async def handle_stop(webui_manager: WebuiManager):
    """Handles clicks on the 'Stop' button."""
    logger.info("Stop button clicked.")
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        agent.state.stopped = True
        agent.state.paused = False
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False, value="⏹️ Stopping..."),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=False),
        }
    else:
        logger.warning("Stop clicked but agent is not running or task is already done.")
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=True),
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.clear_button"
            ): gr.update(interactive=True),
        }


async def handle_pause_resume(webui_manager: WebuiManager):
    """Handles clicks on the 'Pause/Resume' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        if agent.state.paused:
            logger.info("Resume button clicked.")
            agent.resume()
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="⏸️ Pause", interactive=True)
            }
        else:
            logger.info("Pause button clicked.")
            agent.pause()
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="▶️ Resume", interactive=True)
            }
    else:
        logger.warning(
            "Pause/Resume clicked but agent is not running or doesn't support state."
        )
        return {}


async def handle_clear(webui_manager: WebuiManager):
    """Handles clicks on the 'Clear' button."""
    logger.info("Clear button clicked.")

    task = webui_manager.bu_current_task
    if task and not task.done():
        logger.info("Clearing requires stopping the current task.")
        webui_manager.bu_agent.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"Error stopping task on clear: {e}")
    webui_manager.bu_current_task = None

    if webui_manager.bu_controller:
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None
    webui_manager.bu_agent = None

    webui_manager.bu_chat_history = []
    webui_manager.bu_response_event = None
    webui_manager.bu_user_help_response = None
    webui_manager.bu_agent_task_id = None

    logger.info("Agent state and browser resources cleared.")

    return {
        webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(
            value=[]
        ),
        webui_manager.get_component_by_id("browser_use_agent.user_input"): gr.update(
            value="", placeholder="Enter your task here..."
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.agent_history_file"
        ): gr.update(value=None),
        webui_manager.get_component_by_id("browser_use_agent.recording_gif"): gr.update(
            value=None
        ),
        webui_manager.get_component_by_id("browser_use_agent.browser_view"): gr.update(
            value="<div style='...'>Browser Cleared</div>"
        ),
        webui_manager.get_component_by_id("browser_use_agent.run_button"): gr.update(
            value="▶️ Submit Task", interactive=True
        ),
        webui_manager.get_component_by_id("browser_use_agent.stop_button"): gr.update(
            interactive=False
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.pause_resume_button"
        ): gr.update(value="⏸️ Pause", interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.clear_button"): gr.update(
            interactive=True
        ),
    }


# --- Tab Creation Function ---


def create_browser_use_agent_tab(webui_manager: WebuiManager):
    """
    Create the run agent tab, defining UI, state, and handlers.
    """
    webui_manager.init_browser_use_agent()

    tab_components = {}
    with gr.Column():
        chatbot = gr.Chatbot(
            lambda: webui_manager.bu_chat_history,
            elem_id="browser_use_chatbot",
            label="Agent Interaction",
            type="messages",
            height=600,
            show_copy_button=True,
        )
        user_input = gr.Textbox(
            label="Your Task or Response",
            placeholder="Enter your task here or provide assistance when asked.",
            lines=3,
            interactive=True,
            elem_id="user_input",
        )
        with gr.Row():
            stop_button = gr.Button(
                "⏹️ Stop", interactive=False, variant="stop", scale=2
            )
            pause_resume_button = gr.Button(
                "⏸️ Pause", interactive=False, variant="secondary", scale=2, visible=True
            )
            clear_button = gr.Button(
                "🗑️ Clear", interactive=True, variant="secondary", scale=2
            )
            run_button = gr.Button("▶️ Submit Task", variant="primary", scale=3)

        browser_view = gr.HTML(
            value="<div style='width:100%; height:50vh; display:flex; justify-content:center; align-items:center; border:1px solid #ccc; background-color:#f0f0f0;'><p>Browser View (Requires Headless=True)</p></div>",
            label="Browser Live View",
            elem_id="browser_view",
            visible=False,
        )
        with gr.Column():
            gr.Markdown("### Task Outputs")
            agent_history_file = gr.File(label="Agent History JSON", interactive=False)
            recording_gif = gr.Image(
                label="Task Recording GIF",
                format="gif",
                interactive=False,
                type="filepath",
            )

    tab_components.update(
        dict(
            chatbot=chatbot,
            user_input=user_input,
            clear_button=clear_button,
            run_button=run_button,
            stop_button=stop_button,
            pause_resume_button=pause_resume_button,
            agent_history_file=agent_history_file,
            recording_gif=recording_gif,
            browser_view=browser_view,
        )
    )
    webui_manager.add_components(
        "browser_use_agent", tab_components
    )

    all_managed_components = set(
        webui_manager.get_components()
    )
    run_tab_outputs = list(tab_components.values())

    async def submit_wrapper(
            components_dict: Dict[Component, Any],
    ) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_submit that yields its results."""
        async for update in handle_submit(webui_manager, components_dict):
            yield update

    async def stop_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_stop."""
        update_dict = await handle_stop(webui_manager)
        yield update_dict

    async def pause_resume_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_pause_resume."""
        update_dict = await handle_pause_resume(webui_manager)
        yield update_dict

    async def clear_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_clear."""
        update_dict = await handle_clear(webui_manager)
        yield update_dict

    run_button.click(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs, trigger_mode="multiple"
    )
    user_input.submit(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs
    )
    stop_button.click(fn=stop_wrapper, inputs=None, outputs=run_tab_outputs)
    pause_resume_button.click(
        fn=pause_resume_wrapper, inputs=None, outputs=run_tab_outputs
    )
    clear_button.click(fn=clear_wrapper, inputs=None, outputs=run_tab_outputs)