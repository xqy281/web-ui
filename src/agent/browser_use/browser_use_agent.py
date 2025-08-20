# src/agent/browser_use/browser_use_agent.py

from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Literal, Optional, Dict, Any

from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc
from browser_use.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentStepInfo,
    ToolCallingMethod,
    AgentOutput,
    ActionModel,
)
from browser_use.browser.views import BrowserStateHistory
from browser_use.utils import time_execution_async, SignalHandler
from dotenv import load_dotenv
# 确保导入所有需要的辅助函数和异常类型
from browser_use.agent.message_manager.utils import is_model_without_tool_support, convert_input_messages, extract_json_from_model_output
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from pydantic import ValidationError

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
        os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)


class BrowserUseAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_scene: Optional[Literal['web', 'desktop']] = None
        self.context_data: Dict[str, Any] = {}
        self.last_desktop_screenshot: Optional[str] = None

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if is_model_without_tool_support(self.model_name):
                return 'raw'
            elif self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def _log_messages_for_llm(self, messages: list[BaseMessage]):
        log_str = "\n--- START: CONTEXT FOR LLM ---\n"
        for i, msg in enumerate(messages):
            log_str += f"--- Message {i+1}: Type: {type(msg).__name__} ---\n"
            if isinstance(msg.content, str):
                log_str += f"{msg.content}\n"
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        log_str += f"Text: {item.get('text')}\n"
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        log_str += "Image: [Base64 Data]\n"
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                try:
                    log_str += f"Tool Calls: {json.dumps(msg.tool_calls, indent=2)}\n"
                except TypeError:
                    log_str += f"Tool Calls: {msg.tool_calls}\n"
        log_str += "--- END: CONTEXT FOR LLM ---\n"
        logger.info(log_str)

    async def _decide_scene(self) -> Literal['web', 'desktop']:
        """
        Uses the LLM to decide which scene (web or desktop) to focus on for the next step.
        """
        try:
            memory_summary = "None"
            if self.state.history and self.state.history.model_thoughts():
                thoughts = self.state.history.model_thoughts()
                if thoughts:
                    memory_summary = thoughts[-1].memory

            last_action_summary = "None"
            if (self.state.history and self.state.history.history 
                and self.state.history.history[-1].model_output 
                and self.state.history.history[-1].model_output.action):
                last_action_summary = str(self.state.history.last_action())

            prompt = f"""
            Your ultimate task is: "{self.task}"
            Your most recent memory is: "{memory_summary}"
            The last action you took was: {last_action_summary}

            Based on this information, to most effectively continue the task, what is the next scene you need to observe?
            Choose between "web" for the web browser or "desktop" for the computer's desktop.
            Your answer must be a single word: either "web" or "desktop".
            """

            response = await self.llm.ainvoke(prompt)
            decision = response.content.strip().lower()

            logger.debug(f"Scene decision raw response: '{decision}'")

            if "desktop" in decision:
                return "desktop"
            return "web"
        except Exception as e:
            logger.error(f"Error during scene decision: {e}. Defaulting to 'web'.", exc_info=True)
            return "web"

    @time_execution_async('--get_next_action')
    async def get_next_action(self, messages: list[BaseMessage]) -> AgentOutput | None:
        """
        Gets the next action from the LLM by calling it and parsing the response.
        This method is overridden to handle scene-specific action model creation and add logging.
        """
        if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
            self.state.last_plan = await self._get_plan(messages)
            self.message_manager.add_plan(self.state.last_plan)
            messages = self.message_manager.get_messages()

        output_model: type[ActionModel]
        if self.current_scene == 'desktop':
            output_model = self.controller.registry.create_action_model(page=None)
        else:
            current_page = await self.browser_context.get_current_page()
            output_model = self.controller.registry.create_action_model(page=current_page)
        
        full_output_model = AgentOutput.type_with_custom_actions(output_model)

        input_messages = convert_input_messages(messages, self.model_name)

        try:
            response = await self.llm.ainvoke(input_messages)
            
            raw_response_content = response.content if isinstance(response, AIMessage) else str(response)
            logger.info(f"\n--- RAW LLM RESPONSE ---\n{raw_response_content}\n--- END RAW LLM RESPONSE ---\n")

            if self.settings.tool_calling_method in {'function_calling', 'tools'}:
                if not response.tool_calls:
                    return None
                tool_call = response.tool_calls[0]
                if 'args' not in tool_call or not isinstance(tool_call['args'], dict):
                    return None
                return full_output_model.model_validate(tool_call['args'])
            
            elif self.settings.tool_calling_method == 'json_mode':
                if isinstance(response, AIMessage):
                    try:
                        result_dict = extract_json_from_model_output(response.content)
                        return full_output_model.model_validate(result_dict)
                    except (json.JSONDecodeError, ValueError, ValidationError) as e:
                        logger.warning(f'Failed to parse model output: {response.content} {str(e)}')
                        return None
            else:  # raw or auto
                if isinstance(response, AIMessage):
                    try:
                        result_dict = extract_json_from_model_output(response.content)
                        return full_output_model.model_validate(result_dict)
                    except (json.JSONDecodeError, ValueError, ValidationError) as e:
                        logger.warning(f'Failed to parse model output: {response.content} {str(e)}')
                        return None
        except Exception as e:
            logger.error(f"Error getting next action from LLM: {e}", exc_info=True)
            return None
        
        return None

    async def _desktop_step(self, step_info: AgentStepInfo) -> None:
        """
        Handles the full logic for a step occurring in the 'desktop' scene.
        """
        # 1. Perceive Desktop Scene
        desktop_elements, screenshot_b64 = await self.controller.get_desktop_elements()
        self.last_desktop_screenshot = screenshot_b64
        
        elements_text = "\n".join([f"- {el['content']} ({el['type']})" for el in desktop_elements])
        state_description = f"[DESKTOP SCENE]\n\nYou are currently observing the desktop.\n\nVisible UI elements:\n{elements_text}"
        
        self.context_data['desktop_elements'] = desktop_elements
        
        state_message = HumanMessage(content=state_description)
        self.message_manager._add_message_with_tokens(state_message)

        # 2. Think & Act
        self.message_manager.cut_messages()
        messages = self.message_manager.get_messages()
        
        self._log_messages_for_llm(messages)

        model_output: AgentOutput | None = await self.get_next_action(messages)
        logger.info(f"\n--- PARSED AGENT OUTPUT ---\n{model_output}\n--- END PARSED AGENT OUTPUT ---\n")

        if not model_output or not model_output.action:
            logger.warning("LLM failed to produce a valid action. Creating empty action to proceed.")
            model_output = AgentOutput(
                current_state={"evaluation_previous_goal": "Failure", "memory": "LLM did not generate a valid action.", "next_goal": "Retry previous step."},
                action=[]
            )

        self.message_manager.add_model_output(model_output)
        
        result = await self.multi_act(model_output.action)
        self.state.last_result = result
        
        # 3. Record Desktop History
        history_state = BrowserStateHistory(
            url="desktop",
            title="Desktop",
            tabs=[],
            interacted_element=[None] * len(model_output.action),
            screenshot=self.last_desktop_screenshot
        )

        self.state.history.history.append(
            AgentHistory(
                model_output=model_output,
                result=self.state.last_result,
                state=history_state,
                metadata=None,
            )
        )

        # 4. Update failure count
        if any(r.error for r in self.state.last_result):
            self.state.consecutive_failures += 1
        else:
            self.state.consecutive_failures = 0
        
        # 5. 修正: 精确复制 v0.1.48 的日志记录逻辑
        if model_output:
            logger.info(f'🤷 Eval: {model_output.current_state.evaluation_previous_goal}')
            logger.info(f'🧠 Memory: {model_output.current_state.memory}')
            logger.info(f'🎯 Next goal: {model_output.current_state.next_goal}')
            for i, action in enumerate(model_output.action):
                logger.info(f'🛠️  Action {i + 1}/{len(model_output.action)}: {action.model_dump_json(exclude_unset=True)}')

    async def step(self, step_info: AgentStepInfo) -> None:
        """
        Overrides the base 'step' method to act as a scene router.
        """
        logger.info(f"--- Step {step_info.step_number + 1} ---")

        self.current_scene = await self._decide_scene()
        logger.info(f"Scene decision: Focusing on '{self.current_scene}'")

        if self.current_scene == 'desktop':
            await self._desktop_step(step_info)
        else:
            self._log_messages_for_llm(self.message_manager.get_messages())
            await super().step(step_info)

    async def multi_act(self, actions: list[ActionModel], check_for_new_elements: bool = True) -> list[ActionResult]:
        """
        Execute a list of actions, passing the agent's context to the controller.
        """
        if not actions:
            return [ActionResult(error="Received an empty action list to execute.")]

        results = []
        for action in actions:
            if not action.model_dump(exclude_unset=True):
                results.append(ActionResult(error="Received an empty action object {}."))
                continue

            result = await self.controller.act(
                action,
                self.browser_context,
                self.settings.page_extraction_llm,
                self.sensitive_data,
                self.settings.available_file_paths,
                context=self.context_data,
            )
            results.append(result)
        return results

    @time_execution_async("--run (agent)")
    async def run(
            self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None,
            on_step_end: AgentHookFunc | None = None
    ) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""

        loop = asyncio.get_event_loop()
        signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.pause,
            resume_callback=self.resume,
            custom_exit_callback=None,
            exit_on_second_int=True,
        )
        signal_handler.register()

        try:
            self._log_agent_run()

            if self.current_scene is None:
                self.current_scene = await self._decide_scene()
                logger.info(f"Initial scene decision: {self.current_scene}")

            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
                if self.state.paused:
                    signal_handler.wait_for_resume()
                    signal_handler.reset()

                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)
                    if self.state.stopped:
                        break

                if on_step_start is not None:
                    await on_step_start(self)

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                await self.step(step_info)

                if on_step_end is not None:
                    await on_step_end(self)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
                error_message = 'Failed to complete task in maximum steps'
                self.state.history.history.append(
                    AgentHistory(
                        model_output=None,
                        result=[ActionResult(error=error_message, include_in_memory=True)],
                        state=BrowserStateHistory(
                            url='', title='', tabs=[], interacted_element=[], screenshot=None,
                        ),
                        metadata=None,
                    )
                )
                logger.info(f'❌ {error_message}')

            return self.state.history

        except KeyboardInterrupt:
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history

        finally:
            signal_handler.unregister()

            if self.settings.save_playwright_script_path:
                logger.info(
                    f'Agent run finished. Attempting to save Playwright script to: {self.settings.save_playwright_script_path}'
                )
                try:
                    keys = list(self.sensitive_data.keys()) if self.sensitive_data else None
                    self.state.history.save_as_playwright_script(
                        self.settings.save_playwright_script_path,
                        sensitive_data_keys=keys,
                        browser_config=self.browser.config,
                        context_config=self.browser_context.config,
                    )
                except Exception as script_gen_err:
                    logger.error(f'Failed to save Playwright script: {script_gen_err}', exc_info=True)

            await self.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)