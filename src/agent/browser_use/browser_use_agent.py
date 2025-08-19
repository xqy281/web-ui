# src/agent/browser_use/browser_use_agent.py

from __future__ import annotations

import asyncio
import logging
import os
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
from browser_use.agent.message_manager.utils import is_model_without_tool_support
from langchain_core.messages import HumanMessage

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
            if self.state.history and self.state.history.last_action():
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
        
        model_output: AgentOutput | None = await self.get_next_action(messages)

        if not model_output:
            self.state.consecutive_failures += 1
            return

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
                result=result,
                state=history_state,
                metadata=None,
            )
        )

        # 4. Update failure count
        if any(r.error for r in result):
            self.state.consecutive_failures += 1
        else:
            self.state.consecutive_failures = 0

    async def step(self, step_info: AgentStepInfo) -> None:
        """
        Overrides the base 'step' method to act as a scene router.
        """
        logger.info(f"--- Step {step_info.step_number + 1} ---")

        self.current_scene = await self._decide_scene()
        logger.info(f"Scene decision: Focusing on '{self.current_scene}'")

        if self.current_scene == 'desktop':
            await self._desktop_step(step_info)
        else:  # 'web' scene
            # Delegate to the original, robust implementation for web logic
            await super().step(step_info)


    async def multi_act(self, actions: list[ActionModel], check_for_new_elements: bool = True) -> list[ActionResult]:
        """
        Execute a list of actions, passing the agent's context to the controller.
        """
        results = []
        for action in actions:
            result = await self.controller.act(
                action,
                self.browser_context,
                self.settings.page_extraction_llm,
                self.sensitive_data,
                self.settings.available_file_paths,
                context=self.context_data,
            )
            results.append(result)
            
            if self.current_scene == 'web':
                if await self.browser_context.check_for_page_change(check_for_new_elements):
                    break
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