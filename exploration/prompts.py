# openrlhf/agent/exploration/prompts.py
"""
All prompt templates for the exploration pipeline.

Prompts are organized into two groups:
- Online exploration: used during the active exploration phase
- Post-processing: used for offline trajectory analysis

All prompts are English-only for consistency.
"""

# ============================================================
# Online Exploration Prompts
# ============================================================

EXPLORATION_PROMPT_TEMPLATE_EN = """# Role: Autonomous GUI Exploration Agent
You are a professional autonomous AI agent. Your core task is to simulate and generate **1~{max_candidates}** possible different user actions based on the current GUI environment screenshot and the user's historical behavior trajectory, and to design reasonable single-step goals and overall goals for each action.
You need to analyze screen information, infer user intent, and generate several reasonable, direct "next actions" and the corresponding final goals.

---
### 1. Generation Principles
When generating the candidate list `candidates` for the next move, please strictly adhere to the following principles:
*   **Coherence Priority:**
    *   The generated next action must be logically coherent with the historical trajectory, capable of forming a reasonable and complete task flow. This is the highest priority.
    *   The generated actions should progressively delve deeper into the current scene's various functionalities compared to previous steps, rather than being another task unrelated to the historical steps. Moreover, one should not readily return to or undo past operations. (When you feel that there is no more room for in-depth exploration in the current task scenario, you can choose to end it.)
*   **Determine the number of candidate actions based on the scenario:**
    *   **Scenario**: If the historical trajectory shows the user is in an unfinished action sequence (e.g., menu opened waiting for selection, dialog popped up waiting for confirmation, currently typing text), generate 1~3 reasonable and necessary actions. Do not cancel operations or leave the current interface without cause, interrupting the current task.
    *   **Scenario**: If the current interface has multiple parallel functional entries (e.g., multiple icons on a toolbar), generate as many different types of future actions as possible. There should be obvious type differences between different branch actions, pointing to different kinds of functions, rather than several actions that are similar in type but differ only in details.
    *   **Scenario**: If the current trajectory can already form a complete, valuable GUI task, you may generate candidate actions containing the termination action 'done'.
---

### 2. Detailed Requirements for Fields
Please generate the following fields for each candidate action:
#### A. step_reason (Simulate Thinking Process)
    Describe which **key GUI elements** (coordinates, text, icons) related to the goal were seen in the current screenshot, and explain the reason for taking the action and the expected effect.

#### B. step_goal (Current Goal)
*   **step_goal (Single-step Goal)**: Describe specifically what to do in this step (e.g., "Click 'File' menu"). Only describe the single-step action itself; do not include the impact on future goals (e.g., do not add "explore... function").

#### C. step_action (Single-step Action)
*   Each step action contains a set of actions and parameters defined in the `computer_use` toolset apart from the 'screenshot' action (such as coordinates, text, etc.).
*   **Handling Latency**: If an action (such as opening a webpage, launching a program) will trigger loading, please append a `wait` action to the action list. If the current screen is currently loading from the previous step, generate only a `wait` action.

#### D. final_goal (Future Goal)
    *   Describe a phase-based task with clear boundaries that can be completed within a reasonable number of steps (1~5 steps) in the future.
    *   **Dynamic Correction**: If the historical trajectory deviates from the `final_goal`, or if you obtain more specific information during exploration, you can redesign a `final_goal` that fits the trajectory based on the latest status; if multiple branches are generated, different `final_goal`s need to be designed for each branch.
    *   The `last_final_goal` designed in the previous step is for reference only. The `final_goal` in this step can be the same, corrected, or completely different, as long as it ensures the coherence of the historical trajectory.
    *   **Consistency**: The `final_goal` and the historical trajectory must be coherent and reasonable, able to explain all actions from the first step to the current step, connecting the historical trajectory into a complete task story.
    *   **Precision**: `final_goal` must be a specific, actionable task with clear completion markers, rather than a broad goal or vague intent.

#### E. expected_observation
*   Describe in detail the specific expected changes in the GUI interface after `step_action` is successfully executed (e.g., what window pops up, what text appears, how the page jumps).

---
### 3. Task Termination Logic:
1. Termination action: `[{{"action": "done"}}]`.
2. If the current trajectory can already form a valuable GUI task and total steps >= 8, the output actions should include a termination action.
3. The maximum step length for a single exploration is 15. When total steps >= 8, if you believe the current state can serve as a reasonable ending point, then the output action should only contain the termination action.

---

### 4. Constraints:
1.  **Task Autonomy**: The generation process does not require user intervention (e.g., do not design tasks requiring mobile verification codes or user account logins).
2.  **Valuable Operations**: Focus on exploration of practical significance such as editing, searching, setting, modifying, etc., avoiding purposeless random browsing.
3.  **Format Constraints**: The output must be pure JSON. It is strictly forbidden to include any explanatory text other than Markdown code block markers (```json ... ```).
4.  **Non-Tool Call**: Please note, you need to generate data for several candidate actions organized in the JSON format below, rather than returning a single-step `tool_call` result. Since the screenshots have already been provided, please do not take the action of 'screenshot'.
5. **Maintain Continuity**: Do not interrupt or undo previous actions, such as opening a window and immediately closing it without making any valuable modifications, or selecting a piece of text and immediately canceling the selection.
---

### Extra Guidance
{extra_guidance}

### Input Information
**1. Current GUI Screenshot:**
[Screenshot Data]

**2. Historical Action Trajectory:**
{history}

---

### Output Format

Please strictly output in the following JSON format:
```json
{{
  "candidates": [
    {{
      "step_goal": "String, the specific goal of this current step",
      "expected_observation": "String, detailed description of expected interface changes after execution",
      "step_reason": "String, detailed visual analysis and decision-making process",
      "step_action": [
        {{
          "action": "Tool name, e.g., left_click",
          "coordinate": [x, y],
          "text": "Include this field if it is an input operation"
        }},
        ...
      ],
      "final_goal": "String, corrected or summarized phase-based future goal"
    }},
    {{
       "step_goal": "Task completed",
       "expected_observation": "Task ends",
       "step_reason": "Detected that the goal has been achieved/steps reached the limit, ending the task.",
       "step_action": [{{"action": "done"}}],
       "final_goal": "String, long-term goal"
    }}
  ]
}}
```
"""


VERIFICATION_PROMPT_TEMPLATE = """
You are an AI assistant for GUI operation verification. Your goal is to classify the operation result by comparing screenshots taken before and after the operation.

You will receive the following inputs: a **screenshot before operation**, the **executed action**, the **expected outcome**, and a **screenshot after operation**.

**Your task is to select only the one category that best describes the result from the following four categories:**
1.  **SUCCESS**: The screenshot after operation is clear and completely matches the expected outcome.
2.  **NO_CHANGE**: The screenshot after operation differs from the expected outcome and shows absolutely no change, implying the operation had no effect.
3.  **UNEXPECTED_CHANGE**: The screen has changed, but it does *not match* the expected outcome. This includes showing error messages, popping up different menus, or being in an incorrect state.
4.  **NEEDS_MORE_TIME**: The screen is in an obvious loading state (e.g., visible progress bar, spinning icon, or "Loading..." text), indicating the operation is still in progress.

**Output Format:**
You must strictly follow the JSON format below for your response. Do not include any other text or explanation. Please select the most appropriate category.
The language used in the analysis should be consistent with the language of the input target.

```json
{{
  "result_type": "SUCCESS | NO_CHANGE | UNEXPECTED_CHANGE | NEEDS_MORE_TIME",
  "feedback": "A brief explanation for your selection of this category."
}}
```
---
**Current Task Details:**

**Screenshot Before Operation (First Image):**
<image>

**Executed Action:**
`{action}`

**Expected Outcome:**
`{expected_outcome}`

**Screenshot After Operation (Second Image):**
<image>
"""


SUMMARY_PROMPT_TEMPLATE = """
# Role
You are an expert action trajectory analyst. Your goal is to analyze a series of user operation logs (`history`) and convert them into structured, high-quality imperative task instructions. You need to make this trajectory the standard answer for completing this task. Each step is essential and necessary for finishing the task.

# Task
- Analyze the Global Goal: Connect all the user's historical actions together and merge them into a coherent final_task_summary to serve as the task description for the entire trajectory.
- Try to summarize all the steps in the "task_summary".

---

# Guidelines for generating task_summary:
1. Ensure Completeness and Accuracy
The instruction must include all important, achieved goals from the operation history. No core sub-task can be omitted.
This is the most important principle.
All the steps that should be fully mentioned must be included.

2. Use the Imperative Mood
Instructions must start with a verb, directly commanding an action. The final output must be an imperative instruction.

3. Be Concrete and Unambiguous
Only use specific and clear action descriptions as the task_summary. There is no need to retain vague descriptions of intentions.
*   Vague Instruction: "To explore the text formatting options."
*   Specific Instruction: "Set the font of the selected text to 'Arial', the size to 12pt, and apply bold formatting."

4. Be Concise and Highly Abstracted
If you are 100 percent certain that a step is a necessary prerequisite for the next step, you can appropriately combine the steps.
You can abstract low-level, trivial UI interactions (like multiple clicks, typing, dragging) into high-level, generalized operational commands.
*   Low-level Action Flow: `Click text box -> Type 'Hello' -> Click font menu -> Select 'Arial'`
*   High-level Instruction: "Enter the text 'Hello' and change its font to 'Arial'."


### Input History
{history}
```

---
### Output Format
You must strictly adhere to the JSON format below in your response, without including any additional text or explanations.
The language of the task instruction should be consistent with the language of the input history.
```json
{{
  "final_task_summary":"<The overall task instruction summarizing the entire history>",
}}
```
"""


THINKING_PROMPT_TEMPLATE = """
## ROLE ##
You are a top-tier AI agent, an expert at executing complex tasks on a computer desktop. You excel not only at completing the current step but also at long-term strategic planning.

## TASK ##
Your task is to reverse-engineer the "Thinking Process" for each step of a successful operational trajectory. This thinking process must clearly explain "why this specific action was taken in the current state" and must demonstrate foresight and planning for future steps.

## CONTEXTUAL INFORMATION ##
You will be provided with the complete context for a specific step, including the global task, the current state, and the subsequent future steps.

---
### 1. Global Task Objective ###
{global_task_summary}

### 2. Current Step's Goal and Action ###
- Step Goal: {current_step_goal}
- Action Command: {current_action_command}

### 3. Future Sequential Steps (Planning Horizon) ###
{future_steps_formatted}

## YOUR REQUIREMENTS ##
Based on the information above, generate a structured "Thinking Process". This process must include the following four parts, strictly adhering to the specified format:

1.  **Observation:** Briefly describe the state before the action was executed. Although there is no image, infer the key elements on the screen based on the "Current Step's Goal" and the "Global Task".
2.  **Reasoning:** This is the most critical part. Explain why the "Current Step's Goal" is a logical and necessary step towards achieving the "Global Task Objective". **Explicitly state how the current action paves the way for the "Future Sequential Steps"**. Showcase your planning ability.
3.  **Anticipation:** Describe what specific changes you expect to see on the screen after executing the "Action Command".
4.  **Self-Correction/Confirmation:** Briefly state how you will proceed if the anticipation is met, and what you might do if it is not.

## OUTPUT FORMAT ##
Please return your response strictly in JSON format, with a single key "thinking". The value should be a string containing Markdown headers.

{
  "thinking": "#### Observation\\n...\\n\\n#### Reasoning\\n...\\n\\n#### Anticipation\\n...\\n\\n#### Self-Correction/Confirmation\\n..."
}
---
"""


AVOID_REPETITION_PROMPT = """
---
To maximize test coverage, we have recorded the action combinations already attempted by **all other exploration trajectories** in the initial phase.

**You MUST avoid repeating the following "first two action combinations" and explore entirely new functional entry points or operational logic!**

Existing starting paths (Avoid these exact Step 1 -> Step 2 combinations):
{forbidden_paths}

**Requirements:**
1. Carefully read the above list. If your plan completely overlaps with any entry, immediately change your strategy.
2. Try clicking different menus, using different toolbar buttons, or operating on different objects.
3. If the current interface has only one reasonable path (e.g., you must click "OK" to proceed), it may be repeated, but you must clearly explain the reason in `step_reason`.
---
"""


# ============================================================
# Post-Processing Prompts
# ============================================================

POSTPROCESS_SUMMARY_OVERALL_PROMPT = """
# Role
You are an expert action trajectory analyst. Your goal is to analyze a series of user operation logs and convert them into a single, concise, high-quality imperative task instruction.

# Task
Connect all the user's historical actions together and merge them into one coherent task description. Each step in the history is essential and necessary for finishing the task.

# Guidelines
1. **Completeness and Accuracy**: The instruction must include all important, achieved goals from the operation history. No core sub-task can be omitted.
2. **Imperative Mood**: Instructions must start with a verb, directly commanding an action.
3. **Be Specific**: Use concrete action descriptions. Avoid vague intentions.
   - Vague: "To explore the text formatting options."
   - Specific: "Set the font of the selected text to 'Arial', size 12pt, and apply bold."
4. **Be Concise**: Abstract low-level UI interactions into high-level commands.
   - Low-level: "Click text box -> Type 'Hello' -> Click font menu -> Select 'Arial'"
   - High-level: "Enter the text 'Hello' and change its font to 'Arial'."

### Input History
{history}

---
### Output Format
You must strictly adhere to the JSON format below. Do not include any other text.
```json
{{
  "final_task_summary": "<The overall task instruction summarizing the entire history>"
}}
```
"""


POSTPROCESS_SUMMARY_STAGES_PROMPT = """
# Role
You are an expert action trajectory analyst. Your goal is to analyze a series of user operation logs and restructure them into meaningful, classified task stages.

# Task
1. Break the history into logical stages based on intent completion.
2. Classify each stage into one of three categories.
3. Generate a final effective-only task summary.

# Stage Categories

### A. EFFECTIVE (High-value tasks with meaningful state changes)
- The steps can be summarized as a high-level cognitive task where the user successfully modified parameters, applied tools, changed view layouts, confirmed settings, or obtained valuable information.
- Includes all prerequisite navigation steps needed to reach that result.
- Typically 3-7 steps.

### B. NAVIGATION (Valuable interface access without state changes)
- The user successfully navigated to a meaningful, non-trivial interface but did not modify any values or obtain valuable information.
- If this is a prerequisite to an EFFECTIVE stage, merge it into that stage instead.

### C. NOISE (Ineffective operations)
- Failed operations (NO_CHANGE or UNEXPECTED_CHANGE verification results).
- Redundant operations (switching menus without making selections).
- These should be identified and excluded from the final task description.

# Guidelines for Stage Summaries
1. **Consistency**: The generated stage summaries and final task description must completely and accurately explain all operations from the first step to the last.
2. **Result-oriented**: Focus on the final result. If an action is a prerequisite for another action (e.g., opening a menu to click an option), ignore the procedural action and describe only the core action.
3. **Precise state description**: Include specific values, option names, or parameters from the history.
4. **Imperative mood**: All task descriptions must start with a verb.

### Input History
{history}

---
### Output Format
You must strictly adhere to the JSON format below.
```json
{{
  "stages": [
    {{
      "start_step": <int>,
      "end_step": <int>,
      "stage_summary": "<Precise, result-oriented task summary>",
      "category": "<EFFECTIVE | NAVIGATION | NOISE>"
    }}
  ],
  "final_task_summary_effective": "<Coherent long sentence composed from EFFECTIVE stages only>"
}}
```
"""


POSTPROCESS_SCORING_SYSTEM_PROMPT = """
You are an expert Data Quality Analyst for GUI Agent training data.
Evaluate the user trajectory based on the "Target Task" and the "Execution Storyline".

### Scoring Scale (0-3)

**Metric 1: Task Utility & Realism (0-3)**
*Focus: Is this a valuable real-world task, or just meaningless browsing?*
- **3 (High Value)**: The entire process involved valuable editing, design, and modification operations that produced meaningful, realistic user-like impacts on the environment, demonstrating a certain level of difficulty and depth—not merely simple clicking and browsing.
- **2 (Moderate Value)**: A valid but simple task (e.g., simple editing operations). The intent is clear, though the scope is limited.
- **1 (Low Value)**: Includes a large number of simple browsing clicks with little to no value, having almost no impact on the environment.
- **0 (No Value)**: Pure meaningless browsing. The user is just clicking around randomly with no valuable goal. The trajectory represents noise rather than a task.

**Metric 2: Step Efficiency & Validity (0-3)**
*Focus: Penalize failed steps, useless actions, and "hanging" starts.*
- **3 (Pure)**: Free of all negative patterns. No failed steps. No actions that yield zero info. No redundant steps. The trajectory is complete (does not stop immediately after opening an interface).
- **2 (Acceptable)**: High quality but contains trace amounts of inefficiency. May have 1-2 useless clicks, a very minor detour, failed steps or "hanging" incomplete steps at the end.
- **1 (Noisy)**: Contains a lot of negative patterns: (a) Actions that failed/errored, (b) Steps that provided no new information or value, OR (c) The task feels abruptly cut off (incomplete).
- **0 (Dirty)**: Dominated by negative patterns. The trajectory is either: (a) Mostly failed/useless actions, (b) Just opening an interface and immediately ending (hanging start), or (c) Providing absolutely no help toward the final goal.

**Metric 3: Task-Action Consistency (0-3)**
*Focus: All successful steps in the trajectory are strictly necessary and sufficient conditions for completing the final task.*
- **3 (Perfect Match)**: The executed actions perfectly align with the text description.
- **2 (General Match)**: A small number of steps were successfully executed but did not contribute to completing the target task, or the task description was insufficiently clear.
- **1 (Weak Match)**: The actions diverge significantly; only tangentially related.
- **0 (Mismatch)**: The user is doing something completely different from the described task.

**Metric 4: Action Coherence & Continuity (0-3)**
*Focus: Is the flow logical? Penalize "Context Switching" without completion.*
- **3 (Smooth Flow)**: Strong causal links. The user sticks to one context until the sub-task is done.
- **2 (Minor Detours)**: Generally logical, but may momentarily lose focus before correcting (less than 2 steps).
- **1 (Fragmented)**: "Context Hopping." Opens an app/menu, does nothing meaningful, then jumps to another context.
- **0 (Chaotic)**: Random jumping between unrelated apps or UI elements with no logic.

### Output Format (JSON):
{
    "reason": "<Concise analysis covering utility, step purity, and coherence>",
    "task_utility": <int 0-3>,
    "step_efficiency": <int 0-3>,
    "task_consistency": <int 0-3>,
    "action_coherence": <int 0-3>
}
"""


POSTPROCESS_REASON_SYNTHESIS_PROMPT = """
## Background
You are an expert at reverse-engineering the thinking process for GUI agent trajectories. You will receive a GUI task, a sequence of exploration steps, and the current step to analyze. Generate a coherent, task-oriented thinking process for the current step by reasoning backward from the final task.

## Task: {final_task}

## History Trajectory (steps already completed):
{history_trajectory}

## Current Step (Step {step_id}):
- Goal: {goal}
- Verification Result: {verification_result}

## Future Trajectory (remaining steps to complete the task):
{future_trajectory}

## Requirements
Based on the information above, generate a structured thinking process for the current step. This must include the following four components:

### 1. Observation
Analyze the current UI state before the action. Identify key elements (menus, buttons, dialogs, parameters, etc.) relevant to the current step and the final task.

### 2. Progress
Summarize what has been accomplished so far based on the history trajectory. Explain how the latest state changes relate to the overall task progress.

### 3. Plan
Describe the plan moving forward, covering multiple time scales:
- The specific action to take in the current step
- The immediate next steps
- The longer-term strategy to complete the final task

### 4. Impact
Explain how the current step advances progress toward the final goal:
- Short-term impact: The direct UI changes expected from the current action
- Long-term impact: How this step fits into the overall task completion

## Important Constraints
1. **Task-oriented reasoning**: Derive the thinking from the final task, not from the original exploration intent.
2. **Backward reasoning**: Use the final task to determine what each step's reasoning should have been.
3. **Strong coherence**: Each step's thinking must inherit from all prior steps and set up subsequent steps.
4. **Goal consistency**: All thinking must serve completing the final task.
5. **Language**: Write the thinking process in {language}.

## Output Format
Return a JSON object with a single key "synthesized_thoughts" containing the four components:
```json
{{
  "synthesized_thoughts": {{
    "observation": "<analysis of current UI elements>",
    "progress": "<summary of history and overall task progress>",
    "plan": "<current step, immediate next steps, and longer-term strategy>",
    "impact": "<direct impact and role in overall task>"
  }}
}}
```
"""
