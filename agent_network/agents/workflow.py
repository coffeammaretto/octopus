"""
LangGraph workflow with Creator → Validator → Publisher chain.
Hard recursion_limit=3 for cycle protection.
Per-agent API key injection. Traceback handling in nodes.
Economic tracking via usage_metadata.
"""
import asyncio
import json
from typing import TypedDict, Annotated, List, Any, Optional
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types

import db
import analytics
from config_loader import AgentConfig
from agents.tools import search_tools


class AgentState(TypedDict):
    """State for the agent workflow."""
    topic: str
    channel_id: str
    template_name: str
    draft: str
    validation_result: str
    is_approved: bool
    post_id: Optional[int]
    tokens_used: int
    cost_usd: float
    search_results: List[dict]
    error: Optional[str]
    current_agent: str


def _get_gemini_client(api_key: str):
    """Create Gemini client with given API key."""
    return genai.Client(api_key=api_key)


async def creator_node(state: AgentState, agent_config: AgentConfig) -> AgentState:
    """Creator agent: generates draft content."""
    try:
        state["current_agent"] = "creator"
        
        # Check rate limit
        from rate_limiter import rate_limiter
        if not await rate_limiter.check_limit(agent_config.id, agent_config.rpm_limit, agent_config.tpm_limit):
            state["error"] = f"Rate limit exceeded for {agent_config.id}"
            return state
        
        client = _get_gemini_client(agent_config.gemini_api_key)
        
        # Build prompt with template injection
        template_prompt = ""
        if state.get("template_name"):
            # Template resolution happens via UI-driven config
            template_prompt = f"Use template style: {state['template_name']}. "
        
        # Search if needed
        search_context = ""
        if state.get("search_results"):
            search_context = "\n\nSearch results:\n" + "\n".join(
                f"- {r['title']}: {r['content']}" for r in state["search_results"]
            )
        
        prompt = f"""{template_prompt}Create an engaging Telegram post about: {state['topic']}
        
Requirements:
- Keep it concise and engaging (under 1000 characters)
- Use emojis appropriately
- Include relevant hashtags at the end
- Write in Russian language

{search_context}

Generate the post content:"""

        # Call Gemini with grounding enabled
        response = client.models.generate_content(
            model=agent_config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
            )
        )
        
        # Extract usage metadata for economic tracking
        tokens_in = 0
        tokens_out = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count or 0
            tokens_out = response.usage_metadata.candidates_token_count or 0
        
        cost = analytics.analytics_tracker.estimate_cost(tokens_in, tokens_out, agent_config.model)
        
        # Update rate limiter
        await rate_limiter.increment(agent_config.id, tokens_in + tokens_out)
        
        # Record analytics
        await analytics.analytics_tracker.record(
            agent_id=agent_config.id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model_name=agent_config.model
        )
        
        state["draft"] = response.text or ""
        state["tokens_used"] = tokens_in + tokens_out
        state["cost_usd"] = cost
        
        return state
        
    except Exception as e:
        state["error"] = f"Creator error: {str(e)}"
        return state


async def validator_node(state: AgentState, agent_config: AgentConfig) -> AgentState:
    """Validator agent: reviews and approves/rejects draft."""
    try:
        state["current_agent"] = "validator"
        
        # Check rate limit
        from rate_limiter import rate_limiter
        if not await rate_limiter.check_limit(agent_config.id, agent_config.rpm_limit, agent_config.tpm_limit):
            state["error"] = f"Rate limit exceeded for {agent_config.id}"
            return state
        
        client = _get_gemini_client(agent_config.gemini_api_key)
        
        prompt = f"""Review this Telegram post draft for quality and accuracy:

{state['draft']}

Check for:
1. Factual accuracy
2. Appropriate tone for Telegram
3. No harmful or misleading content
4. Proper formatting

Respond with ONLY 'APPROVED' or 'REJECTED' followed by a brief explanation."""

        response = client.models.generate_content(
            model=agent_config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
            )
        )
        
        # Extract usage metadata
        tokens_in = 0
        tokens_out = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count or 0
            tokens_out = response.usage_metadata.candidates_token_count or 0
        
        cost = analytics.analytics_tracker.estimate_cost(tokens_in, tokens_out, agent_config.model)
        
        await rate_limiter.increment(agent_config.id, tokens_in + tokens_out)
        await analytics.analytics_tracker.record(
            agent_id=agent_config.id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model_name=agent_config.model
        )
        
        state["validation_result"] = response.text or ""
        state["is_approved"] = "APPROVED" in (response.text or "").upper()
        state["tokens_used"] += tokens_in + tokens_out
        state["cost_usd"] += cost
        
        return state
        
    except Exception as e:
        state["error"] = f"Validator error: {str(e)}"
        return state


async def publisher_node(state: AgentState, agent_config: AgentConfig) -> AgentState:
    """Publisher agent: saves post to database (no actual TG send in simulation)."""
    try:
        state["current_agent"] = "publisher"
        
        if not state.get("is_approved"):
            # Save as rejected
            post_id = await db.insert_post(
                channel_id=state["channel_id"],
                content=state.get("draft", ""),
                status="rejected",
                validation_result=state.get("validation_result", "")
            )
            state["post_id"] = post_id
            return state
        
        # Save approved post
        post_id = await db.insert_post(
            channel_id=state["channel_id"],
            content=state.get("draft", ""),
            status="approved",
            validation_result=state.get("validation_result", ""),
            template_name=state.get("template_name", ""),
        )
        
        await db.update_post_status(
            post_id=post_id,
            status="pending",
            tokens_used=state.get("tokens_used", 0),
            cost_usd=state.get("cost_usd", 0.0)
        )
        
        state["post_id"] = post_id
        
        # Log audit event
        await db.log_audit_event(
            event_type="post_published",
            message=f"Post {post_id} created for channel {state['channel_id']}",
            severity="info",
            details=json.dumps({"topic": state["topic"], "tokens": state["tokens_used"]})
        )
        
        return state
        
    except Exception as e:
        state["error"] = f"Publisher error: {str(e)}"
        return state


def create_workflow(agent_configs: dict) -> StateGraph:
    """Create LangGraph workflow with per-agent API routing."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes with bound agent configs
    creator_cfg = agent_configs.get("creator")
    validator_cfg = agent_configs.get("validator")
    publisher_cfg = agent_configs.get("publisher")
    
    # Bind configs to nodes using partial-like pattern
    async def creator_bound(state):
        return await creator_node(state, creator_cfg)
    
    async def validator_bound(state):
        return await validator_node(state, validator_cfg)
    
    async def publisher_bound(state):
        return await publisher_node(state, publisher_cfg)
    
    workflow.add_node("creator", creator_bound)
    workflow.add_node("validator", validator_bound)
    workflow.add_node("publisher", publisher_bound)
    
    # Define edges
    workflow.set_entry_point("creator")
    workflow.add_edge("creator", "validator")
    
    # Conditional edge after validation
    def should_publish(state):
        if state.get("is_approved"):
            return "publisher"
        return "end"
    
    workflow.add_conditional_edges(
        "validator",
        should_publish,
        {"publisher": "publisher", "end": END}
    )
    
    workflow.add_edge("publisher", END)
    
    # Compile with hard recursion limit
    return workflow.compile(recursion_limit=3)


async def run_workflow(topic: str, channel_id: str, 
                       creator_cfg: AgentConfig, 
                       validator_cfg: AgentConfig,
                       publisher_cfg: AgentConfig,
                       template_name: str = "",
                       do_search: bool = False) -> AgentState:
    """Run the complete workflow."""
    
    # Initialize state
    initial_state: AgentState = {
        "topic": topic,
        "channel_id": channel_id,
        "template_name": template_name,
        "draft": "",
        "validation_result": "",
        "is_approved": False,
        "post_id": None,
        "tokens_used": 0,
        "cost_usd": 0.0,
        "search_results": [],
        "error": None,
        "current_agent": ""
    }
    
    # Perform search if requested
    if do_search:
        search_results = await search_tools.search(topic)
        initial_state["search_results"] = search_results
    
    # Create workflow with agent configs
    agent_configs = {
        "creator": creator_cfg,
        "validator": validator_cfg,
        "publisher": publisher_cfg
    }
    
    graph = create_workflow(agent_configs)
    
    # Run graph with error handling
    try:
        result = await graph.ainvoke(initial_state)
        return result
    except Exception as e:
        return {
            **initial_state,
            "error": f"Workflow execution failed: {str(e)}"
        }
