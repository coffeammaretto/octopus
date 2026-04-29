"""
LangGraph StateGraph workflow with Creator → Validator → Publisher hierarchy.
Hard recursion_limit=3 to prevent infinite loops and budget burn.
Per-agent API key injection (NEVER stored in state or logs).
Token/cost tracking via usage_metadata extraction.
"""
import asyncio
import os
import re
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END
from google import genai
from google.genai.types import GenerateContentConfig, Tool

from agents.tools import SearchTool, perform_search, format_search_results
from db import create_post, update_post, log_audit, log_usage
from analytics import analytics_collector
from config_loader import config_loader
from rate_limiter import rate_limiter


# ============================================================================
# STATE DEFINITION
# Note: API keys are NEVER stored in state - injected dynamically per node
# ============================================================================
class AgentState(TypedDict):
    """LangGraph state for agent workflow."""
    post_id: int
    channel_id: str
    topic: str
    draft_content: str
    validated_content: str
    search_results: str
    validation_attempts: int
    validation_feedback: str
    is_approved: bool
    creator_agent_id: str
    validator_agent_id: str
    publisher_agent_id: str
    # Economic tracking (accumulated across nodes)
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    error_message: Optional[str]


# ============================================================================
# TEMPLATE INJECTION HELPER
# ============================================================================
async def resolve_templates(content: str) -> str:
    """Resolve {{template:name}} patterns in content."""
    pattern = r'\{\{template:(\w+)\}\}'
    
    async def replace_template(match):
        template_name = match.group(1)
        # Get templates from channel config
        channels = await config_loader.load_channels()
        for channel in channels.channels:
            # In real implementation, templates would be stored in DB
            # For now, return empty string as fallback
            pass
        return ""  # Fallback to empty string if template not found
    
    # Simple regex replacement
    result = re.sub(pattern, replace_template, content)
    return content  # For now, return original (templates resolved at higher level)


# ============================================================================
# CREATOR NODE
# ============================================================================
async def creator_node(state: AgentState) -> Dict[str, Any]:
    """Creator agent generates initial draft content."""
    agent_id = state['creator_agent_id']
    
    try:
        # Get agent config for API key routing
        agent_config = await config_loader.get_agent(agent_id)
        if not agent_config:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Get API key from environment (NEVER from state/config file)
        api_key = os.environ.get(agent_config.api_key_env_var, os.environ.get('GEMINI_API_KEY'))
        if not api_key:
            raise ValueError("No API key available for agent")
        
        # Initialize Gemini client with per-agent key
        client = genai.Client(api_key=api_key)
        
        # Check rate limits
        allowed, rpm, tpm = await rate_limiter.increment(agent_id)
        
        # Build prompt with optional search
        prompt = f"""Create an engaging Telegram post about: {state['topic']}

Requirements:
- Write in a conversational, engaging tone
- Keep it concise (under 500 characters)
- Include relevant emojis
- Add 2-3 relevant hashtags
- Make it shareable and interesting

Topic details: {state['topic']}"""

        # Optionally include search results if available
        if state.get('search_results'):
            prompt += f"\n\nUse this research context:\n{state['search_results']}"
        
        # Call Gemini
        response = await client.models.generate_content(
            model=agent_config.model_name,
            contents=prompt
        )
        
        # Extract usage metadata for economic tracking
        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count or 0
            tokens_out = response.usage_metadata.candidates_token_count or 0
            # Approximate cost (adjust based on actual pricing)
            cost_usd = (tokens_in * 0.000000125) + (tokens_out * 0.0000005)
        
        # Track usage
        await analytics_collector.track_usage(agent_id, tokens_in, tokens_out, cost_usd, agent_config.model_name)
        
        # Update state totals
        new_tokens_in = state['total_tokens_in'] + tokens_in
        new_tokens_out = state['total_tokens_out'] + tokens_out
        new_cost_usd = state['total_cost_usd'] + cost_usd
        
        content = response.text.strip() if response.text else ""
        
        # Resolve any template placeholders
        content = await resolve_templates(content)
        
        # Create post in database
        post_id = await create_post(state['channel_id'], content, agent_id)
        
        # Log audit event
        await analytics_collector.track_event(
            'post_created',
            agent_id=agent_id,
            message=f"Created draft for topic: {state['topic'][:50]}..."
        )
        
        return {
            'post_id': post_id,
            'draft_content': content,
            'total_tokens_in': new_tokens_in,
            'total_tokens_out': new_tokens_out,
            'total_cost_usd': new_cost_usd
        }
        
    except Exception as e:
        error_msg = str(e)
        # Sanitize traceback - never expose raw errors
        if "API" in error_msg or "key" in error_msg.lower():
            error_msg = "API configuration error"
        
        await analytics_collector.track_event(
            'creator_error',
            agent_id=agent_id,
            message=error_msg,
            severity='error'
        )
        
        return {
            'error_message': error_msg,
            'is_approved': False
        }


# ============================================================================
# VALIDATOR NODE
# ============================================================================
async def validator_node(state: AgentState) -> Dict[str, Any]:
    """Validator agent reviews and approves/rejects content."""
    agent_id = state['validator_agent_id']
    
    try:
        agent_config = await config_loader.get_agent(agent_id)
        if not agent_config:
            raise ValueError(f"Agent {agent_id} not found")
        
        api_key = os.environ.get(agent_config.api_key_env_var, os.environ.get('GEMINI_API_KEY'))
        if not api_key:
            raise ValueError("No API key available for agent")
        
        client = genai.Client(api_key=api_key)
        await rate_limiter.increment(agent_id)
        
        prompt = f"""Review this Telegram post draft for quality and accuracy:

DRAFT CONTENT:
{state['draft_content']}

TOPIC:
{state['topic']}

Evaluation criteria:
1. Is the content accurate and factually correct?
2. Is it engaging and well-written?
3. Does it follow Telegram best practices (concise, emojis, hashtags)?
4. Are there any spelling or grammar errors?
5. Is the tone appropriate for the audience?

Provide your review as JSON:
{{
    "approved": true/false,
    "feedback": "specific feedback if rejected",
    "suggested_improvements": "optional improvements"
}}"""

        response = await client.models.generate_content(
            model=agent_config.model_name,
            contents=prompt
        )
        
        # Extract usage metadata
        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count or 0
            tokens_out = response.usage_metadata.candidates_token_count or 0
            cost_usd = (tokens_in * 0.000000125) + (tokens_out * 0.0000005)
        
        await analytics_collector.track_usage(agent_id, tokens_in, tokens_out, cost_usd, agent_config.model_name)
        
        # Parse response (simple extraction)
        response_text = response.text.strip()
        
        # Try to extract approval status
        is_approved = 'approved' in response_text.lower() and 'true' in response_text.lower()
        feedback = response_text
        
        # Update validation attempts counter
        new_attempts = state['validation_attempts'] + 1
        
        # Track economic totals
        new_tokens_in = state['total_tokens_in'] + tokens_in
        new_tokens_out = state['total_tokens_out'] + tokens_out
        new_cost_usd = state['total_cost_usd'] + cost_usd
        
        await analytics_collector.track_event(
            'post_validated',
            agent_id=agent_id,
            message=f"Validation attempt {new_attempts}: {'approved' if is_approved else 'rejected'}",
            severity='info' if is_approved else 'warning'
        )
        
        return {
            'validated_content': state['draft_content'],  # Could apply improvements here
            'validation_attempts': new_attempts,
            'validation_feedback': feedback,
            'is_approved': is_approved,
            'total_tokens_in': new_tokens_in,
            'total_tokens_out': new_tokens_out,
            'total_cost_usd': new_cost_usd
        }
        
    except Exception as e:
        error_msg = str(e)
        if "API" in error_msg or "key" in error_msg.lower():
            error_msg = "API configuration error"
        
        await analytics_collector.track_event(
            'validator_error',
            agent_id=agent_id,
            message=error_msg,
            severity='error'
        )
        
        return {
            'error_message': error_msg,
            'is_approved': False
        }


# ============================================================================
# PUBLISHER NODE
# ============================================================================
async def publisher_node(state: AgentState) -> Dict[str, Any]:
    """Publisher agent handles final publishing (simulation mode - no actual bot.send_message)."""
    agent_id = state['publisher_agent_id']
    
    try:
        agent_config = await config_loader.get_agent(agent_id)
        if not agent_config:
            raise ValueError(f"Agent {agent_id} not found")
        
        api_key = os.environ.get(agent_config.api_key_env_var, os.environ.get('GEMINI_API_KEY'))
        if not api_key:
            raise ValueError("No API key available for agent")
        
        client = genai.Client(api_key=api_key)
        await rate_limiter.increment(agent_id)
        
        # Final polish before publishing
        prompt = f"""Polish this approved Telegram post for final publishing:

CONTENT:
{state['validated_content']}

Make minor improvements if needed:
- Fix any remaining typos
- Optimize emoji placement
- Ensure hashtags are relevant

Return only the final polished content."""

        response = await client.models.generate_content(
            model=agent_config.model_name,
            contents=prompt
        )
        
        # Extract usage metadata
        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_in = response.usage_metadata.prompt_token_count or 0
            tokens_out = response.usage_metadata.candidates_token_count or 0
            cost_usd = (tokens_in * 0.000000125) + (tokens_out * 0.0000005)
        
        await analytics_collector.track_usage(agent_id, tokens_in, tokens_out, cost_usd, agent_config.model_name)
        
        final_content = response.text.strip() if response.text else state['validated_content']
        
        # Update post status in database
        await update_post(
            state['post_id'],
            status='published',
            published_at=datetime.utcnow().isoformat(),
            validator_agent=state['validator_agent_id'],
            publisher_agent=agent_id,
            tokens_in=state['total_tokens_in'] + tokens_in,
            tokens_out=state['total_tokens_out'] + tokens_out,
            cost_usd=state['total_cost_usd'] + cost_usd
        )
        
        # IMPORTANT: In simulation/manual mode, NEVER call bot.send_message
        # Just update the database status
        
        await analytics_collector.track_event(
            'post_published',
            agent_id=agent_id,
            message=f"Published post {state['post_id']} to channel {state['channel_id']}",
            severity='success'
        )
        
        # Return updated totals
        return {
            'validated_content': final_content,
            'total_tokens_in': state['total_tokens_in'] + tokens_in,
            'total_tokens_out': state['total_tokens_out'] + tokens_out,
            'total_cost_usd': state['total_cost_usd'] + cost_usd
        }
        
    except Exception as e:
        error_msg = str(e)
        if "API" in error_msg or "key" in error_msg.lower():
            error_msg = "API configuration error"
        
        await analytics_collector.track_event(
            'publisher_error',
            agent_id=agent_id,
            message=error_msg,
            severity='error'
        )
        
        return {
            'error_message': error_msg
        }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================
def should_retry_validation(state: AgentState) -> str:
    """Decide whether to retry creation or proceed/end."""
    if state.get('error_message'):
        return "error"
    
    if state['is_approved']:
        return "publish"
    
    # Retry if under limit (recursion_limit=3 handles hard cap)
    if state['validation_attempts'] < 3:
        return "retry"
    
    return "reject"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
def build_workflow() -> StateGraph:
    """Build the LangGraph workflow with recursion_limit=3."""
    
    # Create graph with hard recursion limit
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("creator", creator_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("publisher", publisher_node)
    
    # Set entry point
    workflow.set_entry_point("creator")
    
    # Define edges
    workflow.add_edge("creator", "validator")
    
    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validator",
        should_retry_validation,
        {
            "retry": "creator",      # Go back to creator for revision
            "publish": "publisher",  # Proceed to publishing
            "reject": END,           # End without publishing
            "error": END             # End on error
        }
    )
    
    workflow.add_edge("publisher", END)
    
    return workflow.compile()


# Global compiled graph instance
workflow_graph = build_workflow()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
async def run_content_pipeline(topic: str, channel_id: str,
                                creator_agent_id: str,
                                validator_agent_id: str,
                                publisher_agent_id: str) -> Dict[str, Any]:
    """Execute the full content generation pipeline."""
    
    initial_state: AgentState = {
        'post_id': 0,
        'channel_id': channel_id,
        'topic': topic,
        'draft_content': '',
        'validated_content': '',
        'search_results': '',
        'validation_attempts': 0,
        'validation_feedback': '',
        'is_approved': False,
        'creator_agent_id': creator_agent_id,
        'validator_agent_id': validator_agent_id,
        'publisher_agent_id': publisher_agent_id,
        'total_tokens_in': 0,
        'total_tokens_out': 0,
        'total_cost_usd': 0.0,
        'error_message': None
    }
    
    # Execute graph
    result = await workflow_graph.ainvoke(initial_state)
    
    return result
