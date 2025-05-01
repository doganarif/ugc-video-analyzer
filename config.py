import os
import json
from enum import Enum
from dotenv import load_dotenv

# Default configuration
DEFAULT_SHOT_INTERVAL = 10  # In seconds - reduced for shorter ad content
DEFAULT_FRAMES_PER_SECOND = 1
DEFAULT_TEMPERATURE = 0.5
RESIZE_OF_FRAMES = 4

# Database configuration
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/video_analytics"

# Define prompts as an enum for easier selection
class PromptType(str, Enum):
    GENERAL = "general"
    AUDIENCE = "audience"
    VISUAL = "visual"
    MESSAGING = "messaging"
    BRANDING = "branding"
    ENGAGEMENT = "engagement"
    TECHNICAL = "technical"
    CONVERSION = "conversion"

# System prompts dictionary
SYSTEM_PROMPTS = {
    PromptType.GENERAL: """You are an expert social media ad analyst with deep knowledge of digital marketing, advertising psychology, and visual communication. Analyze the frames from this video and provide:
1. Ad type and platform context (Instagram, TikTok, YouTube, etc.)
2. Key messaging and value proposition
3. Visual techniques and aesthetic choices
4. Narrative structure and storytelling approach
5. Emotional appeal and persuasion tactics
Be specific about elements that enhance brand appeal and potential viewer engagement.""",

    PromptType.AUDIENCE: """You are a target audience and consumer behavior specialist. From these sequential frames, identify:
1. The likely target demographic and psychographic profile
2. How the content speaks to audience pain points or desires
3. Cultural relevance and demographic appeal
4. How the content builds trust or credibility with the audience
5. Potential viewer reactions and emotional responses
Provide insightful observations about how effectively the content connects with its intended audience.""",

    PromptType.VISUAL: """You are a visual marketing expert specializing in social media creative analysis. From these video frames, provide detailed observations about:
1. Visual composition, color schemes, and brand aesthetics
2. Content authenticity and production quality
3. Visual hooks and attention-grabbing elements
4. Brand identity integration and visual consistency
5. Trendy visual elements or platform-specific optimizations
Explain how these visual elements enhance message effectiveness and brand perception.""",

    PromptType.MESSAGING: """You are a marketing messaging expert. Analyze these sequential frames to identify:
1. The primary and secondary message points
2. The clarity and effectiveness of the value proposition
3. The call-to-action strength and placement
4. Messaging hierarchy and information flow
5. Emotional triggers and persuasive language
Provide context for how effectively the messaging conveys benefits and motivates viewer action.""",

    PromptType.BRANDING: """You are a brand strategy specialist. From these video frames, identify:
1. Brand presence and logo usage
2. Brand voice and personality elements
3. Brand consistency with known identity (if recognizable)
4. Brand differentiation tactics
5. Brand trust signals and credibility markers
Analyze how effectively the content builds and reinforces brand identity.""",

    PromptType.ENGAGEMENT: """You are a social media engagement specialist who understands what drives shares, comments, and interactions. From these frames, analyze:
1. Hook effectiveness and opening impact
2. Elements that encourage viewer retention
3. Share triggers and viral potential
4. Comment-worthy moments or discussion starters
5. Elements that invite user interaction or participation
Provide detailed analysis of how the content is structured to maximize engagement metrics.""",

    PromptType.TECHNICAL: """You are a technical content production expert. From these video frames, provide precise analysis of:
1. Production quality and execution (amateur vs. professional)
2. Technical choices that impact viewer experience
3. Platform optimization elements (aspect ratio, length indicators, etc.)
4. Text overlay readability and placement
5. Sound design indicators (if detectable from visuals)
6. UGC authenticity markers vs. polished production elements
Explain how these technical choices impact the content's effectiveness for its intended platform.""",

    PromptType.CONVERSION: """You are a conversion optimization specialist. Examine these sequential frames and analyze:
1. Call-to-action clarity, placement and effectiveness
2. Friction points that might prevent viewer action
3. Trust elements and social proof integration
4. Urgency or scarcity triggers
5. Value proposition clarity at decision points
Explain how this content is constructed to drive specific viewer actions and conversions."""
}

# Default user prompt
USER_PROMPT = "These are the frames from the ad/UGC video. Analyze them according to your expertise."

# Load environment variables
load_dotenv(override=True)

# OpenAI GPT-4o configuration
AOAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AOAI_APIKEY = os.environ.get("AZURE_OPENAI_API_KEY")
AOAI_APIVERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AOAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

# Whisper configuration
WHISPER_ENDPOINT = os.environ.get("WHISPER_ENDPOINT")
WHISPER_APIKEY = os.environ.get("WHISPER_API_KEY")
WHISPER_APIVERSION = os.environ.get("WHISPER_API_VERSION")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_DEPLOYMENT_NAME")

# Helper functions to get configuration values
def get_openai_api_key():
    """Get the OpenAI API key from environment variables"""
    # First try to get the OpenAI API key directly
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # If not found, try to get the Azure OpenAI API key
    if not api_key:
        api_key = AOAI_APIKEY
        
    return api_key

def get_database_url():
    """Get the database connection URL from environment variables or use default"""
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL

# Chat model configuration
def get_chat_model():
    """Get the chat model name to use"""
    return os.environ.get("CHAT_MODEL", "gpt-4o") 