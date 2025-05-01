from typing import List, Optional
from pydantic import BaseModel, Field
from config import PromptType

# Pydantic models for structured output
class Person(BaseModel):
    role: str = Field(..., description="Person's role in the ad (influencer, actor, customer, etc.)")
    description: str = Field(..., description="Physical description of the person")
    actions: List[str] = Field(default_factory=list, description="Key actions performed by the person")
    emotional_tone: Optional[str] = Field(None, description="Emotional tone conveyed by the person")

class VisualDetails(BaseModel):
    composition: List[str] = Field(default_factory=list, description="Visual composition elements (framing, layout, etc.)")
    camera_movement: Optional[str] = Field(None, description="Description of camera movements")
    lighting: Optional[str] = Field(None, description="Lighting style and techniques")
    color_scheme: Optional[str] = Field(None, description="Description of color scheme and brand alignment")

class ContentAnalysis(BaseModel):
    setting: str = Field(..., description="Setting or context of the ad")
    platform: Optional[str] = Field(None, description="Apparent target platform(s)")
    message_purpose: Optional[str] = Field(None, description="Purpose of the content (educate, entertain, convert, etc.)")
    emotional_appeal: Optional[str] = Field(None, description="Overall emotional appeal of the content")

class TechnicalDetails(BaseModel):
    production_quality: Optional[str] = Field(None, description="Production quality assessment")
    platform_optimization: Optional[str] = Field(None, description="Platform-specific optimizations")
    text_elements: Optional[str] = Field(None, description="Text overlays and on-screen elements")
    notable_techniques: List[str] = Field(default_factory=list, description="Other notable technical aspects")

class AudienceAnalysis(BaseModel):
    target_demographics: List[str] = Field(default_factory=list, description="Target demographic groups")
    pain_points: List[str] = Field(default_factory=list, description="Audience pain points addressed")
    appeal_elements: List[str] = Field(default_factory=list, description="Elements designed to appeal to the audience")

class BrandAnalysis(BaseModel):
    brand_elements: List[str] = Field(default_factory=list, description="Visual brand elements present")
    brand_voice: Optional[str] = Field(None, description="Brand voice and personality conveyed")
    consistency: Optional[str] = Field(None, description="Brand consistency assessment")

class VideoAnalysis(BaseModel):
    video_name: str = Field(..., description="Name of the analyzed video file")
    segment_timeframe: Optional[str] = Field(None, description="Timeframe of the analyzed segment")
    analysis_type: PromptType = Field(..., description="Type of analysis performed")
    text_analysis: str = Field(..., description="Full text analysis from the AI")
    people: List[Person] = Field(default_factory=list, description="People identified in the content")
    visuals: VisualDetails = Field(default_factory=VisualDetails, description="Visual elements analysis")
    content_details: ContentAnalysis = Field(default_factory=ContentAnalysis, description="Content analysis details")
    technical_details: Optional[TechnicalDetails] = Field(None, description="Technical production details")
    audience_analysis: Optional[AudienceAnalysis] = Field(None, description="Target audience analysis")
    brand_analysis: Optional[BrandAnalysis] = Field(None, description="Brand elements analysis")
    call_to_action: Optional[str] = Field(None, description="Call to action identified")
    key_messages: List[str] = Field(default_factory=list, description="Key messages identified")
    transcription: Optional[str] = Field(None, description="Audio transcription if available")

    class Config:
        json_schema_extra = {
            "example": {
                "video_name": "product_ad.mp4",
                "segment_timeframe": "00:00:00 - 00:00:15",
                "analysis_type": "general",
                "text_analysis": "This appears to be a UGC-style product demonstration...",
                "people": [
                    {
                        "role": "Influencer",
                        "description": "Young woman in casual attire",
                        "actions": ["Demonstrating product", "Pointing to features"],
                        "emotional_tone": "Enthusiastic and genuine"
                    }
                ],
                "visuals": {
                    "composition": ["Close-up product shot", "Lifestyle context"],
                    "camera_movement": "Handheld selfie-style filming",
                    "lighting": "Natural indoor lighting",
                    "color_scheme": "Bright and vibrant with brand colors"
                },
                "content_details": {
                    "setting": "Home environment",
                    "platform": "Instagram Reels / TikTok",
                    "message_purpose": "Product demonstration and testimonial",
                    "emotional_appeal": "Authentic enthusiasm and relatability"
                }
            }
        } 