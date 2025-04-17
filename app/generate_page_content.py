import os
import json
from dotenv import load_dotenv
from app.utils.perplexity_client import PerplexityClient
from app.utils.anthropic_client import AnthropicClient

# Load environment variables
load_dotenv()

# --- CONFIG ---
KEYWORD = "marketing automation agency"
POST_ID = "1234"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{POST_ID}.json")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are writing a service page for Single Grain, a digital marketing agency. "
    "The content must be high quality, professional, and vanilla (no risky or exaggerated claims). "
    "Do not make guarantees or risky promises. Maintain a confident, modern, and clear tone. "
    "Do not generate testimonials."
)

# --- SECTION STRUCTURE & EXAMPLES ---
EXAMPLES = {
    "hero": {
        "headline": "Supercharge Your SEO Single Grain's AI-Powered Search Everywhere Philosophy",
        "description": "Single Grain is an SEO Agency that helps you dominate organic visibility on every channel, from Google and Bing to TikTok and ChatGPT. We pioneer the use of cutting-edge AI and automation technologies.",
        "cta": "Rank and Convert Everywhere"
    },
    "why_search_everywhere": {
        "title": "Why 'Search Everywhere' is The Future of SEO",
        "intro": "Traditional SEO is no longer enough. People don't just look for information on Google; they're discovering brands, products, and services on Amazon, searching inside TikTok, using voice assistants like Gemini, and asking AI chatbots like DeepSeek for recommendations.\n\nAt Single Grain, we're innovating and ensuring your brand is visible and optimized across all digital touchpoints, including:",
        "channels": [
            {"title": "Search Engines", "description": "Google, Bing, Yahoo, Yandex, AI Overviews"},
            {"title": "Social Platforms", "description": "Meta (Facebook/Instagram), TikTok, LinkedIn, X (Twitter), Pinterest"},
            {"title": "AI Platforms (LLMs)", "description": "Manus, ChatGPT, Gemini, Perplexity, Claude, Grok"},
            {"title": "Forums & Communities", "description": "Reddit, Quora, Stack Overflow"}
        ],
        "mini_cta": {"title": "Ready to Elevate Your Creative? Let's Talk!", "cta": "Work With Us"}
    },
    "methodology": {
        "title": "How We Use 'Search Everywhere' to Supercharge Your Online Presence",
        "intro": "We don't just sprinkle keywords on a webpage. At Single Grain, Search Everywhere Optimization means embedding your brand in every corner of the digital landscape.",
        "steps": [
            {"title": "Keyword Research: The Right Keywords on the Right Platforms", "description": "We go beyond Google. Our team researches high-intent keywords and trending queries across each platform your audience uses — from traditional search engines and social platforms to AI chat tools and online communities. Especially with LLMs & AI taking a larger market share of searches each day, it's not just SEO anymore — it's search everywhere."},
            {"title": "Competitor Research per Platform", "description": "We analyze how your top competitors perform on each channel, breaking down their keyword wins, content strategies, and share of voice. By platform-specific benchmarking, we help you spot the white space they've missed — and where you can lead."},
            {"title": "Cross-Platform Optimizations", "description": "Once we have the data, we get to work. Whether it's rewriting AI-friendly content, optimizing social metadata, or improving E-E-A-T for search engines, we implement platform-specific improvements to increase discoverability and engagement."},
            {"title": "Cross-Channel Promotion", "description": "Our integrated promotion strategies ensure your content gets amplified across every touchpoint. We leverage channel synergies — like turning a top-ranking blog into viral LinkedIn posts or Reddit discussions — to drive traffic from all angles."},
            {"title": "Reporting: Curious How You Stack Up Everywhere?", "description": "Our Search Everywhere Reporting gives you a full-funnel view of your brand's visibility across search engines, social platforms, AI tools, and forums. From SEO to AI-generated responses to viral social trends, we deliver performance insights by platform and show you where to double down next."}
        ],
        "cta": {"title": "Ready to Skyrocket Your Growth?", "cta": "Work With Us"}
    },
    "services_overview": {
        "title": "Our Services",
        "description": "We offer a comprehensive suite of digital marketing services, each of our specialized solutions is designed to skyrocket your revenue and put you ahead of your competition.",
        "services": [
            {"title": "Content Strategy & Optimization", "description": "Content Strategy, Research & Brief Writing, Content Writing, Content Cluster Strategy, Existing Content Optimization & Pruning, Keyword Research, SERP Analysts with Niche Expertise to Refine Search Intent, FAQs Creation"},
            {"title": "Technical SEO", "description": "Keyword Gap Auditing, Cannibalization Analysis, NLP Optimization for AI Search"},
            {"title": "Link Building & Off-Site SEO", "description": "Featured Snippet / AI Overviews Targeting, Guest Blogging Opportunities, Digital PR & Outreach, Competitor Link Analysis, Authority Building"},
            {"title": "Local SEO", "description": "Google Business Profile Optimization, Local Citation Building, Local Keyword Targeting, Review Management Strategy, Local Link Building"},
            {"title": "UX & CRO", "description": "User Experience Audits, Conversion Rate Optimization, A/B Testing Strategy, User Journey Mapping"},
            {"title": "Reporting & Communications", "description": "Comprehensive SEO Reporting, KPI Tracking & Analysis, Regular Status Updates, Performance Insights"},
            {"title": "Long-Term Strategy & Support", "description": "Ongoing SEO Consulting, Strategy Refinement, Quarterly Business Reviews, Market Trend Analysis"}
        ]
    },
    "differentiators": [
        {"title": "Holistic, Omni-Channel Perspective", "description": "We believe SEO should encompass all the ways customers search. Whether they type on a laptop keyboard, speak into Siri, or ask ChatGPT, your brand needs to show up—and stand out."},
        {"title": "Data-Driven Decision Making", "description": "Our team of analysts, strategists, and AI specialists doesn't rely on guesswork. We leverage robust analytics, user behavior data, and real-time signals to identify high-impact actions."},
        {"title": "Custom Strategy & Execution", "description": "No cookie-cutter tactics. Every brand is unique, so we tailor an in-depth plan around your specific goals, industry, and target audience—then continually refine it for maximum ROI."},
        {"title": "Transparent Reporting & Collaboration", "description": "You always know what we're working on, how your campaigns are performing, and where new opportunities lie. We treat your brand like our own, operating as an extension of your marketing team."}
    ],
    "closing": {
        "title": "Ready to Win Across Every Search Channel?",
        "description": "If you're ready to move beyond outdated SEO tactics and start showing up where your audience actually searches, Single Grain is the partner to make it happen. While others are still focused on metadata tweaks and minor technical fixes, we're leveraging AI, automation, and platform-specific strategies to help you dominate across Google, TikTok, YouTube, Reddit, and beyond. Let's future-proof your visibility - everywhere search happens.",
        "cta": "Boost Rankings & Revenue"
    }
}

# --- MAIN GENERATION LOGIC ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p_client = PerplexityClient()
    a_client = AnthropicClient()

    # 1. Get knowledge backbone for the keyword
    print(f"Getting knowledge backbone for '{KEYWORD}'...")
    kb_result = p_client.query_with_search(
        prompt=KEYWORD,
        system_prompt=SYSTEM_PROMPT,
        search_context_size="medium",
        max_tokens=600
    )
    if not kb_result["success"] or not kb_result["content"]:
        raise RuntimeError(f"Failed to get knowledge backbone: {kb_result.get('error')}")
    knowledge_backbone = kb_result["content"].strip()

    # 2. Build the single prompt for all sections
    prompt = f"""
Here is some background information about the topic \"{KEYWORD}\":\n\n{knowledge_backbone}\n\nGenerate a complete service page for Single Grain, a digital marketing agency, about \"{KEYWORD}\".\nReturn a single JSON object with the following sections and structure (see examples):\n\n{json.dumps(EXAMPLES, indent=2)}\n\nEach section should follow the style and structure of the provided examples. Do not generate testimonials. Do not make risky or exaggerated claims.\n\nIMPORTANT: Use simple, clear language (aim for grade 11 reading level), but do not skip using marketing jargon where appropriate. Avoid typical AI cliches such as 'in today's world', 'in today's digital landscape', 'in today's fast-paced environment', and similar phrases. Do NOT wrap the JSON in a string or markdown code block. Do NOT include ```json or any markdown formatting. The output must be a valid JSON object only.\n"""

    # 3. Generate all sections at once using Anthropic (Claude) extended thinking
    print("Generating all sections in one call with Claude extended thinking...")
    response = a_client.query_with_extended_thinking(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=6000,
        thinking_budget=3000
    )
    # response["response"] is the final output
    if response and "response" in response and response["response"]:
        raw_response = response["response"].strip()
        try:
            # If the model still returns a code block, strip it
            if raw_response.startswith("```json"):
                import re
                raw_response = re.sub(r'^```json\\n?|```$', '', raw_response.strip(), flags=re.MULTILINE)
            content = json.loads(raw_response)
        except Exception as e:
            content = {"error": f"Failed to parse JSON: {str(e)}", "raw": response["response"]}
    else:
        content = {"error": f"[Error generating content: {response}]"}

    # 4. Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"keyword": KEYWORD, "knowledge_backbone": knowledge_backbone, "content": content}, f, indent=2, ensure_ascii=False)
    print(f"Content for post {POST_ID} written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 