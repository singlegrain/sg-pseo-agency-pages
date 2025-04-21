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
    "You are writing a service page for Single Grain, a full-service digital marketing agency. "
    "Adopt a confident, expert yet approachable tone: frame content as a partnership (we/you), focus on clear, data-driven benefits and actionable language, and use friendly but professional voice. "
    "Write concise, scannable copy with strong verbs and outcome-oriented statements. "
    "Avoid AI cliches, risky promises, or guarantees, and do not generate testimonials."
)

# --- SECTION STRUCTURE & EXAMPLES ---
EXAMPLES = {
    "hero": {
        "headline": "Rank Higher Everywhere with Single Grain's Search Everywhere Approach",
        "description": "We help you show up where your customers actually look—not just Google, but TikTok, Reddit, YouTube and AI chatbots. Our team builds visibility that drives real traffic and sales.",
        "cta": "Rank and Convert Everywhere"
    },
    "why_search_everywhere": {
        "title": "Why Your Brand Needs to Be Everywhere",
        "intro": "Google isn't the only search game in town anymore. Your customers look for solutions on Amazon, scroll TikTok for recommendations, and ask AI chatbots what to buy.\n\nAt Single Grain, we make sure you show up in all these places, not just one:",
        "channels": [
            {"title": "Search Engines", "description": "Google, Bing, Yahoo, Yandex, AI Overviews"},
            {"title": "Social Platforms", "description": "Meta (Facebook/Instagram), TikTok, LinkedIn, X (Twitter), Pinterest"},
            {"title": "AI Platforms", "description": "ChatGPT, Gemini, Perplexity, Claude, Copilot"},
            {"title": "Forums & Communities", "description": "Reddit, Quora, Stack Overflow"}
        ],
        "mini_cta": {"title": "Ready to be found everywhere?", "cta": "Let's Talk"}
    },
    "methodology": {
        "title": "How We Help You Show Up Everywhere",
        "intro": "We don't just tweak title tags and call it a day. Our team builds a complete visibility plan that puts your brand in front of real buyers wherever they search.",
        "steps": [
            {"title": "Find Your Best Keywords Everywhere", "description": "We dig into what your customers actually type, ask, and search for on every platform—not just Google. Our team finds the high-intent keywords and questions that drive real sales, whether people are typing them into search bars or asking AI assistants."},
            {"title": "Spot Gaps Your Competitors Missed", "description": "We track how your competitors perform across each channel, from SEO rankings to social engagement. This shows us exactly where they're weak and where you can quickly gain ground."},
            {"title": "Optimize for Each Platform", "description": "Different platforms need different approaches. We adapt your content for search engines, social feeds, and AI systems—improving your E-E-A-T signals, optimizing metadata, and creating AI-friendly content that gets recommended."},
            {"title": "Cross-Promote Your Wins", "description": "We turn one piece of winning content into many. Your top blog post becomes a LinkedIn carousel, a Twitter thread, and a YouTube script—reaching new audiences across platforms and building traffic from all directions."},
            {"title": "Track What Actually Works", "description": "Our reporting shows exactly where you're gaining visibility and what's driving real traffic. You'll see how you perform across search engines, social platforms, and AI tools so you know where to focus next."}
        ],
        "cta": {"title": "Ready to outrank your competition?", "cta": "Get Started"}
    },
    "services_overview": {
        "title": "Our Services",
        "description": "Our digital marketing services help you attract more visitors, convert them into leads, and turn those leads into sales.",
        "services": [
            {"title": "Content Strategy & Creation", "description": "Topic research, content briefs, expert writing, topic clusters, content refreshes, SERP-focused optimization, FAQs that answer real questions"},
            {"title": "Technical SEO", "description": "Site audits, speed optimization, keyword cannibalization fixes, schema markup, AI-friendly content structure"},
            {"title": "Link Building & Brand Authority", "description": "Guest posting, digital PR, resource link magnets, competitor link analysis, featured snippet targeting"},
            {"title": "Local SEO", "description": "Google Business Profile optimization, local citation building, location-focused content, review generation, local link acquisition"},
            {"title": "UX & Conversion Optimization", "description": "User journey mapping, A/B testing, form optimization, landing page improvements, analytics implementation"},
            {"title": "Performance Tracking", "description": "Custom dashboards, KPI monitoring, attribution reporting, competitive benchmarking, ROI analysis"},
            {"title": "Strategic Support", "description": "Ongoing consulting, strategy updates, quarterly business reviews, market trend analysis, growth planning"}
        ]
    },
    "differentiators": [
        {"title": "We See The Whole Picture", "description": "We don't just focus on Google. We help you appear wherever your customers look—search engines, social media, voice search, or AI chatbots. This complete approach drives more traffic from more sources."},
        {"title": "We Make Decisions Based on Data", "description": "No guesswork here. We track what works using analytics, user behavior patterns, and conversion signals. This means we put your budget where it actually drives results."},
        {"title": "We Build Custom Strategies", "description": "Your business isn't like everyone else's, so why would your marketing be? We create plans specifically for your goals, industry, and audience—not recycled tactics from other clients."},
        {"title": "We Work As Your Partner", "description": "We don't disappear after the kickoff call. You'll always know what we're working on, how campaigns are performing, and what's coming next. We're an extension of your team, not just a vendor."}
    ],
    "closing": {
        "title": "Ready to Be Found Everywhere?",
        "description": "Stop limiting yourself to Google. Your customers are searching everywhere, and you need to be there too. While your competitors focus on basic SEO tweaks, we'll help you build visibility across Google, TikTok, YouTube, Reddit and AI platforms. Let's get you in front of more buyers, wherever they're looking.",
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