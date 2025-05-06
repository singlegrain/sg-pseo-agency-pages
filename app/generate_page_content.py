import os
import json
import re
from dotenv import load_dotenv
from collections import defaultdict
from app.utils.perplexity_client import PerplexityClient
from app.utils.anthropic_client import AnthropicClient

# Load environment variables
load_dotenv()

# --- ICON MAPPING ---
# Font Awesome icon mapping for highlight categories
ICON_MAPPING = {
    # Search related
    "search": "fa-magnifying-glass",
    "seo": "fa-magnifying-glass",
    "google": "fa-google",
    "bing": "fa-search",
    "yahoo": "fa-yahoo",
    "serp": "fa-list-ol",
    "ranking": "fa-ranking-star",
    
    # Social media related
    "social": "fa-share-nodes",
    "facebook": "fa-facebook",
    "instagram": "fa-instagram",
    "linkedin": "fa-linkedin",
    "twitter": "fa-twitter",
    "tiktok": "fa-tiktok",
    "pinterest": "fa-pinterest",
    "youtube": "fa-youtube",
    "reddit": "fa-reddit",
    
    # AI and tech related
    "ai": "fa-robot",
    "machine learning": "fa-brain",
    "automation": "fa-gears",
    "chatbot": "fa-message-bot",
    "analytics": "fa-chart-line",
    "data": "fa-database",
    "algorithm": "fa-code",
    
    # Marketing related
    "marketing": "fa-bullhorn",
    "email": "fa-envelope",
    "newsletter": "fa-newspaper",
    "campaign": "fa-bullseye",
    "lead": "fa-user-plus",
    "conversion": "fa-funnel-dollar",
    "sales": "fa-money-bill-trend-up",
    "roi": "fa-chart-pie",
    "growth": "fa-chart-line",
    "branding": "fa-copyright",
    
    # Content related
    "content": "fa-file-lines",
    "blog": "fa-blog",
    "article": "fa-newspaper",
    "video": "fa-video",
    "podcast": "fa-podcast",
    "webinar": "fa-users-rectangle",
    "infographic": "fa-chart-column",
    
    # Customer related
    "customer": "fa-users",
    "personalization": "fa-user-pen",
    "experience": "fa-face-smile",
    "engagement": "fa-hand-point-up",
    "loyalty": "fa-heart",
    "retention": "fa-user-check",
    "feedback": "fa-comments",
    
    # Business operations
    "strategy": "fa-chess",
    "planning": "fa-calendar",
    "management": "fa-tasks",
    "performance": "fa-gauge-high",
    "tracking": "fa-chart-line",
    "reporting": "fa-file-chart-column",
    "integration": "fa-plug",
    
    # E-commerce
    "ecommerce": "fa-cart-shopping",
    "product": "fa-box",
    "pricing": "fa-tag",
    "payment": "fa-credit-card",
    "checkout": "fa-cash-register",
    
    # Technology & platforms
    "mobile": "fa-mobile-screen",
    "website": "fa-globe",
    "app": "fa-mobile-screen-button",
    "software": "fa-laptop-code",
    "cloud": "fa-cloud",
    "api": "fa-code",
    "crm": "fa-address-book",
    "platform": "fa-server",
    
    # Default fallback
    "default": "fa-circle-info"
}

# Create a list of available Font Awesome icons for the AI to use
AVAILABLE_FONT_AWESOME_ICONS = [
    "fa-magnifying-glass", "fa-share-nodes", "fa-robot", "fa-comments",
    "fa-chart-line", "fa-envelope", "fa-bullhorn", "fa-brain",
    "fa-users", "fa-newspaper", "fa-file-lines", "fa-cog",
    "fa-user-check", "fa-arrows-to-circle", "fa-magnifying-glass-chart",
    "fa-gears", "fa-bullseye", "fa-user-plus", "fa-plug",
    "fa-server", "fa-globe", "fa-mobile-screen", "fa-database",
    "fa-chart-pie", "fa-cart-shopping", "fa-address-book", "fa-heart",
    "fa-money-bill-trend-up", "fa-bolt", "fa-shield-halved", 
    "fa-lightbulb", "fa-handshake", "fa-code", "fa-gauge-high"
]

# --- CONFIG ---
KEYWORDS_POSTS = [
    {"keyword": "adtech consulting agency", "post_id": "56767"},
    {"keyword": "affiliate marketing management agency", "post_id": "57583"},
    {"keyword": "agencia de contenidos ai agency", "post_id": "65466"},
    {"keyword": "agencia de gestion de canales de youtube agency", "post_id": "65464"},
    {"keyword": "agencia de marketing reddit agency", "post_id": "65516"},
    {"keyword": "ai acquisition agency", "post_id": "60003"},
    {"keyword": "ai business consulting/l/san francisco", "post_id": "51536"},
    {"keyword": "ai chatbot agency", "post_id": "56758"},
    {"keyword": "ai content agency", "post_id": "54911"},
    {"keyword": "ai facebook agency", "post_id": "60027"},
    {"keyword": "ai influencer agency", "post_id": "59840"},
    {"keyword": "ai instagram services agency", "post_id": "60029"},
    {"keyword": "ai marketing agency", "post_id": "56767"},
    {"keyword": "ai media buying agency", "post_id": "60122"},
    {"keyword": "ai seo agency", "post_id": "56767"},
    {"keyword": "ai twitter agency", "post_id": "60028"},
    {"keyword": "ai youtube agency", "post_id": "56767"},
    {"keyword": "amazon marketing agency", "post_id": "56767"},
    {"keyword": "app marketing agency", "post_id": "56767"},
    {"keyword": "brand collaboration agency", "post_id": "57319"},
    {"keyword": "branding agency", "post_id": "56767"},
    {"keyword": "chatbot agency", "post_id": "56758"},
    {"keyword": "content agency", "post_id": "56767"},
    {"keyword": "content marketing agency", "post_id": "56767"},
    {"keyword": "content repurposing agency", "post_id": "50319"},
    {"keyword": "conversion rate optimization agency", "post_id": "56767"},
    {"keyword": "crypto twitter marketing agency", "post_id": "58266"},
    {"keyword": "crm agency", "post_id": "56767"},
    {"keyword": "customer retention agency", "post_id": "56767"},
    {"keyword": "data analytics agency", "post_id": "56767"},
    {"keyword": "digital marketing agency", "post_id": "56767"},
    {"keyword": "discord management agency", "post_id": "58272"},
    {"keyword": "edtech marketing agency", "post_id": "55937"},
    {"keyword": "email marketing agency", "post_id": "56767"},
    {"keyword": "facebook marketing agency", "post_id": "56767"},
    {"keyword": "google ads agency", "post_id": "56767"},
    {"keyword": "google my business agency", "post_id": "56365"},
    {"keyword": "growth hacking agency", "post_id": "56767"},
    {"keyword": "growth marketing agency", "post_id": "56767"},
    {"keyword": "go to market strategy agency", "post_id": "50337"},
    {"keyword": "influencer marketing agency", "post_id": "56767"},
    {"keyword": "instagram marketing agency", "post_id": "56767"},
    {"keyword": "lead generation agency", "post_id": "56767"},
    {"keyword": "lead generation agency for crypto", "post_id": "53299"},
    {"keyword": "linkedin management agency", "post_id": "46612"},
    {"keyword": "local seo agency", "post_id": "56767"},
    {"keyword": "link building services agency", "post_id": "46324"},
    {"keyword": "linkedin marketing agency", "post_id": "56767"},
    {"keyword": "mobile marketing agency", "post_id": "56767"},
    {"keyword": "omnichannel marketing agency", "post_id": "46528"},
    {"keyword": "organic seo agency", "post_id": "56767"},
    {"keyword": "paid search agency", "post_id": "56767"},
    {"keyword": "pay per click agency", "post_id": "56767"},
    {"keyword": "pay per lead agency", "post_id": "46592"},
    {"keyword": "performance based marketing agency", "post_id": "56242"},
    {"keyword": "performance marketing agency", "post_id": "56767"},
    {"keyword": "photography advertising agency", "post_id": "57669"},
    {"keyword": "podcast growth agency", "post_id": "58309"},
    {"keyword": "programmatic advertising agency", "post_id": "56767"},
    {"keyword": "programmatic seo agency", "post_id": "47607"},
    {"keyword": "reddit marketing agency", "post_id": "46579"},
    {"keyword": "search engine marketing agency", "post_id": "56767"},
    {"keyword": "search engine optimization agency", "post_id": "56767"},
    {"keyword": "seo agency", "post_id": "56767"},
    {"keyword": "seo consulting agency", "post_id": "56767"},
    {"keyword": "seo services agency", "post_id": "56767"},
    {"keyword": "short form video agency", "post_id": "50315"},
    {"keyword": "social media marketing agency", "post_id": "56767"},
    {"keyword": "sustainable marketing services agency", "post_id": "57310"},
    {"keyword": "tiktok growth services agency", "post_id": "61243"},
    {"keyword": "tiktok marketing agency", "post_id": "56767"},
    {"keyword": "twitter marketing agency", "post_id": "56767"},
    {"keyword": "user retention agency", "post_id": "50331"},
    {"keyword": "video marketing agency", "post_id": "56767"},
    {"keyword": "voice search agency", "post_id": "56767"},
    {"keyword": "web design agency", "post_id": "56767"},
    {"keyword": "web development agency", "post_id": "56767"},
    {"keyword": "whatsapp business consulting agency", "post_id": "57767"},
    {"keyword": "whatsapp marketing agency", "post_id": "57890"},
    {"keyword": "youtube channel management agency", "post_id": "58241"},
    {"keyword": "youtube consulting agency", "post_id": "55719"},
    {"keyword": "youtube growth agency", "post_id": "58099"},
    {"keyword": "youtube marketing agency", "post_id": "56767"},
    {"keyword": "youtube video production agency", "post_id": "56767"}
]
OUTPUT_DIR = "output"

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are writing a service page for Single Grain, a full-service digital marketing agency. "
    "Adopt a confident, expert yet approachable tone: frame content as a partnership (we/you), focus on clear, data-driven benefits and actionable language, and use friendly but professional voice. "
    "Write concise, scannable copy with strong verbs and outcome-oriented statements. "
    "Avoid AI cliches, risky promises, or guarantees, and do not generate testimonials."
)

# --- RESEARCH PROMPT ---
RESEARCH_PROMPT = (
    "Research and summarize the most important, up-to-date facts, trends, challenges, and opportunities about the topic: '{keyword}'. "
    "Focus on actionable insights, statistics, and current best practices relevant to digital marketing decision makers. "
    "Do not generate marketing copy, sales language, or brand-specific content."
)

# --- SECTION STRUCTURE & EXAMPLES ---
EXAMPLES = {
    "hero": {
        "headline": "<span>Rank Higher Everywhere</span> with Single Grain's Search Everywhere Approach",
        "description": "We help you show up where your customers actually look—not just Google, but TikTok, Reddit, YouTube and AI chatbots. Our team builds visibility that drives real traffic and sales.",
        "cta": "Rank and Convert Everywhere"
    },
    "why_us": {
        "title": "Why Your Brand Needs to <strong>Be Everywhere</strong>",
        "intro": "<p>Google isn't the only search game in town anymore. Your customers look for solutions on Amazon, scroll TikTok for recommendations, and ask AI chatbots what to buy.</p><p>At Single Grain, we make sure you show up in all these places, not just one:</p>",
        "highlights": [
            {"title": "Search Engines", "description": "Google, Bing, Yahoo, Yandex, AI Overviews", "icon": "fa-magnifying-glass"},
            {"title": "Social Platforms", "description": "Meta (Facebook/Instagram), TikTok, LinkedIn, X (Twitter), Pinterest", "icon": "fa-share-nodes"},
            {"title": "AI Platforms", "description": "ChatGPT, Gemini, Perplexity, Claude, Copilot", "icon": "fa-robot"},
            {"title": "Forums & Communities", "description": "Reddit, Quora, Stack Overflow", "icon": "fa-comments"}
        ],
        "mini_cta": {"title": "Ready to <span>be found</span> everywhere?", "cta": "Let's Talk"}
    },
    "methodology": {
        "title": "How We Help You <strong>Show Up Everywhere</strong>",
        "intro": "We don't just tweak title tags and call it a day. Our team builds a complete visibility plan that puts your brand in front of real buyers wherever they search.",
        "steps": [
            {"title": "Find Your Best Keywords Everywhere", "description": "We dig into what your customers actually type, ask, and search for on every platform—not just Google. Our team finds the high-intent keywords and questions that drive real sales, whether people are typing them into search bars or asking AI assistants."},
            {"title": "Spot Gaps Your Competitors Missed", "description": "We track how your competitors perform across each channel, from SEO rankings to social engagement. This shows us exactly where they're weak and where you can quickly gain ground."},
            {"title": "Optimize for Each Platform", "description": "Different platforms need different approaches. We adapt your content for search engines, social feeds, and AI systems—improving your E-E-A-T signals, optimizing metadata, and creating AI-friendly content that gets recommended."},
            {"title": "Cross-Promote Your Wins", "description": "We turn one piece of winning content into many. Your top blog post becomes a LinkedIn carousel, a Twitter thread, and a YouTube script—reaching new audiences across platforms and building traffic from all directions."},
            {"title": "Track What Actually Works", "description": "Our reporting shows exactly where you're gaining visibility and what's driving real traffic. You'll see how you perform across search engines, social platforms, and AI tools so you know where to focus next."}
        ],
        "cta": {"title": "Ready to <span class='orange'>outrank</span> your competition?", "cta": "Get Started"}
    },
    "services_overview": {
        "title": "Our Services",
        "description": "Our digital marketing services help you attract more visitors, convert them into leads, and turn those leads into sales.",
        "services": [
            {"title": "Content Strategy & Creation", "description": [
                "Topic research",
                "Content briefs",
                "Expert writing",
                "Topic clusters",
                "Content refreshes",
                "SERP-focused optimization",
                "FAQs that answer real questions"
            ]},
            {"title": "Technical SEO", "description": [
                "Site audits",
                "Speed optimization",
                "Keyword cannibalization fixes",
                "Schema markup",
                "AI-friendly content structure"
            ]},
            {"title": "Link Building & Brand Authority", "description": [
                "Guest posting",
                "Digital PR",
                "Resource link magnets",
                "Competitor link analysis",
                "Featured snippet targeting"
            ]},
            {"title": "Local SEO", "description": [
                "Google Business Profile optimization",
                "Local citation building",
                "Location-focused content",
                "Review generation",
                "Local link acquisition"
            ]},
            {"title": "UX & Conversion Optimization", "description": [
                "User journey mapping",
                "A/B testing",
                "Form optimization",
                "Landing page improvements",
                "Analytics implementation"
            ]},
            {"title": "Performance Tracking", "description": [
                "Custom dashboards",
                "KPI monitoring",
                "Attribution reporting",
                "Competitive benchmarking",
                "ROI analysis"
            ]},
            {"title": "Strategic Support", "description": [
                "Ongoing consulting",
                "Strategy updates",
                "Quarterly business reviews",
                "Market trend analysis",
                "Growth planning"
            ]}
        ]
    },
    "differentiators": {
        "title": "The Single Grain Difference: <span>What a Great SEO Agency as a Partner Looks Like</span>",
        "items": [
            {"title": "We See The Whole Picture", "description": "We don't just focus on Google. We help you appear wherever your customers look—search engines, social media, voice search, or AI chatbots. This complete approach drives more traffic from more sources."},
            {"title": "We Make Decisions Based on Data", "description": "No guesswork here. We track what works using analytics, user behavior patterns, and conversion signals. This means we put your budget where it actually drives results."},
            {"title": "We Build Custom Strategies", "description": "Your business isn't like everyone else's, so why would your marketing be? We create plans specifically for your goals, industry, and audience—not recycled tactics from other clients."},
            {"title": "We Work As Your Partner", "description": "We don't disappear after the kickoff call. You'll always know what we're working on, how campaigns are performing, and what's coming next. We're an extension of your team, not just a vendor."}
        ]
    },
    "closing": {
        "title": "Ready to Be <span class='orange'>Found Everywhere</span>?",
        "description": "Stop limiting yourself to Google. Your customers are searching everywhere, and you need to be there too. While your competitors focus on basic SEO tweaks, we'll help you build visibility across Google, TikTok, YouTube, Reddit and AI platforms. Let's get you in front of more buyers, wherever they're looking.",
        "subscribe": {
            "title": "Tired of one-dimensional, outdated SEO agencies?",
            "description": "Let us show you how we can level up your results.",
            "bottom": "SEO is dead, long live SEVO (search everywhere optimization)."
        },
        "cta": "Boost Rankings & Revenue"
    }
}

# --- MAIN GENERATION LOGIC ---
def generate_page_content(keyword, post_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p_client = PerplexityClient()
    a_client = AnthropicClient()

    # 1. Get knowledge backbone for the keyword
    print(f"Getting knowledge backbone for '{keyword}'...")
    research_prompt = RESEARCH_PROMPT.format(keyword=keyword)
    kb_result = p_client.query_with_search(
        prompt=research_prompt,
        system_prompt=None,
        search_context_size="high",
        max_tokens=1000
    )
    if not kb_result["success"] or not kb_result["content"]:
        raise RuntimeError(f"Failed to get knowledge backbone: {kb_result.get('error')}")
    knowledge_backbone = kb_result["content"].strip()

    # 2. Build the single prompt for all sections
    prompt = f"""
Here is some background information about the topic \"{keyword}\":\n\n```{knowledge_backbone}```\n\n
Generate a complete service page for Single Grain about \"{keyword}\" using THE EXACT STRUCTURE AND FORMATTING shown in the example below. Your task is to fill in the blanks with new content while keeping ALL formatting elements intact.\n\n{json.dumps(EXAMPLES, indent=2)}\n\n
CRITICAL INSTRUCTIONS:
1. Treat the example as an exact template - copy ALL HTML tags (<span>, <strong>, <p>, etc.) exactly as shown
2. Match the exact writing style, sentence structure and length for each section
3. Maintain the same level of detail in each section as shown in the example
4. Keep all bullet formats and list structures exactly as demonstrated
5. Keep the same formatting for CTAs, headlines, and descriptive text
6. Preserve all special text formatting shown (bold, spans with classes, paragraph tags)
7. Your output must be valid JSON with no markdown formatting or code blocks
8. DO NOT wrap your response in ```json code blocks. Just return a raw JSON object.

ICON SELECTION:
For the "highlights" section in "why_us", choose appropriate Font Awesome icons from this list:
{json.dumps(AVAILABLE_FONT_AWESOME_ICONS[:30])}

Assign an appropriate icon from the list to each highlight based on its content and theme.

Use simple, clear language (grade 10-11 reading level) with marketing terminology where appropriate. Avoid AI cliches like 'in today's world', 'digital landscape', 'fast-paced environment'. DO NOT include testimonials or make risky claims.\n"""

    # 3. Generate all sections at once using Anthropic (Claude) extended thinking
    print(f"Generating all sections in one call with Claude extended thinking for '{keyword}'...")
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
            # More robust handling of code blocks - handle both ```json and ``` cases
            if "```" in raw_response:
                # Remove any language specifier (like ```json) and any starting/ending backticks
                raw_response = re.sub(r'```(?:json)?\n?|\n?```', '', raw_response)
                # Further cleanup to ensure we have valid JSON
                raw_response = raw_response.strip()
            content = json.loads(raw_response)
            
            # Post-process the content to add icons to highlights if they don't exist
            print("\nChecking and assigning icons to highlights:")
            if "why_us" in content and "highlights" in content["why_us"]:
                for i, highlight in enumerate(content["why_us"]["highlights"]):
                    print(f"  Highlight {i+1}: '{highlight.get('title', 'Unknown')}'")
                    # Log the icon that was chosen by the AI
                    print(f"    - Icon: {highlight.get('icon', 'None')}")
                    
                    # If no icon was assigned, provide a default one
                    if "icon" not in highlight:
                        print(f"    - No icon found, assigning default...")
                        highlight["icon"] = "fa-circle-info"
                        print(f"    - Assigned default icon: {highlight['icon']}")
            else:
                print("  No highlights section found to assign icons to.")
            
        except Exception as e:
            content = {"error": f"Failed to parse JSON: {str(e)}", "raw": response["response"]}
    else:
        content = {"error": f"[Error generating content: {response}]"}

    # 4. Save output
    output_file = os.path.join(OUTPUT_DIR, f"{post_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"keyword": keyword, "knowledge_backbone": knowledge_backbone, "content": content}, f, indent=2, ensure_ascii=False)
    print(f"Content for post {post_id} written to {output_file}")

def main():
    for item in KEYWORDS_POSTS:
        keyword = item["keyword"]
        post_id = item["post_id"]
        output_file = os.path.join(OUTPUT_DIR, f"{post_id}.json")
        if os.path.exists(output_file):
            print(f"Skipping {keyword} (post_id: {post_id}) - output already exists.")
            continue
        try:
            generate_page_content(keyword, post_id)
        except Exception as e:
            print(f"Error processing '{keyword}' (post_id: {post_id}): {e}")

if __name__ == "__main__":
    main()
